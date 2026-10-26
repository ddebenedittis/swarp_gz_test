"""Run Push-T episodes in Gazebo and score them exactly as swarp does.

Per episode:

1. stop the policies (``/episode`` = -1, they publish zero twists) and let the robots stop;
2. draw the initial state like swarp's reset at full difficulty (:mod:`pusht_policy.task`)
   and teleport robots, T and goal marker with :meth:`GzWorld.set_poses_at_rest`;
3. wait until every robot's scan and the T's pose postdate the teleport; with
   ``initialpose`` (a localizer such as AMCL behind ``/robot_i/pose``), publish each
   robot's true pose on ``/robot_i/initialpose`` and wait until ``/robot_i/pose`` agrees
   with the truth (``localize_tol``), as an operator re-initializes AMCL after placing
   the robots;
4. publish ``/goal`` and ``/episode`` = id, which starts the policies;
5. every ``dt`` of sim time, score the T pose from ``/tee/gt_pose``: success when
   ``|p - g| < goal_tolerance`` and ``|wrap(th - g_th)| < angle_tolerance`` (swarp's
   ``done``), timeout after ``max_steps * dt``; with a settled hold (the checkpoint's
   ``hold_steps``, or the ``hold_steps``/``hold_settle`` parameters) success needs the T on goal
   for ``hold_steps`` steps in a row with the T and every robot slower than the ``hold_settle``
   thresholds (``/<body>/gt_vel``), and ``first_on_goal`` keeps the first pass's time;
6. append one row to ``<out_dir>/<run_name>/episodes.csv``.

Every pose this node reads is ground truth (``/<body>/gt_pose``), whatever the policy sees.
The draws come from one ``numpy`` generator seeded with ``seed``. The process holds no
gz-transport subscription (the Python bindings would stall the ``set_pose`` replies).
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32

from pusht_gz.gz_reset import GzWorld
from pusht_policy.policy_node import LATCHED, stamp_s, yaw_of
from pusht_policy.task import Task, sample_episode, wrap


class EpisodeManager(Node):
    def __init__(self):
        super().__init__("episode_manager")
        ckpt = self.declare_parameter("checkpoint", "").value
        self.episodes = int(self.declare_parameter("episodes", 10).value)
        self.seed = int(self.declare_parameter("seed", 0).value)
        out_dir = self.declare_parameter("out_dir", "out").value
        run_name = self.declare_parameter("run_name", "").value or f"eval_s{self.seed}"
        self.settle = float(self.declare_parameter("settle", 0.3).value)
        self.stop_time = float(self.declare_parameter("stop_time", 0.5).value)
        self.stall_timeout = float(self.declare_parameter("stall_timeout", 60.0).value)
        world = self.declare_parameter("world", "pusht").value
        self.initialpose = bool(self.declare_parameter("initialpose", False).value)
        # > 0: the episode fails as soon as the T centroid is past this on x or y (no T walls
        # in Gazebo; swarp's recipes from L8 on terminate there too). 0: run to the timeout.
        self.tee_out_of_bounds = float(self.declare_parameter("tee_out_of_bounds", 0.0).value)
        self.localize_tol = float(self.declare_parameter("localize_tol", 0.03).value)
        self.localize_timeout = float(self.declare_parameter("localize_timeout", 5.0).value)
        if not ckpt:
            raise ValueError("parameter 'checkpoint' is required")

        # The arena of a checkpoint trained on varying ones (its half side [m]; 0: the default).
        self.task = Task.from_checkpoint(ckpt, float(self.declare_parameter("arena_half", 0.0).value))
        # The settled hold to score with: -1 / "" take the checkpoint's.
        hold = int(self.declare_parameter("hold_steps", -1).value)
        settle = str(self.declare_parameter("hold_settle", "").value)
        self.hold_steps = self.task.hold_steps if hold < 0 else hold
        self.hold_settle = tuple(map(float, settle.split(","))) if settle else self.task.settle
        if len(self.hold_settle) != 4:
            raise ValueError(f"hold_settle takes 4 comma-separated thresholds, got {settle!r}")
        self.robots = [f"robot_{j}" for j in range(1, self.task.n_agents + 1)]
        self.rng = np.random.default_rng(self.seed)
        self.gz = GzWorld(world)
        self.run_dir = os.path.join(out_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_path = os.path.join(self.run_dir, "episodes.csv")

        self.pose: dict[str, tuple[float, float, float, float]] = {}  # body -> (t, x, y, yaw)
        self.scan_t: dict[str, float] = {}
        for body in self.robots + ["tee"]:
            self.create_subscription(PoseStamped, f"/{body}/gt_pose", lambda m, b=body: self._on_pose(b, m), 20)
        self.vel: dict[str, tuple[float, float]] = {}  # body -> (speed, |yaw rate|), world frame
        if self.hold_steps > 0:
            for body in self.robots + ["tee"]:
                self.create_subscription(TwistStamped, f"/{body}/gt_vel", lambda m, b=body: self._on_vel(b, m), 20)
        self.est: dict[str, tuple[float, float, float, float]] = {}  # the policy's /robot_i/pose
        self.init_pubs = {}
        if self.initialpose:
            for r in self.robots:
                self.create_subscription(
                    PoseStamped, f"/{r}/pose", lambda m, r=r: self._on_pose(r, m, self.est), 20
                )
                self.init_pubs[r] = self.create_publisher(PoseWithCovarianceStamped, f"/{r}/initialpose", 10)
        for r in self.robots:
            self.create_subscription(
                LaserScan, f"/{r}/scan", lambda m, r=r: self.scan_t.__setitem__(r, stamp_s(m.header.stamp)),
                qos_profile_sensor_data,
            )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal", LATCHED)
        self.episode_pub = self.create_publisher(Int32, "/episode", LATCHED)
        t = self.task
        self.get_logger().info(
            f"{t.scenario}: {t.n_agents} robots, dt {t.dt}, {t.max_steps} steps, tolerances "
            f"{t.goal_tolerance} m / {t.angle_tolerance} rad, arena +-{t.arena_half}, "
            f"T radius {t.tee_radius:.3f}; {self.episodes} episodes, seed {self.seed} -> {self.csv_path}"
        )

    # ------------------------------------------------------------------ helpers

    def _on_pose(self, body: str, m: PoseStamped, into: dict | None = None) -> None:
        p = m.pose
        into = self.pose if into is None else into
        into[body] = (stamp_s(m.header.stamp), p.position.x, p.position.y, yaw_of(p.orientation))

    def _on_vel(self, body: str, m: TwistStamped) -> None:
        v = m.twist
        self.vel[body] = (math.hypot(v.linear.x, v.linear.y), abs(v.angular.z))

    def settled(self) -> bool:
        vt, wt, vr, wr = self.hold_settle
        if not all(b in self.vel for b in self.robots + ["tee"]):
            return False
        return (self.vel["tee"][0] < vt and self.vel["tee"][1] < wt
                and all(self.vel[r][0] < vr and self.vel[r][1] < wr for r in self.robots))

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_until(self, pred, timeout: float, what: str) -> None:
        """Spin until ``pred()``; ``timeout`` is wall time."""
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out after {timeout}s waiting for {what}")
            rclpy.spin_once(self, timeout_sec=0.01)

    def wait_sim(self, dt: float) -> None:
        t0 = self.now()
        self.spin_until(lambda: self.now() - t0 >= dt, self.stall_timeout, f"{dt}s of sim time")

    def set_episode(self, ep: int) -> None:
        self.episode_pub.publish(Int32(data=ep))

    # ------------------------------------------------------------------ episodes

    def wait_ready(self) -> None:
        self.set_episode(-1)
        self.spin_until(lambda: self.now() > 0.0, 120.0, "/clock")
        self.spin_until(
            lambda: all(b in self.pose for b in self.robots + ["tee"]) and len(self.scan_t) == len(self.robots),
            120.0, "poses and scans",
        )
        self.spin_until(
            lambda: self.episode_pub.get_subscription_count() >= len(self.robots), 120.0,
            f"{len(self.robots)} policy nodes on /episode",
        )
        if self.initialpose:
            # Every localizer must be up and publishing before the first episode: a robot
            # without a pose estimate would never be driven (a node that never came up under
            # load). Failing here lets eval_shards restart the shard.
            self._send_initial_poses()
            self.spin_until(lambda: all(r in self.est for r in self.robots), 120.0,
                            "pose estimates from every localizer")

    def teleport(self, state: dict) -> None:
        poses = {r: p for r, p in zip(self.robots, state["robots"], strict=True)}
        poses["tee"] = state["tee"]
        poses["goal"] = state["goal"]
        for attempt in range(3):
            if not self.gz.set_poses_at_rest(poses, self.settle):
                self.get_logger().warn(f"set_poses_at_rest failed (attempt {attempt})")
                continue
            # Messages stamped after the second teleport (the node's own clock is stale: it
            # did not spin while GzWorld waited).
            t_tp = (self.gz.last_sim_time() or self.now()) + 0.05
            self.spin_until(
                lambda: all(self.scan_t.get(r, -1.0) > t_tp for r in self.robots)
                and all(self.pose[b][0] > t_tp for b in self.robots + ["tee"]),
                self.stall_timeout, "scans and poses after the teleport",
            )
            err = max(
                math.hypot(self.pose[b][1] - poses[b][0], self.pose[b][2] - poses[b][1])
                for b in self.robots + ["tee"]
            )
            if err < 0.02:
                if self.initialpose:
                    self.localize()
                return
            self.get_logger().warn(f"teleport off by {err:.3f} m (attempt {attempt}), retrying")
        raise RuntimeError("could not teleport the bodies to the episode's initial state")

    def localize(self, attempts: int = 3) -> None:
        """Re-initialize the localizers at the true poses and wait until they agree; an
        initial pose AMCL took at a bad moment (a TF lookup into the future) is sent again."""
        for attempt in range(attempts):
            self._send_initial_poses()
            if self._wait_localized(self.localize_timeout):
                return
            self.get_logger().warn(f"localizers off the initial poses (attempt {attempt + 1}), re-sending")
        self.get_logger().warn("localizers did not converge on the initial poses; starting anyway")

    def _send_initial_poses(self) -> None:
        for r in self.robots:
            _, x, y, th = self.pose[r]
            m = PoseWithCovarianceStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "map"
            m.pose.pose.position.x, m.pose.pose.position.y = x, y
            m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
            cov = [0.0] * 36
            cov[0] = cov[7] = 0.02**2
            cov[35] = 0.02**2
            m.pose.covariance = cov
            self.init_pubs[r].publish(m)

    def _wait_localized(self, timeout: float) -> bool:
        def agree() -> bool:
            for r in self.robots:
                e, g = self.est.get(r), self.pose[r]
                if e is None or e[0] < g[0] - 0.1:
                    return False
                if math.hypot(e[1] - g[1], e[2] - g[2]) > self.localize_tol or abs(wrap(e[3] - g[3])) > 0.05:
                    return False
            return True

        t0 = self.now()
        while not agree():
            if self.now() - t0 > timeout:
                return False
            rclpy.spin_once(self, timeout_sec=0.01)
        return True

    def tee_error(self, goal) -> tuple[float, float]:
        _, x, y, th = self.pose["tee"]
        return math.hypot(x - goal[0], y - goal[1]), abs(wrap(th - goal[2]))

    def run_episode(self, ep: int) -> dict:
        t = self.task
        state, rejected = sample_episode(t, self.rng)
        self.set_episode(-1)
        self.wait_sim(self.stop_time)
        self.teleport(state)

        goal = state["goal"]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y = goal[0], goal[1]
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(goal[2] / 2), math.cos(goal[2] / 2)
        self.goal_pub.publish(msg)
        self.set_episode(ep)
        t0 = self.now()
        wall0 = time.monotonic()

        # Score at the control rate, like swarp's done after each step.
        k, success, end = 0, False, "timeout"
        held, first_on_goal = 0, None
        min_pos = math.inf
        pos_err = ang_err = math.nan
        horizon = t.max_steps * t.dt
        while True:
            k += 1
            next_t = t0 + k * t.dt
            self.spin_until(lambda: self.pose["tee"][0] >= next_t, self.stall_timeout, "T pose")
            pos_err, ang_err = self.tee_error(goal)
            min_pos = min(min_pos, pos_err)
            on_goal = pos_err < t.goal_tolerance and ang_err < t.angle_tolerance
            if on_goal and first_on_goal is None:
                first_on_goal = round(k * t.dt, 3)
            held = held + 1 if on_goal and self.settled() else 0
            if on_goal and held >= self.hold_steps:
                success, end = True, "solved"
                break
            _, tx, ty, _ = self.pose["tee"]
            if self.tee_out_of_bounds > 0 and max(abs(tx), abs(ty)) > self.tee_out_of_bounds:
                end = "out_of_bounds"
                break
            if k >= t.max_steps:
                break
        self.set_episode(-1)
        elapsed = min(k * t.dt, horizon)
        row = {
            "seed": self.seed, "episode": ep, "success": int(success), "end": end, "time": round(elapsed, 3),
            "steps": k, "final_pos_err": round(pos_err, 4), "final_ang_err": round(ang_err, 4),
            "min_pos_err": round(min_pos, 4), "rejected_draws": rejected,
            "hold_steps": self.hold_steps, "first_on_goal": "" if first_on_goal is None else first_on_goal,
            "wall_s": round(time.monotonic() - wall0, 2),
        }
        for name, p in [("tee", state["tee"]), ("goal", goal)] + list(zip(self.robots, state["robots"], strict=True)):
            row.update({f"{name}_x": round(p[0], 4), f"{name}_y": round(p[1], 4), f"{name}_yaw": round(p[2], 4)})
        return row

    def run(self) -> None:
        self.wait_ready()
        new = not os.path.exists(self.csv_path)
        solved = 0
        with open(self.csv_path, "a", newline="") as f:
            writer = None
            for ep in range(self.episodes):
                row = self.run_episode(ep)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        writer.writeheader()
                writer.writerow(row)
                f.flush()
                solved += row["success"]
                self.get_logger().info(
                    f"episode {ep}: {'SOLVED' if row['success'] else 'timeout'} in {row['time']:.1f}s sim "
                    f"({row['wall_s']:.0f}s wall), final err {row['final_pos_err']:.3f} m / "
                    f"{row['final_ang_err']:.3f} rad, min {row['min_pos_err']:.3f} m; "
                    f"{solved}/{ep + 1} solved"
                )
        self.get_logger().info(f"done: {solved}/{self.episodes} solved -> {self.csv_path}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EpisodeManager()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.set_episode(-1)
        node.destroy_node()
        rclpy.try_shutdown()
