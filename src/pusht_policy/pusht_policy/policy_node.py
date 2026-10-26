"""Run one robot's swarp ``pusht_lidar`` policy on ROS topics.

One node per robot (``robot_index`` i, 1-based, drives ``robot_i``), as on the real fleet:

* in: its own ``/robot_i/scan``; every robot's ``/robot_j/pose`` and ``/robot_j/vel``
  (world frame, from Gazebo ground truth or mocap); ``/goal``; ``/episode``
  (``std_msgs/Int32``: the running episode's id, ``-1`` while stopped);
* out: ``/robot_i/cmd_vel``;
* with a delayed-comm checkpoint (swarp ``--comm``), out ``/robot_i/msg`` and in every
  teammate's ``/robot_j/msg`` (``std_msgs/Float32MultiArray``, see :mod:`pusht_policy.comm`).

The control step is triggered by the scan rather than by a free-running timer: a new scan
whose stamp is at least ``dt`` (the checkpoint's) after the previous step's starts a step,
with every pose taken at the sample nearest the scan stamp. swarp's frame stack holds the
scans of consecutive control steps, which a timer beating against the lidar's own 10 Hz
would break (a repeated or skipped frame every few steps).

Each step builds the observation with :mod:`pusht_policy.obs` and publishes
``actor.greedy(obs, role)``, or with ``sample`` an action drawn from the policy's
tanh-Gaussian as in training (``squash(loc + scale * eps)``, ``eps`` from a generator seeded
with ``sample_seed``). A delayed-comm actor also reads its teammates' messages from the
previous step of the episode (matched by scan stamp), sends its own after acting, and logs
how many steps heard a teammate when the episode ends. The command is held between
steps. A new episode id zeroes the frame stack; while stopped the node publishes a zero
twist on every scan. Between steps the held command is republished on every scan, so the
onboard watchdog (limo_bridge's robot_bridge, 0.3 s) sees a steady 10 Hz stream and stops the robot when the stream ends; a step parked
waiting for a pose republishes nothing, so a dead pose source stops the robot too. SIGINT
and SIGTERM publish a zero twist before the node exits. With ``record_dir`` set, every episode's per-step
observations, actions, scans and poses (plus the T pose, for diagnosis only) go to
``record_dir/ep<id>_robot_<i>.npz`` (a delayed-comm actor's ``heard`` columns too).
"""

from __future__ import annotations

import bisect
import math
import os
from collections import deque

import numpy as np
import torch
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, Int32

from limo_bridge import spin
from pusht_policy.comm import Messenger
from pusht_policy.obs import (
    ROLE_OBSERVER,
    ROLE_PUSHER,
    ObsBuilder,
    ObsConfig,
    blank_outside_fov,
    _parse,
    crop_fov_at_mount,
    label_rays,
    privileged_cols,
    scan_to_rays,
)
from pusht_policy.task import Task

#: Latched topics (/goal, /episode): a late subscriber gets the last message.
LATCHED = QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
)


def stamp_s(stamp) -> float:
    return stamp.sec + 1e-9 * stamp.nanosec


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class History:
    """The last few stamped samples of one signal, looked up by nearest stamp."""

    def __init__(self, maxlen: int = 50):
        self.t: deque[float] = deque(maxlen=maxlen)
        self.v: deque[tuple[float, ...]] = deque(maxlen=maxlen)

    def push(self, t: float, v: tuple[float, ...]) -> None:
        if self.t and t < self.t[-1]:  # sim time jumped back: forget the old samples
            self.t.clear()
            self.v.clear()
        self.t.append(t)
        self.v.append(v)

    def nearest(self, t: float) -> tuple[float, tuple[float, ...]] | None:
        if not self.t:
            return None
        ts = list(self.t)
        k = bisect.bisect_left(ts, t)
        best = min((j for j in (k - 1, k) if 0 <= j < len(ts)), key=lambda j: abs(ts[j] - t))
        return ts[best], self.v[best]


def roles_for(cfg: ObsConfig) -> np.ndarray:
    """``PushTLidarScenario``'s roles: the first ``n_pushers`` agents push, the rest observe."""
    return np.array(
        [ROLE_PUSHER if a < cfg.pusher_count else ROLE_OBSERVER for a in range(cfg.n_agents)],
        dtype=np.int64,
    )


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy")
        from swarp.rl.ppo import load_checkpoint

        ckpt = self.declare_parameter("checkpoint", "").value
        self.index = int(self.declare_parameter("robot_index", 1).value)
        self.mount_x = float(self.declare_parameter("lidar_mount_x", 0.103).value)
        self.max_pose_age = float(self.declare_parameter("max_pose_age", 0.05).value)
        # The robot's measured unobstructed lidar field of view [rad]; rays outside it are
        # blanked as swarp's lidar_fov does (2 pi: none).
        self.lidar_fov = float(self.declare_parameter("lidar_fov", 2.0 * math.pi).value)
        # The deployed arena's half side [m], for a checkpoint that observes it (trained on
        # swarp's arena_size); required there, ignored by fixed-arena checkpoints.
        arena_half = float(self.declare_parameter("arena_half", 0.0).value)
        self.record_dir = self.declare_parameter("record_dir", "").value
        self.sample = bool(self.declare_parameter("sample", False).value)
        self.gen = torch.Generator().manual_seed(int(self.declare_parameter("sample_seed", 0).value))
        if not ckpt:
            raise ValueError("parameter 'checkpoint' is required")

        torch.set_num_threads(1)
        self.actor, _, meta = load_checkpoint(ckpt, "cpu")
        self.cfg = ObsConfig.from_checkpoint(meta, arena_half=arena_half if arena_half > 0 else None)
        self._check_sensor(meta, bool(self.declare_parameter("allow_fov_below_trained", False).value))
        n_robots = int(self.declare_parameter("n_robots", self.cfg.n_agents).value)
        if n_robots != self.cfg.n_agents:
            raise ValueError(f"checkpoint has {self.cfg.n_agents} agents, n_robots={n_robots}")
        if not 1 <= self.index <= n_robots:
            raise ValueError(f"robot_index {self.index} not in [1, {n_robots}]")
        self.dt = float(meta["dt"])
        self.agent = self.index - 1
        self.builder = ObsBuilder(self.cfg, self.agent)
        self.roles = roles_for(self.cfg)
        # A privileged actor (swarp --privileged-actor) also reads the T's pose from /tee/pose
        # (a mocap rigid body) as swarp's aux targets: ``privileged_cols``.
        self.priv_dim = int(getattr(self.actor, "priv_dim", 0))
        if self.priv_dim:
            task = Task.from_meta(meta, arena_half if arena_half > 0 else None)
            aux_goal = bool(_parse(dict(meta.get("scen_kwargs", {})).get("aux_goal", "False")))
            if self.priv_dim != 4 + int(aux_goal):
                raise ValueError(f"priv_dim {self.priv_dim} is not swarp's aux targets")
            self.priv = (1.0 / self.cfg.max_range, task.goal_tolerance, task.angle_tolerance,
                         self.priv_dim == 5)
        self.role_t = torch.as_tensor(self.roles[self.agent : self.agent + 1])
        self.robots = [f"robot_{j}" for j in range(1, n_robots + 1)]
        self.mates = [j for j in range(n_robots) if j != self.agent]
        self.name = self.robots[self.agent]
        # A delayed-comm actor (swarp --comm): each step it hears the teammates' messages of
        # the previous step and sends its own.
        self.comm = (Messenger(self.actor, self.role_t, self.mates, self.dt)
                     if getattr(self.actor, "delayed", False) else None)
        self.heard_steps = 0  # steps this episode that heard a teammate

        self.poses = [History() for _ in self.robots]
        self.vels = [History() for _ in self.robots]
        for j, r in enumerate(self.robots):
            self.create_subscription(
                PoseStamped, f"/{r}/pose", lambda m, j=j: self._on_pose(j, m), 20
            )
            self.create_subscription(
                TwistStamped, f"/{r}/vel", lambda m, j=j: self._on_vel(j, m), 20
            )
        # The T pose: recorded next to each step, and read by a privileged actor.
        self.tee = History()
        if self.record_dir or self.priv_dim:
            self.create_subscription(PoseStamped, "/tee/pose", self._on_tee, 20)
        self.goal: tuple[float, float, float] | None = None
        self.create_subscription(PoseStamped, "/goal", self._on_goal, LATCHED)
        self.create_subscription(Int32, "/episode", self._on_episode, LATCHED)
        self.create_subscription(LaserScan, f"/{self.name}/scan", self._on_scan, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, f"/{self.name}/cmd_vel", 10)
        if self.comm is not None:
            self.msg_pub = self.create_publisher(Float32MultiArray, f"/{self.name}/msg", 10)
            for j in self.mates:
                self.create_subscription(
                    Float32MultiArray, f"/{self.robots[j]}/msg",
                    lambda m, j=j: self.comm.receive(j, m.data), 10,
                )

        self.episode = -1
        self.episode_t0 = math.inf  # scans stamped before the start message are stale
        self.last_step_t = -math.inf
        self.pending: LaserScan | None = None  # a due scan waiting for the poses to catch up
        self.held = (0.0, 0.0)  # the last step's command, republished between steps
        self.steps = 0
        self.rec: dict[str, list] = {}
        self.get_logger().info(
            f"{self.name}: agent {self.agent} of {n_robots}, role {int(self.roles[self.agent])}, "
            f"obs_dim {self.cfg.obs_dim}, dt {self.dt}, {self.cfg.n_rays} rays to "
            f"{self.cfg.max_range} m, frame_stack {self.cfg.frame_stack}, labels to "
            f"{self.cfg.label_bound} m, arena {self.cfg.arena_half or 'fixed'}, "
            f"{'sampled' if self.sample else 'greedy'} actions"
            + (f", {self.actor.comm} messages of {self.actor.msg_dim}" if self.comm else "")
        )

    def _check_sensor(self, meta, allow_fov_below: bool) -> None:
        """The deployment lidar against the checkpoint's sensor model. The minimum range comes
        from the checkpoint (``ObsConfig.lidar_min_range``, from the mount); the field of view
        is the robot's own (``lidar_fov``), refused below the narrowest one trained on unless
        ``allow_fov_below_trained``; a mount differing from the trained one is reported."""
        from pusht_policy.obs import _parse

        kw = {k: _parse(v) for k, v in dict(meta.get("scen_kwargs", {})).items()}
        fov = kw.get("lidar_fov", 2.0 * math.pi)
        fov_min = float(min(fov) if isinstance(fov, (tuple, list)) else fov)
        if self.lidar_fov < fov_min - 1e-3:
            msg = (f"lidar_fov {math.degrees(self.lidar_fov):.0f} deg is below the checkpoint's trained "
                   f"minimum {math.degrees(fov_min):.0f} deg: the policy never saw so narrow a scan")
            if not allow_fov_below:
                raise ValueError(msg + " (set allow_fov_below_trained to run anyway)")
            self.get_logger().warn(msg)
        mount = float(kw.get("lidar_mount_offset", 0.0))
        if abs(mount - self.mount_x) > 1e-3:
            self.get_logger().warn(
                f"lidar_mount_x {self.mount_x} m differs from the checkpoint's lidar_mount_offset {mount} m")
        self.get_logger().info(
            f"lidar: min range {self.cfg.lidar_min_range} m from the mount (checkpoint), field of view "
            f"{math.degrees(self.lidar_fov):.0f} deg (trained >= {math.degrees(fov_min):.0f} deg, cut "
            f"{'at the mount' if self.cfg.lidar_fov_at_mount else 'per centre ray'}), mount {self.mount_x} m")

    # ------------------------------------------------------------------ inputs

    def _on_pose(self, j: int, m: PoseStamped) -> None:
        p = m.pose
        self.poses[j].push(stamp_s(m.header.stamp), (p.position.x, p.position.y, yaw_of(p.orientation)))
        if self.pending is not None:
            self._step(self.pending)

    def _on_vel(self, j: int, m: TwistStamped) -> None:
        self.vels[j].push(stamp_s(m.header.stamp), (m.twist.linear.x, m.twist.linear.y))

    def _on_tee(self, m: PoseStamped) -> None:
        p = m.pose
        self.tee.push(stamp_s(m.header.stamp), (p.position.x, p.position.y, yaw_of(p.orientation)))

    def _on_goal(self, m: PoseStamped) -> None:
        self.goal = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))

    def _on_episode(self, m: Int32) -> None:
        if m.data == self.episode:
            return
        if self.comm and self.episode >= 0 and self.steps > 1:
            self.get_logger().info(
                f"episode {self.episode}: heard a teammate on {self.heard_steps} of "
                f"{self.steps - 1} steps after the first"
            )
        self._flush()
        self.episode = m.data
        self.pending = None
        if self.episode >= 0:
            self.builder.reset()
            self.episode_t0 = self.get_clock().now().nanoseconds * 1e-9
            self.last_step_t = -math.inf
            self.steps = self.heard_steps = 0
            self.rec = {k: [] for k in ("t", "obs", "act", "scan", "pose", "vel", "goal", "tee")
                        + (("heard",) if self.comm else ())}
        else:
            self.episode_t0 = math.inf
            self.held = (0.0, 0.0)
            self._publish(0.0, 0.0)

    # ------------------------------------------------------------------ control step

    def _publish(self, v: float, w: float) -> None:
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)

    def _on_scan(self, scan: LaserScan) -> None:
        t = stamp_s(scan.header.stamp)
        if self.episode < 0 or self.goal is None or t < self.episode_t0:
            self._publish(0.0, 0.0)
            return
        # Step every dt; a scan period's slack absorbs stamp jitter.
        if t - self.last_step_t < self.dt - 0.5 * max(scan.scan_time, 0.01):
            if self.pending is None:
                self._publish(*self.held)
            return
        if self.pending is not None:
            self.get_logger().warn(
                f"no pose within {self.max_pose_age}s of scan {stamp_s(self.pending.header.stamp):.3f} "
                "before the next one", throttle_duration_sec=2.0,
            )
        self._step(scan)

    def _step(self, scan: LaserScan) -> None:
        """One control step on ``scan``, or, when some pose has no sample near its stamp
        yet (a pose relay lagging behind the lidar), park it until a pose arrives: skipping
        it would stretch this step to the next scan and space the stacked frames unevenly."""
        t = stamp_s(scan.header.stamp)
        state = self._state_at(t)
        tee = self.tee.nearest(t) if self.priv_dim else None
        if state is None or (self.priv_dim and tee is None):
            self.pending = scan
            return
        self.pending = None
        pos, vel, theta = state
        a, cfg = self.agent, self.cfg
        me = (pos[a, 0], pos[a, 1], theta[a])
        # The field of view as the checkpoint was trained: cut at the mount (on the raw
        # scan), or per centre ray (after binning).
        raw = (crop_fov_at_mount(scan.ranges, scan.angle_min, scan.angle_increment, self.lidar_fov)
               if cfg.lidar_fov_at_mount else scan.ranges)
        rays, _ = scan_to_rays(
            raw, scan.angle_min, scan.angle_increment,
            max(scan.range_min, cfg.lidar_min_range), scan.range_max,
            (self.mount_x, 0.0), 0.0, cfg.n_rays, cfg.max_range,
            pose=me, own_half=cfg.half_extents if cfg.agent_shape == "box" else None,
            body_frame=cfg.body_frame, span=cfg.lidar_span,
        )
        box = cfg.agent_shape == "box"
        ranges, cls = label_rays(
            rays, me, cfg.max_range, cfg.label_bound, pos[self.mates], theta[self.mates],
            mate_half=cfg.half_extents if box else None,
            mate_radius=None if box else cfg.agent_radius,
            body_frame=cfg.body_frame, span=cfg.lidar_span,
        )
        if not cfg.lidar_fov_at_mount:
            ranges, cls = blank_outside_fov(ranges, cls, self.lidar_fov, cfg.max_range,
                                            span=cfg.lidar_span)
        gx, gy, gth = self.goal
        obs = self.builder.build(pos, vel, theta, self.roles, np.array([gx, gy]), gth, ranges, cls)
        if self.priv_dim:
            obs = np.concatenate([obs, privileged_cols(tee[1], pos[a], self.goal, *self.priv)])
        obs_t = torch.from_numpy(obs)[None]
        heard = self.comm.heard(self.episode, t) if self.comm else None
        act = self._act(obs_t, heard)
        self.held = (float(act[0]), float(act[1]))
        self._publish(*self.held)
        if self.comm:
            self.msg_pub.publish(Float32MultiArray(data=self.comm.payload(self.episode, t, obs_t)))
            self.heard_steps += int(heard[0, -1] > 0)
        self.last_step_t = t
        self.steps += 1
        if self.record_dir:
            r = self.rec
            r["t"].append(t)
            r["obs"].append(obs)
            r["act"].append(act.astype(np.float32))
            r["scan"].append(np.asarray(scan.ranges, dtype=np.float32))
            r["pose"].append(np.column_stack([pos, theta]).astype(np.float32))
            r["vel"].append(vel.astype(np.float32))
            r["goal"].append(np.array(self.goal, dtype=np.float32))
            tee = self.tee.nearest(t)
            r["tee"].append(np.array(tee[1] if tee else (np.nan,) * 3, dtype=np.float32))
            if self.comm:
                r["heard"].append(heard[0].numpy())

    @torch.no_grad()
    def _act(self, obs: torch.Tensor, heard: torch.Tensor | None = None) -> np.ndarray:
        if self.comm:
            loc, scale = self.comm.step(obs, heard)
            if not self.sample:
                return self.actor.squash(loc)[0].numpy()
        elif not self.sample:
            return self.actor.greedy(obs, self.role_t)[0].numpy()
        else:
            loc, scale = self.actor(obs, self.role_t)
        # The trainer's rollout draw (swarp.rl.ppo): u ~ N(loc, scale), action squash(u).
        u = loc + scale * torch.randn(loc.shape, generator=self.gen, dtype=loc.dtype)
        return self.actor.squash(u)[0].numpy()

    def _state_at(self, t: float):
        pos = np.zeros((len(self.robots), 2))
        vel = np.zeros((len(self.robots), 2))
        theta = np.zeros(len(self.robots))
        for j in range(len(self.robots)):
            p, v = self.poses[j].nearest(t), self.vels[j].nearest(t)
            if p is None or v is None or abs(p[0] - t) > self.max_pose_age:
                return None
            pos[j], theta[j] = p[1][:2], p[1][2]
            vel[j] = v[1]
        return pos, vel, theta

    def _flush(self) -> None:
        """Write the finished episode's record, if any."""
        if not self.record_dir or self.episode < 0 or not self.rec.get("t"):
            return
        os.makedirs(self.record_dir, exist_ok=True)
        path = os.path.join(self.record_dir, f"ep{self.episode:04d}_{self.name}.npz")
        np.savez_compressed(
            path,
            **{k: np.asarray(v) for k, v in self.rec.items()},
            agent=self.agent, roles=self.roles, dt=self.dt, mount_x=self.mount_x,
            layout_names=np.array(list(self.cfg.layout())),
            layout_slices=np.array([(v.start, v.stop) for v in self.cfg.layout().values()]),
        )
        self.get_logger().info(f"episode {self.episode}: {self.steps} steps -> {path}")
        self.rec = {}

    def stop(self) -> None:
        """Zero the robot on the way out (SIGINT/SIGTERM), while the context is alive."""
        self.held = (0.0, 0.0)
        self._publish(0.0, 0.0)
        spin.flush(self.cmd_pub)
        self.get_logger().info(f"{self.name}: stopped, zero twist sent")

    def destroy_node(self):
        self._flush()
        return super().destroy_node()


def main(args=None) -> None:
    spin.init(args)
    node = PolicyNode()
    spin.spin_until_signal(node, node.stop)
