"""Run one robot's swarp ``pusht_lidar`` policy on ROS topics.

One node per robot (``robot_index`` i, 1-based, drives ``robot_i``), as on the real fleet:

* in: its own ``/robot_i/scan``; every robot's ``/robot_j/pose`` and ``/robot_j/vel``
  (world frame, from Gazebo ground truth or mocap); ``/goal``; ``/episode``
  (``std_msgs/Int32``: the running episode's id, ``-1`` while stopped);
* out: ``/robot_i/cmd_vel``.

The control step is triggered by the scan rather than by a free-running timer: a new scan
whose stamp is at least ``dt`` (the checkpoint's) after the previous step's starts a step,
with every pose taken at the sample nearest the scan stamp. swarp's frame stack holds the
scans of consecutive control steps, which a timer beating against the lidar's own 10 Hz
would break (a repeated or skipped frame every few steps).

Each step builds the observation with :mod:`pusht_policy.obs` and publishes
``actor.greedy(obs, role)``, or with ``sample`` an action drawn from the policy's
tanh-Gaussian as in training (``squash(loc + scale * eps)``, ``eps`` from a generator seeded
with ``sample_seed``). The command is held between steps. A new episode id zeroes the frame
stack; while stopped the node publishes a zero twist on every scan. With ``record_dir`` set, every episode's per-step
observations, actions, scans and poses (plus the T pose, for diagnosis only) go to
``record_dir/ep<id>_robot_<i>.npz``.
"""

from __future__ import annotations

import bisect
import math
import os
from collections import deque

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32

from pusht_policy.obs import ROLE_OBSERVER, ROLE_PUSHER, ObsBuilder, ObsConfig, label_rays, scan_to_rays

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
        self.record_dir = self.declare_parameter("record_dir", "").value
        self.sample = bool(self.declare_parameter("sample", False).value)
        self.gen = torch.Generator().manual_seed(int(self.declare_parameter("sample_seed", 0).value))
        if not ckpt:
            raise ValueError("parameter 'checkpoint' is required")

        torch.set_num_threads(1)
        self.actor, _, meta = load_checkpoint(ckpt, "cpu")
        self.cfg = ObsConfig.from_checkpoint(meta)
        n_robots = int(self.declare_parameter("n_robots", self.cfg.n_agents).value)
        if n_robots != self.cfg.n_agents:
            raise ValueError(f"checkpoint has {self.cfg.n_agents} agents, n_robots={n_robots}")
        if not 1 <= self.index <= n_robots:
            raise ValueError(f"robot_index {self.index} not in [1, {n_robots}]")
        self.dt = float(meta["dt"])
        self.agent = self.index - 1
        self.builder = ObsBuilder(self.cfg, self.agent)
        self.roles = roles_for(self.cfg)
        self.role_t = torch.as_tensor(self.roles[self.agent : self.agent + 1])
        self.robots = [f"robot_{j}" for j in range(1, n_robots + 1)]
        self.mates = [j for j in range(n_robots) if j != self.agent]
        self.name = self.robots[self.agent]

        self.poses = [History() for _ in self.robots]
        self.vels = [History() for _ in self.robots]
        for j, r in enumerate(self.robots):
            self.create_subscription(
                PoseStamped, f"/{r}/pose", lambda m, j=j: self._on_pose(j, m), 20
            )
            self.create_subscription(
                TwistStamped, f"/{r}/vel", lambda m, j=j: self._on_vel(j, m), 20
            )
        # Diagnostics only (the policy never reads it): the T pose, recorded next to each step.
        self.tee = History()
        if self.record_dir:
            self.create_subscription(PoseStamped, "/tee/pose", self._on_tee, 20)
        self.goal: tuple[float, float, float] | None = None
        self.create_subscription(PoseStamped, "/goal", self._on_goal, LATCHED)
        self.create_subscription(Int32, "/episode", self._on_episode, LATCHED)
        self.create_subscription(LaserScan, f"/{self.name}/scan", self._on_scan, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, f"/{self.name}/cmd_vel", 10)

        self.episode = -1
        self.episode_t0 = math.inf  # scans stamped before the start message are stale
        self.last_step_t = -math.inf
        self.pending: LaserScan | None = None  # a due scan waiting for the poses to catch up
        self.steps = 0
        self.rec: dict[str, list] = {}
        self.get_logger().info(
            f"{self.name}: agent {self.agent} of {n_robots}, role {int(self.roles[self.agent])}, "
            f"obs_dim {self.cfg.obs_dim}, dt {self.dt}, {self.cfg.n_rays} rays to "
            f"{self.cfg.max_range} m, frame_stack {self.cfg.frame_stack}, "
            f"{'sampled' if self.sample else 'greedy'} actions"
        )

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
        self._flush()
        self.episode = m.data
        self.pending = None
        if self.episode >= 0:
            self.builder.reset()
            self.episode_t0 = self.get_clock().now().nanoseconds * 1e-9
            self.last_step_t = -math.inf
            self.steps = 0
            self.rec = {k: [] for k in ("t", "obs", "act", "scan", "pose", "vel", "goal", "tee")}
        else:
            self.episode_t0 = math.inf
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
        if state is None:
            self.pending = scan
            return
        self.pending = None
        pos, vel, theta = state
        a, cfg = self.agent, self.cfg
        me = (pos[a, 0], pos[a, 1], theta[a])
        rays, _ = scan_to_rays(
            scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max,
            (self.mount_x, 0.0), 0.0, cfg.n_rays, cfg.max_range,
            pose=me, own_half=cfg.half_extents if cfg.agent_shape == "box" else None,
            body_frame=cfg.body_frame,
        )
        box = cfg.agent_shape == "box"
        ranges, cls = label_rays(
            rays, me, cfg.max_range, cfg.world_size, pos[self.mates], theta[self.mates],
            mate_half=cfg.half_extents if box else None,
            mate_radius=None if box else cfg.agent_radius,
            body_frame=cfg.body_frame,
        )
        gx, gy, gth = self.goal
        obs = self.builder.build(pos, vel, theta, self.roles, np.array([gx, gy]), gth, ranges, cls)
        act = self._act(torch.from_numpy(obs)[None])
        self._publish(act[0], act[1])
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

    @torch.no_grad()
    def _act(self, obs: torch.Tensor) -> np.ndarray:
        if not self.sample:
            return self.actor.greedy(obs, self.role_t)[0].numpy()
        # The trainer's rollout draw (swarp.rl.ppo): u ~ N(loc, scale), action squash(u).
        loc, scale = self.actor(obs, self.role_t)
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

    def destroy_node(self):
        self._flush()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._publish(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()
