"""The sysid experiments in a running Gazebo sim (``ros2 run pusht_policy sysid_gz``).

Start the sim first (two robots, real time, so command timing is exact to a few ms)::

    ros2 launch pusht_gz sim.launch.py n_robots:=2 headless:=true rtf:=1 &
    ros2 run pusht_policy sysid_gz --ros-args -p use_sim_time:=true -p checkpoint:=<ckpt.pt>

For each experiment: zero commands, teleport robots and T at rest, wait for fresh poses
(and scans; ground truth from ``/<body>/gt_odom``), then publish command ``k`` at sim time ``t0 + k * DT``. Saves
``<out>/gz_<name>.npz``: the ground truth resampled at the command instants (the same grid
swarp records), the raw 50 Hz streams, and for ``lidar`` experiments robot 1's scan turned
into swarp rays by :mod:`pusht_policy.obs` exactly as the policy node does.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:  # pragma: no cover
    Contacts = None

from pusht_gz.gz_reset import GzWorld
from pusht_policy.obs import ObsConfig, label_rays, scan_to_rays
from pusht_policy.policy_node import stamp_s, yaw_of
from pusht_policy.sysid.experiments import DT, Experiment, all_experiments

BODIES = ("robot_1", "robot_2", "tee")


class SysidGz(Node):
    def __init__(self):
        super().__init__("sysid_gz")
        self.out = self.declare_parameter("out", "out/sysid").value
        only = self.declare_parameter("only", "").value
        ckpt = self.declare_parameter("checkpoint", "").value
        self.mount_x = float(self.declare_parameter("lidar_mount_x", 0.103).value)
        self.exps = [e for e in all_experiments() if not only or e.name in only.split(",") or e.group in only.split(",")]
        self.cfg = None
        if ckpt:
            import torch

            self.cfg = ObsConfig.from_checkpoint(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.gz = GzWorld("pusht")
        self.raw = {b: [] for b in BODIES}  # (t, x, y, yaw, vx, vy, w)
        self.last = {b: None for b in BODIES}
        self.scan: LaserScan | None = None
        for b in BODIES:
            self.create_subscription(Odometry, f"/{b}/gt_odom", lambda m, b=b: self._on_odom(b, m), 50)
        self.create_subscription(LaserScan, "/robot_1/scan", self._on_scan, qos_profile_sensor_data)
        # Robot -> T contact forces, when the sim runs with tee_contact_sensor:=true.
        self.contacts = []  # (t, F1x, F1y, F2x, F2y): horizontal force on the T from robot 1, 2
        if Contacts is not None:
            self.create_subscription(Contacts, "/tee/contacts", self._on_contacts, 50)
        self.cmd = [self.create_publisher(Twist, f"/robot_{i}/cmd_vel", 10) for i in (1, 2)]
        os.makedirs(self.out, exist_ok=True)

    # ------------------------------------------------------------------ io

    def _on_odom(self, b, m: Odometry) -> None:
        p, tw = m.pose.pose, m.twist.twist
        yaw = yaw_of(p.orientation)
        c, s = math.cos(yaw), math.sin(yaw)  # body-frame twist -> world
        vx, vy = c * tw.linear.x - s * tw.linear.y, s * tw.linear.x + c * tw.linear.y
        row = (stamp_s(m.header.stamp), p.position.x, p.position.y, yaw, vx, vy, tw.angular.z)
        self.last[b] = row
        self.raw[b].append(row)

    def _on_contacts(self, m) -> None:
        f = np.zeros(4)
        for c in m.contacts:
            n1, n2 = c.collision1.name, c.collision2.name
            other = n2 if n1.startswith("tee::") else n1
            on_tee_is_1 = n1.startswith("tee::")
            for i, r in enumerate(("robot_1", "robot_2")):
                if other.startswith(r + "::"):
                    for w in c.wrenches:
                        b = w.body_1_wrench if on_tee_is_1 else w.body_2_wrench
                        f[2 * i] += b.force.x
                        f[2 * i + 1] += b.force.y
        self.contacts.append((stamp_s(m.header.stamp), *f))

    def _on_scan(self, m: LaserScan) -> None:
        self.scan = m

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def spin_until(self, pred, timeout=60.0, what="") -> None:
        deadline = time.monotonic() + timeout
        while not pred():
            if time.monotonic() > deadline:
                raise TimeoutError(f"waiting for {what}")
            rclpy.spin_once(self, timeout_sec=0.001)

    def publish(self, cmds: np.ndarray) -> None:
        for i, pub in enumerate(self.cmd):
            m = Twist()
            m.linear.x, m.angular.z = float(cmds[i, 0]), float(cmds[i, 1])
            pub.publish(m)

    # ------------------------------------------------------------------ experiments

    def reset(self, e: Experiment) -> float:
        self.publish(np.zeros((2, 2)))
        t = self.now()
        self.spin_until(lambda: self.now() - t >= 0.5, what="stop")
        poses = {"robot_1": e.robots[0], "robot_2": e.robots[1], "tee": e.tee}
        for attempt in range(3):  # the service can time out on a loaded machine
            if self.gz.set_poses_at_rest(poses, 0.3):
                break
            self.get_logger().warn(f"{e.name}: teleport failed (attempt {attempt})")
        else:
            raise RuntimeError(f"{e.name}: teleport failed")
        t_tp = (self.gz.last_sim_time() or self.now()) + 0.05
        self.spin_until(
            lambda: all(self.last[b] is not None and self.last[b][0] > t_tp for b in BODIES)
            and self.scan is not None and stamp_s(self.scan.header.stamp) > t_tp + 0.1,
            what="fresh poses and scan",
        )
        err = max(np.hypot(self.last[b][1] - p[0], self.last[b][2] - p[1]) for b, p in poses.items())
        if err > 0.02:
            self.get_logger().warn(f"{e.name}: teleport off by {err:.3f} m")
        return t_tp

    def run_lidar(self, e: Experiment) -> None:
        if self.cfg is None:
            raise ValueError("lidar experiments need the 'checkpoint' parameter")
        cfg, s = self.cfg, self.scan
        me = self.last["robot_1"][1:4]
        mate = self.last["robot_2"][1:4]
        rays, _ = scan_to_rays(
            s.ranges, s.angle_min, s.angle_increment, s.range_min, s.range_max,
            (self.mount_x, 0.0), 0.0, cfg.n_rays, cfg.max_range,
            pose=me, own_half=cfg.half_extents, body_frame=cfg.body_frame,
        )
        rays, cls = label_rays(
            rays, me, cfg.max_range, cfg.world_size, np.array([mate[:2]]), np.array([mate[2]]),
            mate_half=cfg.half_extents, body_frame=cfg.body_frame,
        )
        np.savez(
            os.path.join(self.out, f"gz_{e.name}.npz"),
            rays=rays, cls=cls, scan=np.asarray(s.ranges, dtype=np.float32),
            scan_angle_min=s.angle_min, scan_angle_increment=s.angle_increment, scan_range_min=s.range_min,
            robot=np.array(me), tee=np.array(self.last["tee"][1:4]),
        )

    def run_motion(self, e: Experiment) -> None:
        for b in BODIES:
            self.raw[b] = []
        self.contacts = []
        n = len(e.cmds)
        t0 = self.now() + 0.05
        k = 0
        while k < n or self.now() < t0 + n * DT:
            now = self.now()
            while k < n and now >= t0 + k * DT:
                self.publish(e.cmds[k])
                k += 1
            rclpy.spin_once(self, timeout_sec=0.0005)
        self.publish(np.zeros((2, 2)))
        grid = t0 + np.arange(n + 1) * DT
        rec = {}
        raw = {b: np.array(self.raw[b]) for b in BODIES}
        for b in BODIES:
            r = raw[b]
            yaw = np.unwrap(r[:, 3])
            rec[b] = np.column_stack(
                [np.interp(grid, r[:, 0], r[:, c]) for c in (1, 2)]
                + [np.interp(grid, r[:, 0], yaw)]
                + [np.interp(grid, r[:, 0], r[:, c]) for c in (4, 5, 6)]
            )
        robots = np.stack([rec["robot_1"], rec["robot_2"]], 1)  # [T, 2, 6]
        # Mean robot -> T force over each control step (zero when nothing touched).
        cf = np.array(self.contacts).reshape(-1, 5)
        contact = np.zeros((n + 1, 2, 2))
        for k in range(1, n + 1):
            sel = (cf[:, 0] > grid[k - 1]) & (cf[:, 0] <= grid[k])
            if sel.any():
                contact[k] = cf[sel, 1:].mean(0).reshape(2, 2)
        np.savez(
            os.path.join(self.out, f"gz_{e.name}.npz"),
            t=grid - t0, cmds=e.cmds,
            robots=robots[..., :3], robots_vel=robots[..., 3:5], robots_w=robots[..., 5],
            tee=rec["tee"][:, :3], tee_vel=rec["tee"][:, 3:5], tee_w=rec["tee"][:, 5],
            contact=contact, raw_contacts=np.column_stack([cf[:, 0] - t0, cf[:, 1:]]) if len(cf) else cf,
            **{f"raw_{b}": np.column_stack([raw[b][:, 0] - t0, raw[b][:, 1:]]) for b in BODIES},
        )

    def run(self) -> None:
        self.spin_until(lambda: self.now() > 0 and all(self.last[b] for b in BODIES), what="sim")
        for e in self.exps:
            self.reset(e)
            if e.group == "lidar":
                self.run_lidar(e)
            else:
                self.run_motion(e)
            self.get_logger().info(f"{e.name} done")
        self.publish(np.zeros((2, 2)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SysidGz()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
