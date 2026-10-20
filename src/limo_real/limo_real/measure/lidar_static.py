"""A static T-mini Pro scan: the chassis' rear occlusion, self-returns, heading check.

    ros2 run limo_real measure_lidar_static --ros-args -p robot:=robot_1 -p scans:=100

Put the robot in open space, nothing within 1 m except one narrow object (a bottle, a
table leg) straight ahead of the robot at ~``probe_range`` m. Collects ``scans`` scans of
``/<robot>/scan`` (the bridge's relay, after ``scan_yaw_offset``) and reports, per bearing
in 2-degree bins (0 = the robot's heading, CCW):

* ``self``: bearings whose returns sit within ``self_radius`` of the lidar in most scans
  (the chassis, the mast, cables): blank them or exclude them from the FOV;
* ``blind``: bearings with no valid return in most scans;
* ``fov_deg``: the widest sector centred on the heading free of both, the value to use for
  the policy's and the sim's ``lidar_fov`` (and for recipe2's FOV range);
* ``probe_bearing_deg``: the bearing of the nearest return within ±30 deg of the heading
  beyond ``self_radius``: should be 0; if not, set ``scan_yaw_offset`` to minus it.

With ``checkpoint`` set, also the policy's view: every scan turned into the checkpoint's
binned rays (its minimum range from the mount, the robot's own footprint dropped, the
arena around the robot) as policy_node does, and ``policy_rays_hit``: the rays that
read a hit in most scans. In open space with nothing within the lidar's policy range this
must be empty; a hit there is the robot seeing itself (wheels, mount, cables) as an obstacle.

Saves the raw ranges to ``<out>/lidar_static_<robot>.npz``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from limo_real.measure.common import save, wait


class LidarStatic(Node):
    def __init__(self):
        super().__init__("measure_lidar_static")
        p = self.declare_parameter
        self.robot = p("robot", "robot_1").value
        self.n = int(p("scans", 100).value)
        self.self_radius = float(p("self_radius", 0.25).value)
        self.probe_range = float(p("probe_range", 1.0).value)
        self.out = p("out", "out/measure").value
        self.checkpoint = p("checkpoint", "").value
        self.mount_x = float(p("lidar_mount_x", 0.103).value)
        self.scans: list[LaserScan] = []
        self.create_subscription(LaserScan, f"/{self.robot}/scan", self.scans.append, qos_profile_sensor_data)

    def run(self) -> dict:
        if not wait(self, 60.0, lambda: len(self.scans) >= self.n):
            raise RuntimeError(f"only {len(self.scans)} scans on /{self.robot}/scan")
        bins = np.arange(-180, 180, 2)
        self_hits = np.zeros(bins.size)
        valid_hits = np.zeros(bins.size)
        count = np.zeros(bins.size)
        probe = []
        raw = []
        for s in self.scans:
            r = np.asarray(s.ranges, dtype=float)
            raw.append(r)
            ang = s.angle_min + np.arange(r.size) * s.angle_increment
            deg = np.degrees(np.arctan2(np.sin(ang), np.cos(ang)))
            k = np.clip(((deg + 180) // 2).astype(int), 0, bins.size - 1)
            ok = np.isfinite(r) & (r >= s.range_min) & (r < s.range_max) & (r > 0)
            np.add.at(count, k, 1)
            np.add.at(valid_hits, k, ok)
            np.add.at(self_hits, k, ok & (r < self.self_radius))
            front = ok & (np.abs(deg) < 30) & (r >= self.self_radius)
            if front.any():
                j = np.flatnonzero(front)[np.argmin(r[front])]
                probe.append(deg[j])
        count = np.maximum(count, 1)
        is_self = self_hits / count > 0.5
        is_blind = valid_hits / count < 0.5
        bad = is_self | is_blind
        # widest symmetric sector around 0 deg free of bad bins
        half = 0
        for h in range(1, 181):
            sel = np.abs(bins + 1) < h
            if bad[sel].any():
                break
            half = h
        sectors = []
        for mask, name in ((is_self, "self"), (is_blind, "blind")):
            idx = np.flatnonzero(mask)
            if idx.size:
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                sectors.append({name: [[int(bins[r[0]]), int(bins[r[-1]] + 2)] for r in runs]})
        policy = self._policy_view() if self.checkpoint else None
        s0 = self.scans[-1]
        os.makedirs(self.out, exist_ok=True)
        np.savez_compressed(os.path.join(self.out, f"lidar_static_{self.robot}.npz"), ranges=np.array(raw, dtype=object),
                            angle_min=s0.angle_min, angle_increment=s0.angle_increment)
        return {
            "robot": self.robot, "scans": len(self.scans), "samples_per_scan": len(s0.ranges),
            "angle_min_deg": math.degrees(s0.angle_min), "angle_increment_deg": math.degrees(s0.angle_increment),
            "range_min": s0.range_min, "range_max": s0.range_max,
            "sectors": sectors, "fov_deg": 2 * half, "fov_rad": round(math.radians(2 * half), 4),
            "probe_bearing_deg": round(float(np.median(probe)), 2) if probe else None,
            **({"policy_rays_hit": policy} if policy is not None else {}),
        }

    def _policy_view(self) -> list:
        """Rays (index, bearing deg, share of scans) the checkpoint's observation reads as hits."""
        import torch
        from pusht_policy.obs import ObsConfig, label_rays, scan_to_rays

        cfg = ObsConfig.from_checkpoint(torch.load(self.checkpoint, map_location="cpu", weights_only=False))
        hits = np.zeros(cfg.n_rays)
        for s in self.scans:
            rays, _ = scan_to_rays(
                s.ranges, s.angle_min, s.angle_increment, max(s.range_min, cfg.lidar_min_range), s.range_max,
                (self.mount_x, 0.0), 0.0, cfg.n_rays, cfg.max_range,
                own_half=cfg.half_extents if cfg.agent_shape == "box" else None, body_frame=cfg.body_frame)
            _, cls = label_rays(rays, (0.0, 0.0, 0.0), cfg.max_range, cfg.world_size, np.zeros((0, 2)),
                                np.zeros(0), body_frame=cfg.body_frame)
            hits += cls != 0
        share = hits / len(self.scans)
        return [[int(k), round(k * 360.0 / cfg.n_rays, 2), round(float(share[k]), 2)]
                for k in np.flatnonzero(share > 0.5)]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarStatic()
    try:
        save(node.out, f"lidar_static_{node.robot}.json", node.run())
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
