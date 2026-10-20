"""T-mini Pro rate, scan geometry and stamp latency, as the laptop receives them.

    ros2 run limo_real measure_lidar_rate --ros-args -p robot:=robot_1 -p seconds:=30

Reports the scan rate (mean, std of the period, gaps over 1.5 periods), samples per scan
(min/max: ydlidar's count varies with the motor speed), ``scan_time``, the share of
invalid ranges, and ``receive time - header stamp`` (mean, p95): the transport latency
plus the robot-laptop clock offset. A mean far from a few ms, or negative, means the
clocks are not synchronised (chrony, docs/real_robot.md), which breaks the policy's
pose-at-scan-stamp lookup.
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from limo_real.frames import stamp_s
from limo_real.measure.common import save, wait


class LidarRate(Node):
    def __init__(self):
        super().__init__("measure_lidar_rate")
        p = self.declare_parameter
        self.robot = p("robot", "robot_1").value
        self.seconds = float(p("seconds", 30.0).value)
        self.out = p("out", "out/measure").value
        self.rows: list[tuple[float, float, int, float, float]] = []
        self.create_subscription(LaserScan, f"/{self.robot}/scan", self._on_scan, qos_profile_sensor_data)

    def _on_scan(self, s: LaserScan) -> None:
        r = np.asarray(s.ranges, dtype=float)
        ok = np.isfinite(r) & (r >= s.range_min) & (r < s.range_max) & (r > 0)
        self.rows.append((time.time(), stamp_s(s.header.stamp), r.size, float(s.scan_time), 1.0 - ok.mean()))

    def run(self) -> dict:
        wait(self, self.seconds)
        if len(self.rows) < 10:
            raise RuntimeError(f"only {len(self.rows)} scans on /{self.robot}/scan")
        a = np.array(self.rows)
        period = np.diff(a[:, 1])
        lat = a[:, 0] - a[:, 1]
        return {
            "robot": self.robot, "scans": len(a), "rate_hz": round(1.0 / period.mean(), 3),
            "period_std_s": round(float(period.std()), 4),
            "gaps_over_1.5_periods": int((period > 1.5 * period.mean()).sum()),
            "samples_min": int(a[:, 2].min()), "samples_max": int(a[:, 2].max()),
            "scan_time_s": round(float(np.median(a[:, 3])), 4),
            "invalid_share": round(float(a[:, 4].mean()), 3),
            "receive_minus_stamp_s": {"mean": round(float(lat.mean()), 4),
                                      "p95": round(float(np.percentile(lat, 95)), 4),
                                      "min": round(float(lat.min()), 4)},
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarRate()
    try:
        save(node.out, f"lidar_rate_{node.robot}.json", node.run())
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
