"""Log a pose estimate against a reference and summarise the error, per robot.

    ros2 run limo_real pose_error --ros-args -p est:=pose -p ref:=gt_pose -p out:=out/amcl_err   # Gazebo
    ros2 run limo_real pose_error --ros-args -p est:=amcl/pose -p ref:=pose -p out:=out/real_amcl  # mocap

Every ``/<robot>/<est>`` sample is paired with the ``/<robot>/<ref>`` sample nearest its
stamp (within ``max_dt``) and tagged with the running ``/episode`` id (``-1``: between
episodes). On SIGINT/SIGTERM writes ``<out>/samples.csv`` and ``<out>/summary.json`` and
prints the summary, over the samples taken during episodes:

* ``pos``: |position error| mean, median, p95, max [m]; ``yaw``: |heading error| [rad];
* ``bias``: per robot and episode, the mean error vector and mean signed heading error;
  their norms' mean and max over episodes (what swarp's per-episode ``pose_bias_std`` models);
* ``jitter``: the std of the error about its episode mean (``pose_jitter_std``);
* ``along`` / ``across``: the bias split along and across the robot's heading.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32

from limo_bridge import spin
from limo_real.frames import stamp_s, wrap, yaw_of

COLUMNS = ["t", "robot", "episode", "ex", "ey", "eyaw", "ref_yaw"]
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class PoseError(Node):
    def __init__(self):
        super().__init__("pose_error")
        p = self.declare_parameter
        self.robots = list(p("robots", ["robot_1", "robot_2"]).value)
        est, ref = p("est", "pose").value, p("ref", "gt_pose").value
        self.out = p("out", "out/pose_error").value
        self.max_dt = float(p("max_dt", 0.02).value)
        self.ref: dict[str, tuple[list[float], list[tuple[float, float, float]]]] = {
            r: ([], []) for r in self.robots
        }
        self.rows: list[tuple] = []
        self.episode = -1
        self.create_subscription(Int32, "/episode", lambda m: setattr(self, "episode", m.data), LATCHED)
        for r in self.robots:
            self.create_subscription(PoseStamped, f"/{r}/{ref}", lambda m, r=r: self._on_ref(r, m), 50)
            self.create_subscription(PoseStamped, f"/{r}/{est}", lambda m, r=r: self._on_est(r, m), 50)
        self.get_logger().info(f"/<robot>/{est} against /<robot>/{ref} for {self.robots} -> {self.out}")

    @staticmethod
    def _xyth(m: PoseStamped) -> tuple[float, float, float]:
        return m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation)

    def _on_ref(self, r: str, m: PoseStamped) -> None:
        ts, vs = self.ref[r]
        t = stamp_s(m.header.stamp)
        if ts and t < ts[-1]:
            ts.clear()
            vs.clear()
        ts.append(t)
        vs.append(self._xyth(m))
        if len(ts) > 2000:
            del ts[:1000], vs[:1000]

    def _on_est(self, r: str, m: PoseStamped) -> None:
        ts, vs = self.ref[r]
        if not ts:
            return
        t = stamp_s(m.header.stamp)
        k = bisect.bisect_left(ts, t)
        j = min((i for i in (k - 1, k) if 0 <= i < len(ts)), key=lambda i: abs(ts[i] - t))
        if abs(ts[j] - t) > self.max_dt:
            return
        x, y, th = self._xyth(m)
        gx, gy, gth = vs[j]
        self.rows.append((t, r, self.episode, x - gx, y - gy, wrap(th - gth), gth))

    def stop(self) -> None:
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "samples.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(self.rows)
        s = summary(self.rows)
        with open(os.path.join(self.out, "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        self.get_logger().info(f"{len(self.rows)} samples -> {self.out}\n{json.dumps(s, indent=2)}")


def summary(all_rows) -> dict:
    """The statistics of the module docstring over ``(t, robot, episode, ex, ey, eyaw,
    ref_yaw[, run])`` rows; an episode is identified by ``(robot, episode, run)``."""
    rows = [row for row in all_rows if int(row[2]) >= 0]
    out: dict = {"samples": len(rows)}
    if not rows:
        return out
    ex = np.array([float(row[3]) for row in rows])
    ey = np.array([float(row[4]) for row in rows])
    eth = np.array([float(row[5]) for row in rows])
    gth = np.array([float(row[6]) for row in rows])
    pos = np.hypot(ex, ey)

    def stats(v: np.ndarray) -> dict:
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    out["pos"] = stats(pos)
    out["yaw"] = stats(np.abs(eth))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row[1], row[2], row[7] if len(row) > 7 else "")].append(i)
    bias_xy, bias_th, along, across, res_xy, res_th = [], [], [], [], [], []
    for idx in groups.values():
        idx = np.array(idx)
        bx, by, bt = ex[idx].mean(), ey[idx].mean(), eth[idx].mean()
        bias_xy.append(math.hypot(bx, by))
        bias_th.append(abs(bt))
        c, s = np.cos(gth[idx]), np.sin(gth[idx])
        along.append(float(np.mean(c * ex[idx] + s * ey[idx])))
        across.append(float(np.mean(-s * ex[idx] + c * ey[idx])))
        res_xy += list(np.hypot(ex[idx] - bx, ey[idx] - by))
        res_th += list(eth[idx] - bt)
    out["episodes"] = len(groups)
    out["bias"] = {"pos_mean": float(np.mean(bias_xy)), "pos_max": float(np.max(bias_xy)),
                   "yaw_mean": float(np.mean(bias_th)), "yaw_max": float(np.max(bias_th)),
                   "along_mean": float(np.mean(along)), "across_mean": float(np.mean(across))}
    # Per-axis std of the residual: a 2D isotropic jitter of std s has |res| rms s * sqrt(2).
    out["jitter"] = {"pos_std_per_axis": float(np.sqrt(np.mean(np.square(res_xy)) / 2.0)),
                     "yaw_std": float(np.std(res_th))}
    return out


def main(args=None) -> None:
    spin.init(args)
    node = PoseError()
    spin.spin_until_signal(node, node.stop)


def merge_main() -> None:
    """``pose_error_merge <dir>...``: one summary over several runs' ``samples.csv``."""
    import sys

    rows = []
    for k, d in enumerate(sys.argv[1:]):
        with open(os.path.join(d, "samples.csv")) as f:
            rows += [r + [str(k)] for r in list(csv.reader(f))[1:]]
    print(json.dumps(summary(rows), indent=2))
