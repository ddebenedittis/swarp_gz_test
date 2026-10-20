"""Log a pose estimate against a reference and summarise the error, per robot.

    ros2 run limo_real pose_error --ros-args -p est:=pose -p ref:=gt_pose -p ref_vel:=gt_vel -p out:=out/amcl_err  # Gazebo
    ros2 run limo_real pose_error --ros-args -p est:=amcl/pose -p ref:=pose -p ref_vel:=vel -p out:=out/real_amcl  # mocap

Every ``/<robot>/<est>`` sample is paired with the ``/<robot>/<ref>`` sample nearest its
stamp (within ``max_dt``) and tagged with the running ``/episode`` id (``-1``: between
episodes) and the robot's current **slip**: the difference between its true body-frame
velocity (``/<robot>/<ref_vel>``, world frame, rotated into the body) and its wheel
odometry's (``/<robot>/odom``), i.e. motion the odometry misses, as when the T or the
teammate shoves the robot. On SIGINT/SIGTERM writes ``<out>/samples.csv`` and
``<out>/summary.json`` and prints the summary, over the samples taken during episodes:

* ``pos``: |position error| mean, median, p95, max [m]; ``yaw``: |heading error| [rad];
* ``bias``: per robot and episode, the mean error vector and mean signed heading error;
  their norms' mean, std and max over episodes (swarp's ``pose_bias_std``), and the bias
  split along / across the robot's heading;
* ``jitter``: the std of the error about its episode mean (``pose_jitter_std``);
* ``autocorr_s``: how long the error about the episode mean takes to decorrelate (the lag
  where its autocorrelation drops below 1/e; swarp's ``pose_drift_tau``);
* ``excursions``: stretches with |position error| above ``excursion_m``: count per minute,
  duration and peak (median, p90), and the share that starts within ``shove_window`` s
  after a slip above ``slip_m_s``;
* ``by_motion``: the position / heading error split into samples within ``shove_window``
  s after such a slip ("after_shove") and the rest ("free"), with each share of the time.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32

from limo_bridge import spin
from limo_real.frames import stamp_s, wrap, yaw_of

COLUMNS = ["t", "robot", "episode", "ex", "ey", "eyaw", "ref_yaw", "slip"]
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class PoseError(Node):
    def __init__(self):
        super().__init__("pose_error")
        p = self.declare_parameter
        self.robots = list(p("robots", ["robot_1", "robot_2"]).value)
        est, ref, ref_vel = p("est", "pose").value, p("ref", "gt_pose").value, p("ref_vel", "gt_vel").value
        self.out = p("out", "out/pose_error").value
        self.max_dt = float(p("max_dt", 0.02).value)
        self.ref: dict[str, tuple[list[float], list[tuple[float, float, float]]]] = {
            r: ([], []) for r in self.robots
        }
        self.vel = {r: (0.0, 0.0) for r in self.robots}  # true world-frame velocity
        self.odom_v = {r: 0.0 for r in self.robots}  # odometry forward speed
        self.rows: list[tuple] = []
        self.episode = -1
        self.create_subscription(Int32, "/episode", lambda m: setattr(self, "episode", m.data), LATCHED)
        for r in self.robots:
            self.create_subscription(PoseStamped, f"/{r}/{ref}", lambda m, r=r: self._on_ref(r, m), 50)
            self.create_subscription(PoseStamped, f"/{r}/{est}", lambda m, r=r: self._on_est(r, m), 50)
            self.create_subscription(TwistStamped, f"/{r}/{ref_vel}", lambda m, r=r: self.vel.__setitem__(
                r, (m.twist.linear.x, m.twist.linear.y)), 50)
            self.create_subscription(Odometry, f"/{r}/odom", lambda m, r=r: self.odom_v.__setitem__(
                r, m.twist.twist.linear.x), 50)
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
        vx, vy = self.vel[r]
        c, s = math.cos(gth), math.sin(gth)
        slip = math.hypot(c * vx + s * vy - self.odom_v[r], -s * vx + c * vy)
        self.rows.append((t, r, self.episode, x - gx, y - gy, wrap(th - gth), gth, slip))

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


def _stats(v: np.ndarray) -> dict:
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def _decorrelation_time(t: np.ndarray, res: np.ndarray) -> float | None:
    """Lag [s] where the autocorrelation of ``res`` ([N] or [N, D], about zero) falls below 1/e,
    on a uniform resampling at the median sample spacing."""
    if t.size < 50:
        return None
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None
    grid = np.arange(t[0], t[-1], dt)
    res = res.reshape(t.size, -1)
    r = np.stack([np.interp(grid, t, res[:, k]) for k in range(res.shape[1])], axis=1)
    var = float((r * r).sum())
    if var <= 0:
        return None
    for lag in range(1, grid.size // 2):
        if float((r[lag:] * r[:-lag]).sum()) / var < math.exp(-1.0):
            return lag * dt
    return None


def summary(all_rows, excursion_m: float = 0.15, slip_m_s: float = 0.15, shove_window: float = 5.0) -> dict:
    """The statistics of the module docstring over ``COLUMNS`` rows (plus an optional
    run tag); an episode is identified by ``(robot, episode, run)``."""
    rows = [row for row in all_rows if int(row[2]) >= 0]
    out: dict = {"samples": len(rows)}
    if not rows:
        return out
    t = np.array([float(row[0]) for row in rows])
    ex = np.array([float(row[3]) for row in rows])
    ey = np.array([float(row[4]) for row in rows])
    eth = np.array([float(row[5]) for row in rows])
    gth = np.array([float(row[6]) for row in rows])
    slip = np.array([float(row[7]) if len(row) > 7 and row[7] != "" else 0.0 for row in rows])
    pos = np.hypot(ex, ey)
    out["pos"] = _stats(pos)
    out["yaw"] = _stats(np.abs(eth))

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row[1], row[2], row[8] if len(row) > 8 else "")].append(i)
    bias_xy, bias_th, along, across, res_xy, res_th = [], [], [], [], [], []
    tau_pos, tau_yaw, w_tau = [], [], []
    after = np.zeros(len(rows), dtype=bool)
    exc_dur, exc_peak, exc_shoved = [], [], []
    duration = 0.0
    for idx in groups.values():
        idx = np.array(sorted(idx, key=lambda i: t[i]))
        tt = t[idx]
        duration += tt[-1] - tt[0]
        bx, by, bt = ex[idx].mean(), ey[idx].mean(), eth[idx].mean()
        bias_xy.append(math.hypot(bx, by))
        bias_th.append(bt)
        c, s = np.cos(gth[idx]), np.sin(gth[idx])
        along.append(float(np.mean(c * ex[idx] + s * ey[idx])))
        across.append(float(np.mean(-s * ex[idx] + c * ey[idx])))
        rx, ry, rt = ex[idx] - bx, ey[idx] - by, eth[idx] - bt
        res_xy += list(np.hypot(rx, ry))
        res_th += list(rt)
        tp, ty = _decorrelation_time(tt, np.stack([rx, ry], 1)), _decorrelation_time(tt, rt)
        if tp is not None and ty is not None:
            tau_pos.append(tp)
            tau_yaw.append(ty)
            w_tau.append(tt[-1] - tt[0])
        # within shove_window s after a slip
        shoves = tt[slip[idx] > slip_m_s]
        if shoves.size:
            k = np.searchsorted(shoves, tt, side="right") - 1
            after[idx] = (k >= 0) & (tt - shoves[np.maximum(k, 0)] <= shove_window)
        # excursions
        hi = pos[idx] > excursion_m
        edges = np.flatnonzero(np.diff(np.concatenate([[0], hi.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            exc_dur.append(tt[b - 1] - tt[a])
            exc_peak.append(float(pos[idx][a:b].max()))
            exc_shoved.append(bool(np.any((shoves <= tt[a]) & (shoves >= tt[a] - shove_window))))
    out["episodes"] = len(groups)
    bias_th = np.array(bias_th)
    out["bias"] = {"pos_mean": float(np.mean(bias_xy)), "pos_std": float(np.std(bias_xy)),
                   "pos_max": float(np.max(bias_xy)),
                   "yaw_abs_mean": float(np.mean(np.abs(bias_th))), "yaw_std": float(np.std(bias_th)),
                   "yaw_abs_max": float(np.max(np.abs(bias_th))),
                   "along_mean": float(np.mean(along)), "across_mean": float(np.mean(across))}
    # Per-axis std of the residual: a 2D isotropic jitter of std s has |res| rms s * sqrt(2).
    out["jitter"] = {"pos_std_per_axis": float(np.sqrt(np.mean(np.square(res_xy)) / 2.0)),
                     "yaw_std": float(np.std(res_th))}
    if w_tau:
        out["autocorr_s"] = {"pos": float(np.average(tau_pos, weights=w_tau)),
                             "yaw": float(np.average(tau_yaw, weights=w_tau))}
    exc_dur, exc_peak = np.array(exc_dur), np.array(exc_peak)
    out["excursions"] = {
        "threshold_m": excursion_m, "count": int(exc_dur.size),
        "per_robot_minute": float(exc_dur.size / max(duration / 60.0, 1e-9)),
        "share_of_time": float(np.mean(pos > excursion_m)),
        **({"duration_s_median": float(np.median(exc_dur)), "duration_s_p90": float(np.percentile(exc_dur, 90)),
            "peak_m_median": float(np.median(exc_peak)), "peak_m_p90": float(np.percentile(exc_peak, 90)),
            "share_after_shove": float(np.mean(exc_shoved))} if exc_dur.size else {}),
    }
    out["by_motion"] = {"slip_threshold_m_s": slip_m_s, "window_s": shove_window}
    for name, sel in (("after_shove", after), ("free", ~after)):
        if sel.any():
            out["by_motion"][name] = {"time_share": float(sel.mean()), "pos": _stats(pos[sel]),
                                      "yaw": _stats(np.abs(eth[sel]))}
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
            for r in list(csv.reader(f))[1:]:
                rows.append(r + [""] * (len(COLUMNS) - len(r)) + [str(k)])
    print(json.dumps(summary(rows), indent=2))
