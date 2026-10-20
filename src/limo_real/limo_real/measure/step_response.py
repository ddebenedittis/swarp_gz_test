"""v / ω step responses of one Limo, against mocap (truth) and its own odometry.

    ros2 run limo_real measure_step_response --ros-args -p robot:=robot_1 -p out:=out/measure

Needs ~1.5 m free in front of the robot (it drives forward, back, then turns in place),
the robot_bridge running normally (not dry_run) and, for the true response, mocap
(``/<robot>/pose`` from mocap_relay). Each step holds the command for ``hold`` s, then
``rest`` s of zero; forward steps are followed by an equal reverse step to come back.

Per step it reports, from mocap (and odometry for comparison): the steady-state gain
(measured / commanded over the last half of the hold), the 10-90 % rise time, the implied
acceleration (0.8 · steady / rise), and the dead time to 10 %. These map onto swarp's
``cmd_gain_v``, ``cmd_gain_w``, ``servo_accel`` and ``servo_ang_accel`` ranges and the
control latency. Raw samples go to ``<out>/step_<robot>.csv``.
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from limo_real.frames import yaw_of
from limo_real.measure.common import save, wait

STEPS = [("v", 0.1), ("v", -0.1), ("v", 0.2), ("v", -0.2), ("v", 0.4), ("v", -0.4),
         ("w", 0.5), ("w", -0.5), ("w", 1.0), ("w", -1.0), ("w", 1.5), ("w", -1.5)]


class StepResponse(Node):
    def __init__(self):
        super().__init__("measure_step_response")
        p = self.declare_parameter
        self.robot = p("robot", "robot_1").value
        self.hold = float(p("hold", 2.0).value)
        self.rest = float(p("rest", 1.5).value)
        self.out = p("out", "out/measure").value
        self.pub = self.create_publisher(Twist, f"/{self.robot}/cmd_vel", 10)
        self.mocap: list[tuple[float, float, float, float]] = []  # wall t, x, y, yaw
        self.odom: list[tuple[float, float, float]] = []  # wall t, v, w
        self.create_subscription(PoseStamped, f"/{self.robot}/pose", self._on_pose, 50)
        self.create_subscription(Odometry, f"/{self.robot}/odom", lambda m: self.odom.append(
            (time.monotonic(), m.twist.twist.linear.x, m.twist.twist.angular.z)), 50)

    def _on_pose(self, m: PoseStamped) -> None:
        self.mocap.append((time.monotonic(), m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation)))

    def _drive(self, v: float, w: float, seconds: float) -> None:
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = v, w
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.pub.publish(cmd)
            wait(self, 0.05)

    def _mocap_vel(self, t0: float, t1: float, axis: str):
        s = np.array([r for r in self.mocap if t0 - 0.2 <= r[0] <= t1 + 0.2])
        if len(s) < 10:
            return None
        t = s[:, 0]
        yaw = np.unwrap(s[:, 3])
        if axis == "w":
            sig = np.gradient(yaw, t)
        else:  # forward speed: world velocity projected on the heading
            vx, vy = np.gradient(s[:, 1], t), np.gradient(s[:, 2], t)
            sig = np.cos(yaw) * vx + np.sin(yaw) * vy
        k = max(1, int(round(0.05 / max(np.median(np.diff(t)), 1e-3))))  # 50 ms moving average
        sig = np.convolve(sig, np.ones(k) / k, mode="same")
        return t, sig

    def _analyse(self, t: np.ndarray, sig: np.ndarray, t0: float, cmd: float) -> dict:
        hold = (t >= t0 + self.hold / 2) & (t <= t0 + self.hold)
        if not hold.any():
            return {}
        steady = float(np.mean(sig[hold]))
        frac = sig / steady if abs(steady) > 1e-6 else np.zeros_like(sig)
        after = t >= t0

        def first(level):
            idx = np.flatnonzero(after & (frac >= level))
            return float(t[idx[0]] - t0) if idx.size else math.nan

        t10, t90 = first(0.1), first(0.9)
        rise = t90 - t10
        return {"gain": round(steady / cmd, 3), "steady": round(steady, 4), "dead_time_s": round(t10, 3),
                "rise_10_90_s": round(rise, 3),
                "accel": round(0.8 * abs(steady) / rise, 3) if rise > 0 else math.nan}

    def run(self) -> dict:
        if not wait(self, 5.0, lambda: bool(self.odom)):
            raise RuntimeError(f"no /{self.robot}/odom")
        have_mocap = wait(self, 2.0, lambda: bool(self.mocap))
        marks = []
        self._drive(0.0, 0.0, self.rest)
        for axis, val in STEPS:
            t0 = time.monotonic()
            self._drive(val if axis == "v" else 0.0, val if axis == "w" else 0.0, self.hold)
            marks.append((axis, val, t0))
            self._drive(0.0, 0.0, self.rest)
        results = []
        odom = np.array(self.odom)
        for axis, val, t0 in marks:
            r = {"axis": axis, "cmd": val}
            col = 1 if axis == "v" else 2
            sel = (odom[:, 0] >= t0 - 0.2) & (odom[:, 0] <= t0 + self.hold + 0.2)
            r["odom"] = self._analyse(odom[sel, 0], odom[sel, col], t0, val)
            if have_mocap:
                mv = self._mocap_vel(t0, t0 + self.hold, axis)
                r["mocap"] = self._analyse(*mv, t0, val) if mv else {}
            results.append(r)
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, f"step_{self.robot}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source", "t", "a", "b", "c"])
            w.writerows([("odom", *row) for row in self.odom])
            w.writerows([("mocap", *row) for row in self.mocap])
            w.writerows([("step", t0, axis, val, "") for axis, val, t0 in marks])
        src = "mocap" if have_mocap else "odom"
        gains = {a: [r[src].get("gain") for r in results if r["axis"] == a and r[src]] for a in ("v", "w")}
        return {"robot": self.robot, "source": src, "steps": results,
                "gain_v_range": [min(gains["v"]), max(gains["v"])] if gains["v"] else None,
                "gain_w_range": [min(gains["w"]), max(gains["w"])] if gains["w"] else None}


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StepResponse()
    try:
        save(node.out, f"step_{node.robot}.json", node.run())
    finally:
        for _ in range(3):
            node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()
