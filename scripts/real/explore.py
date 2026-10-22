"""Slow reactive exploration for autonomous mapping (UNTESTED on the robots).

    laptop.bash py scripts/real/explore.py ROBOT [--duration 300] [--v 0.12] [--w 0.5]

Drives ROBOT through its robot_bridge (start it with robots.bash start map: 0.15 m/s / 0.6 rad/s caps,
arena stop off) while slam_toolbox (laptop.bash slam ROBOT) builds the map: forward while the front
sector is clear, steering toward the more open side; when the front closes in, turns in place toward
the most open bearing; every ~25 s of driving, turns toward the most open bearing anyway, to cover
side branches. Never reverses (the rear is blind). Ctrl-C or the time limit stops it with zeros.

Use robot_2 (lm2935): robot_1's lidar sees nothing between +8 and +92 deg, so its front-left is blind.
"""

import argparse
import math
import random
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

SELF_R = 0.15
FRONT = math.radians(35)
STOP_R = 0.35      # something this close in front: turn
GO_R = 0.8         # front clear beyond this: drive

ap = argparse.ArgumentParser()
ap.add_argument("robot")
ap.add_argument("--duration", type=float, default=300.0)
ap.add_argument("--v", type=float, default=0.12)
ap.add_argument("--w", type=float, default=0.5)
a = ap.parse_args()

rclpy.init()
n = rclpy.create_node("explore")
pub = n.create_publisher(Twist, f"/{a.robot}/cmd_vel", 10)
scan = {}
n.create_subscription(LaserScan, f"/{a.robot}/scan", lambda m: scan.__setitem__("m", m), qos_profile_sensor_data)


def send(v, w):
    t = Twist()
    t.linear.x, t.angular.z = v, w
    pub.publish(t)


def sectors(m):
    r = np.array(m.ranges, dtype=float)
    ang = m.angle_min + np.arange(r.size) * m.angle_increment
    ok = np.isfinite(r) & (r > SELF_R)
    r = np.where(ok, r, np.nan)
    # missing returns inside the driver's span read as far (open space beyond range_max)
    r = np.where(np.isnan(r), m.range_max, r)
    front = r[np.abs(ang) < FRONT]
    left = r[(ang > FRONT) & (ang < math.radians(100))]
    right = r[(ang < -FRONT) & (ang > math.radians(-100))]
    # most open bearing: widest smoothed clearance
    k = max(1, int(math.radians(20) / m.angle_increment))
    smooth = np.convolve(r, np.ones(k) / k, mode="same")
    return (float(np.min(front)) if front.size else 0.0, float(np.median(left)) if left.size else 0.0,
            float(np.median(right)) if right.size else 0.0, float(ang[int(np.argmax(smooth))]))


t0 = time.time()
mode, turn_dir, turn_until, drive_since = "turn", 1.0, 0.0, time.time()
try:
    while time.time() - t0 < a.duration:
        rclpy.spin_once(n, timeout_sec=0.1)
        m = scan.get("m")
        if m is None or time.time() - (m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec) > 1.0:
            send(0.0, 0.0)          # no fresh scan: stand still
            continue
        front, left, right, open_bearing = sectors(m)
        now = time.time()
        if mode == "drive":
            if front < STOP_R or (front < GO_R and now - drive_since > 2.0) or now - drive_since > 25.0:
                mode, turn_dir = "turn", (1.0 if open_bearing > 0 else -1.0)
                turn_until = now + min(abs(open_bearing) / a.w, 6.0) + random.uniform(0.0, 1.0)
            else:
                steer = float(np.clip(0.4 * (left - right), -0.3, 0.3))
                send(a.v if front > GO_R else a.v * 0.5, steer)
                continue
        if mode == "turn":
            if now >= turn_until and front > GO_R:
                mode, drive_since = "drive", now
                continue
            if now >= turn_until:
                turn_until = now + 1.0     # keep turning the same way until the front opens
            send(0.0, turn_dir * a.w)
    print(f"explored {time.time() - t0:.0f} s")
except KeyboardInterrupt:
    pass
finally:
    for _ in range(5):
        send(0.0, 0.0)
        time.sleep(0.05)
    print("stopped", flush=True)
