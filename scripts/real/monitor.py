"""Print both robots' arena poses, gated commands and bridge gates every PERIOD s, until the episode
ends (/episode back to -1) or DUR s pass.

    laptop.bash py scripts/real/monitor.py [DUR] [PERIOD]

Gate `arena` for many seconds in a row means a robot is pinned at the arena stop: stop the episode.
"""

import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
period = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
LATCHED = QoSProfile(depth=5, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
ROBOTS = ("robot_1", "robot_2")

rclpy.init()
n = rclpy.create_node("monitor")
s = {}
for r in ROBOTS:
    n.create_subscription(PoseStamped, f"/{r}/pose", lambda m, r=r: s.__setitem__(r + "p", m), 10)
    n.create_subscription(Twist, f"/{r}/cmd_vel_gated", lambda m, r=r: s.__setitem__(r + "c", m), 10)
    n.create_subscription(String, f"/{r}/bridge_state", lambda m, r=r: s.__setitem__(r + "g", m.data), LATCHED)
n.create_subscription(Int32, "/episode", lambda m: s.__setitem__("ep", m.data), LATCHED)
pinned = {r: 0 for r in ROBOTS}
t0 = time.time()
nxt = t0
while time.time() - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.05)
    if time.time() < nxt:
        continue
    nxt += period
    out = [f"t={time.time() - t0:5.1f} ep={s.get('ep')}"]
    for r in ROBOTS:
        p, c, g = s.get(r + "p"), s.get(r + "c"), s.get(r + "g", "?")
        q = p.pose.orientation if p else None
        ps = (f"({p.pose.position.x:+.2f},{p.pose.position.y:+.2f},{math.degrees(2 * math.atan2(q.z, q.w)):+4.0f})"
              if p else "-")
        cs = f"v{c.linear.x:+.2f} w{c.angular.z:+.2f}" if c else "-"
        pinned[r] = pinned[r] + 1 if g.endswith("arena") else 0
        out.append(f"{r[-1]}: {ps} {cs} {g}" + ("  PINNED" if pinned[r] * period >= 9 else ""))
    print(" | ".join(out), flush=True)
    if s.get("ep") == -1 and time.time() - t0 > 5:
        break
