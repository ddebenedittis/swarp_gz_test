"""Wait until every given topic delivered a message (any type, any QoS); exit 1 at the timeout.

    laptop.bash py scripts/real/wait_topics.py [--timeout 60] TOPIC ...
"""

import argparse
import sys
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message

ap = argparse.ArgumentParser()
ap.add_argument("--timeout", type=float, default=60.0)
ap.add_argument("topics", nargs="+")
a = ap.parse_args()
rclpy.init()
n = rclpy.create_node("wait_topics")
got, subs, t0 = set(), {}, time.time()
while len(got) < len(a.topics) and time.time() - t0 < a.timeout:
    for name, types in n.get_topic_names_and_types():
        if name in a.topics and name not in subs:
            # best effort, volatile: matches reliable and sensor-data publishers alike
            subs[name] = n.create_subscription(get_message(types[0]), name, lambda m, t=name: got.add(t),
                                               qos_profile_sensor_data)
    rclpy.spin_once(n, timeout_sec=0.2)
missing = [t for t in a.topics if t not in got]
print(f"{len(got)}/{len(a.topics)} topics up after {time.time() - t0:.1f} s" + (f", missing {missing}" if missing else ""))
sys.exit(1 if missing else 0)
