"""Does the Limo's firmware stop on its own when commands stop?

    # robot: ros2 launch limo_bridge robot.launch.py robot_name:=robot_1 dry_run:=true
    ros2 run limo_real measure_cmd_timeout --ros-args -p robot:=robot_1 -p v:=0.1

Run with the robot on blocks (wheels off the floor) or with >= 2 m free ahead. The bridge
must be in ``dry_run`` (it then sends nothing, so it cannot mask the firmware). Publishes
``v`` straight to the driver (``/<robot>/limo/cmd_vel``) at 20 Hz for ``drive`` seconds,
then goes silent for ``observe`` seconds while watching ``/<robot>/odom``, then sends zeros.
Reports how long after the last command the wheels kept turning (odometry |v| above
``still``): about ``observe`` means there is no firmware timeout, and the onboard
robot_bridge watchdog is the only one.
"""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from limo_real.measure.common import save, wait


class CmdTimeout(Node):
    def __init__(self):
        super().__init__("measure_cmd_timeout")
        p = self.declare_parameter
        self.robot = p("robot", "robot_1").value
        self.v = float(p("v", 0.1).value)
        self.drive = float(p("drive", 1.0).value)
        self.observe = float(p("observe", 3.0).value)
        self.still = float(p("still", 0.01).value)
        self.out = p("out", "out/measure").value
        self.pub = self.create_publisher(Twist, p("driver_cmd_vel", f"/{self.robot}/limo/cmd_vel").value, 10)
        self.odom: list[tuple[float, float]] = []
        self.create_subscription(Odometry, f"/{self.robot}/odom",
                                 lambda m: self.odom.append((time.monotonic(), m.twist.twist.linear.x)), 50)

    def run(self) -> dict:
        if not wait(self, 5.0, lambda: bool(self.odom)):
            raise RuntimeError(f"no /{self.robot}/odom")
        cmd = Twist()
        cmd.linear.x = self.v
        end = time.monotonic() + self.drive
        while time.monotonic() < end:
            self.pub.publish(cmd)
            wait(self, 0.05)
        t_last = time.monotonic()
        wait(self, self.observe)
        for _ in range(5):
            self.pub.publish(Twist())
            wait(self, 0.05)
        before = [v for t, v in self.odom if t_last - 0.3 < t <= t_last]
        after = [(t - t_last, v) for t, v in self.odom if t > t_last]
        moving = [dt for dt, v in after if abs(v) > self.still]
        kept = max(moving) if moving else 0.0
        return {
            "robot": self.robot, "commanded_v": self.v,
            "odom_v_before_silence": sum(before) / max(len(before), 1),
            "kept_moving_for_s": round(kept, 3), "observed_s": self.observe,
            "firmware_timeout": None if kept >= self.observe - 0.1 else round(kept, 3),
            "verdict": "no firmware timeout within the observation" if kept >= self.observe - 0.1
            else f"chassis stopped {kept:.2f} s after the last command",
        }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdTimeout()
    try:
        save(node.out, f"cmd_timeout_{node.robot}.json", node.run())
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()
