"""Gazebo stand-in for the Qualisys + mocap4ros2 driver: ``/<body>/gt_odom`` ->
``/rigid_bodies`` (mocap_msgs/RigidBodies), so ``pose_source:=mocap`` runs the real
``mocap_relay`` in simulation.

Each frame holds the latest ground-truth pose of every body in ``bodies``, named
``names[k]`` (``limo_1``, ``limo_2``, ``tee``, as in the lab), plus Gaussian noise of
``noise_xy`` [m] and ``noise_yaw`` [rad] per frame. A frame goes out on every ground-truth
message of the first body (50 Hz in the sim; Qualisys runs 100-300 Hz).
"""

from __future__ import annotations

import math
import random

import rclpy
from mocap_msgs.msg import RigidBodies, RigidBody
from nav_msgs.msg import Odometry
from rclpy.node import Node

from limo_real.frames import yaw_of


class FakeMocap(Node):
    def __init__(self):
        super().__init__("fake_mocap")
        p = self.declare_parameter
        bodies = list(p("bodies", ["robot_1", "robot_2", "tee"]).value)
        names = list(p("names", ["limo_1", "limo_2", "tee"]).value)
        self.noise_xy = float(p("noise_xy", 0.001).value)
        self.noise_yaw = float(p("noise_yaw", 0.003).value)
        self.rng = random.Random(int(p("seed", 0).value))
        self.latest: dict[str, Odometry] = {}
        self.names = dict(zip(bodies, names))
        self.pub = self.create_publisher(RigidBodies, "/rigid_bodies", 10)
        self.frame = 0
        for b in bodies:
            self.create_subscription(Odometry, f"/{b}/gt_odom", lambda m, b=b: self._on_odom(b, m), 10)
        self.first = bodies[0]

    def _on_odom(self, body: str, m: Odometry) -> None:
        self.latest[body] = m
        if body != self.first:
            return
        out = RigidBodies()
        out.header.stamp = m.header.stamp
        out.header.frame_id = "mocap"
        out.frame_number = self.frame
        self.frame += 1
        for b, o in self.latest.items():
            rb = RigidBody()
            rb.rigid_body_name = self.names[b]
            p = o.pose.pose.position
            yaw = yaw_of(o.pose.pose.orientation) + self.rng.gauss(0.0, self.noise_yaw)
            rb.pose.position.x = p.x + self.rng.gauss(0.0, self.noise_xy)
            rb.pose.position.y = p.y + self.rng.gauss(0.0, self.noise_xy)
            rb.pose.position.z = p.z
            rb.pose.orientation.z, rb.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            out.rigidbodies.append(rb)
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeMocap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
