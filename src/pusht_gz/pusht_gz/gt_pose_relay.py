"""Relay Gazebo ground truth to the pose topics the policy reads.

For every body ``b`` in the ``bodies`` parameter, ``/b/gt_odom`` (nav_msgs/Odometry from
gz-sim's OdometryPublisher: world-frame pose, body-frame twist) becomes

* ``/b/gt_pose`` (geometry_msgs/PoseStamped, frame ``world``), and
* ``/b/gt_vel`` (geometry_msgs/TwistStamped, world-frame linear velocity, yaw rate),

and, for the bodies in ``pose_bodies`` (default: all), the same on ``/b/pose`` and ``/b/vel``,
the topics the policy reads. On the real robots a mocap or AMCL relay (limo_real) publishes
those, so the policy cannot tell the sources apart; with a non-ground-truth pose source in
Gazebo the robots are left out of ``pose_bodies`` and ``gt_*`` stays the reference.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GtPoseRelay(Node):
    def __init__(self):
        super().__init__("gt_pose_relay")
        bodies = self.declare_parameter("bodies", ["robot_1", "robot_2", "tee"]).value
        pose_bodies = self.declare_parameter("pose_bodies", list(bodies)).value
        self.frame_id = self.declare_parameter("frame_id", "world").value
        self._subs = []
        for body in bodies:
            pubs = [(self.create_publisher(PoseStamped, f"/{body}/gt_pose", 10),
                     self.create_publisher(TwistStamped, f"/{body}/gt_vel", 10))]
            if body in pose_bodies:
                pubs.append((self.create_publisher(PoseStamped, f"/{body}/pose", 10),
                             self.create_publisher(TwistStamped, f"/{body}/vel", 10)))
            self._subs.append(
                self.create_subscription(
                    Odometry, f"/{body}/gt_odom", lambda msg, pubs=pubs: self._relay(msg, pubs), 10
                )
            )
        self.get_logger().info(f"relaying ground truth of {list(bodies)}, as the pose of {list(pose_bodies)}")

    def _relay(self, odom: Odometry, pubs) -> None:
        pose = PoseStamped()
        pose.header.stamp = odom.header.stamp
        pose.header.frame_id = self.frame_id
        pose.pose = odom.pose.pose

        # Body-frame twist -> world frame.
        yaw = yaw_of(odom.pose.pose.orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        vb = odom.twist.twist.linear
        vel = TwistStamped()
        vel.header = pose.header
        vel.twist.linear.x = c * vb.x - s * vb.y
        vel.twist.linear.y = s * vb.x + c * vb.y
        vel.twist.angular.z = odom.twist.twist.angular.z
        for pose_pub, vel_pub in pubs:
            pose_pub.publish(pose)
            vel_pub.publish(vel)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GtPoseRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
