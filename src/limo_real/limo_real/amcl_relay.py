"""AMCL -> the policy's pose topics.

For each robot ``r``, on every ``/<r>/odom`` message (wheel odometry, ~50 Hz):

* pose = ``map_to_arena`` ∘ (TF ``map`` -> ``<r>/odom``, AMCL's latest correction) ∘ odom
  pose, published on ``/<r>/<out>`` (``out`` default ``pose``; PoseStamped, arena frame,
  the odometry's stamp). AMCL itself updates only every few centimetres of motion; the
  odometry carries the pose between updates at its own rate, so the policy always finds a
  pose within a few ms of its scan's stamp;
* velocity = the odometry's body-frame twist rotated by that heading, on ``/<r>/<out_vel>``
  (TwistStamped, arena frame). Wheel odometry overestimates turn rates on a skid steer,
  but the policy observes only the linear part.

Convention: AMCL's ``map`` frame is the arena frame (``map_to_arena`` = 0). For a map
built in another frame, ``map_to_arena`` is the map origin's ``x, y, yaw`` in the arena.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener

from limo_real.frames import compose, pose_msg, rotate, vel_msg, yaw_of


class AmclRelay(Node):
    def __init__(self):
        super().__init__("amcl_relay")
        p = self.declare_parameter
        robots = list(p("robots", ["robot_1", "robot_2"]).value)
        self.map_frame = p("map_frame", "map").value
        self.frame_id = p("frame_id", "arena").value
        self.to_arena = tuple(float(v) for v in p("map_to_arena", [0.0, 0.0, 0.0]).value)
        out, out_vel = p("out", "pose").value, p("out_vel", "vel").value
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.static_tf.sendTransform(self._arena_tf())
        for r in robots:
            pubs = (self.create_publisher(PoseStamped, f"/{r}/{out}", 10),
                    self.create_publisher(TwistStamped, f"/{r}/{out_vel}", 10))
            self.create_subscription(Odometry, f"/{r}/odom", lambda m, r=r, pubs=pubs: self._on_odom(r, m, pubs), 20)
        self.get_logger().info(f"AMCL ({self.map_frame}) + odometry of {robots} -> /<robot>/{out}, /<robot>/{out_vel}")

    def _arena_tf(self) -> TransformStamped:
        """The arena's pose in the map frame: the inverse of ``map_to_arena``."""
        x, y, yaw = self.to_arena
        c, s = math.cos(yaw), math.sin(yaw)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = self.map_frame, self.frame_id
        t.transform.translation.x, t.transform.translation.y = -c * x - s * y, s * x - c * y
        t.transform.rotation.z, t.transform.rotation.w = math.sin(-yaw / 2), math.cos(-yaw / 2)
        return t

    def _on_odom(self, robot: str, m: Odometry, pubs) -> None:
        odom_frame = m.header.frame_id or f"{robot}/odom"
        try:
            tf = self.tf.lookup_transform(self.map_frame, odom_frame, Time())
        except TransformException as e:
            self.get_logger().warn(f"{robot}: no {self.map_frame} -> {odom_frame} yet ({e})",
                                   throttle_duration_sec=5.0)
            return
        tr = tf.transform
        map_odom = (tr.translation.x, tr.translation.y, yaw_of(tr.rotation))
        p = m.pose.pose
        x, y, yaw = compose(self.to_arena, compose(map_odom, (p.position.x, p.position.y, yaw_of(p.orientation))))
        vx, vy = rotate(yaw, m.twist.twist.linear.x, m.twist.twist.linear.y)
        pubs[0].publish(pose_msg(m.header.stamp, self.frame_id, x, y, yaw))
        pubs[1].publish(vel_msg(m.header.stamp, self.frame_id, vx, vy, m.twist.twist.angular.z))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AmclRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
