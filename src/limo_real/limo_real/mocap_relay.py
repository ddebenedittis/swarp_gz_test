"""Mocap (mocap4ros2 ``/rigid_bodies``) -> the policy's pose topics.

For each rigid body ``bodies[k]`` (Qualisys name, e.g. ``limo_1``) publishes, under the name
``names[k]`` (e.g. ``robot_1``; ``tee``; ``goal_marker``):

* ``/<name>/pose`` (PoseStamped, arena frame): the body's planar pose mapped by
  ``mocap_to_arena`` (the mocap origin's ``x, y, yaw`` in the arena frame), its heading
  corrected by ``yaw_offsets[k]`` (the robot's heading in the rigid body's frame, from how
  the markers were defined) and its position moved by ``offsets_xy`` (the robot centre in
  the rigid body's frame, 2 values per body);
* ``/<name>/vel`` (TwistStamped, arena frame): linear velocity and yaw rate by finite
  differences of consecutive frames, low-passed with time constant ``vel_tau`` (a first-
  order filter, ``vel_tau / 2``-ish of lag at the frequencies that matter; swarp observes
  the true velocity, so keep it short).

A body missing from a frame or with a non-finite pose (occluded markers) is skipped; after
a gap longer than ``max_gap`` its velocity filter restarts at zero. Stamps are the mocap
message's, which must share a clock with the robots' scans (see docs/real_robot.md).
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mocap_msgs.msg import RigidBodies
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from limo_real.frames import compose, pose_msg, stamp_s, vel_msg, wrap, yaw_of


class _Body:
    def __init__(self, node: Node, name: str, yaw_offset: float, offset_xy: tuple[float, float]):
        self.pose_pub = node.create_publisher(PoseStamped, f"/{name}/pose", 10)
        self.vel_pub = node.create_publisher(TwistStamped, f"/{name}/vel", 10)
        self.yaw_offset = yaw_offset
        self.offset = offset_xy
        self.last: tuple[float, float, float, float] | None = None  # t, x, y, yaw
        self.v = (0.0, 0.0, 0.0)


class MocapRelay(Node):
    def __init__(self):
        super().__init__("mocap_relay")
        p = self.declare_parameter
        bodies = list(p("bodies", ["limo_1", "limo_2", "tee"]).value)
        names = list(p("names", ["robot_1", "robot_2", "tee"]).value)
        yaw_off = list(p("yaw_offsets", [0.0] * len(bodies)).value)
        off_xy = list(p("offsets_xy", [0.0] * (2 * len(bodies))).value)
        if not len(names) == len(yaw_off) == len(bodies) or len(off_xy) != 2 * len(bodies):
            raise ValueError("bodies, names, yaw_offsets (1 each) and offsets_xy (2 each) must match")
        self.to_arena = tuple(float(v) for v in p("mocap_to_arena", [0.0, 0.0, 0.0]).value)
        self.frame_id = p("frame_id", "arena").value
        self.tau = float(p("vel_tau", 0.05).value)
        self.max_gap = float(p("max_gap", 0.2).value)
        self.use_header_stamp = bool(p("use_header_stamp", True).value)
        self.bodies = {
            b: _Body(self, n, float(yaw_off[k]), (float(off_xy[2 * k]), float(off_xy[2 * k + 1])))
            for k, (b, n) in enumerate(zip(bodies, names))
        }
        topic = p("topic", "/rigid_bodies").value
        # mocap4ros2 publishes best effort on some drivers, reliable on others: sensor QoS
        # (best effort) accepts both.
        self.create_subscription(RigidBodies, topic, self._on_bodies, qos_profile_sensor_data)
        self.get_logger().info(f"{topic}: {dict(zip(bodies, names))}, mocap origin in arena {self.to_arena}")

    def _on_bodies(self, m: RigidBodies) -> None:
        stamp = m.header.stamp if self.use_header_stamp else self.get_clock().now().to_msg()
        t = stamp_s(stamp)
        for rb in m.rigidbodies:
            body = self.bodies.get(rb.rigid_body_name)
            if body is None:
                continue
            q, pp = rb.pose.orientation, rb.pose.position
            vals = (pp.x, pp.y, q.x, q.y, q.z, q.w)
            if not all(math.isfinite(v) for v in vals) or (q.x == q.y == q.z == q.w == 0.0):
                continue
            # rigid body -> robot centre and heading, then mocap -> arena.
            rb_pose = (pp.x, pp.y, yaw_of(q))
            x, y, yaw = compose(self.to_arena, compose(rb_pose, (body.offset[0], body.offset[1], body.yaw_offset)))
            self._publish(body, stamp, t, x, y, yaw)

    def _publish(self, body: _Body, stamp, t: float, x: float, y: float, yaw: float) -> None:
        if body.last is not None and 0.0 < t - body.last[0] <= self.max_gap:
            dt = t - body.last[0]
            raw = ((x - body.last[1]) / dt, (y - body.last[2]) / dt, wrap(yaw - body.last[3]) / dt)
            a = dt / (self.tau + dt)
            body.v = tuple(v + a * (r - v) for v, r in zip(body.v, raw))
        elif body.last is None or t - body.last[0] > self.max_gap or t < body.last[0]:
            body.v = (0.0, 0.0, 0.0)
        elif t == body.last[0]:
            return  # duplicate frame
        body.last = (t, x, y, yaw)
        body.pose_pub.publish(pose_msg(stamp, self.frame_id, x, y, yaw))
        body.vel_pub.publish(vel_msg(stamp, self.frame_id, *body.v))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MocapRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

