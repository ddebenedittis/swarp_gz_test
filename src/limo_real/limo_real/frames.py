"""Planar pose helpers shared by the relays.

Convention: the **arena frame** is the frame the policy was trained in, origin at the arena
centre, the arena spanning ``[-world_size, world_size]^2``. Every ``/<body>/pose`` and
``/goal`` is in it. A pose source in another frame (mocap's origin, a map built from wherever
SLAM started) is mapped into it by a fixed planar transform ``(x, y, yaw)``: the pose of that
source frame's origin in the arena frame.
"""

from __future__ import annotations

import math

from geometry_msgs.msg import PoseStamped, TwistStamped


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def compose(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    """``a ∘ b``: pose ``b``, given in frame ``a``, expressed in ``a``'s parent frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def rotate(yaw: float, vx: float, vy: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx - s * vy, s * vx + c * vy


def pose_msg(stamp, frame_id: str, x: float, y: float, yaw: float) -> PoseStamped:
    m = PoseStamped()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.pose.position.x, m.pose.position.y = float(x), float(y)
    m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    return m


def vel_msg(stamp, frame_id: str, vx: float, vy: float, w: float) -> TwistStamped:
    m = TwistStamped()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.twist.linear.x, m.twist.linear.y, m.twist.angular.z = float(vx), float(vy), float(w)
    return m


def stamp_s(stamp) -> float:
    return stamp.sec + 1e-9 * stamp.nanosec
