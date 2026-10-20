"""The command gate of :mod:`limo_bridge.robot_bridge`, free of ROS so it can be unit-tested.

Python 3.8 (ROS 2 Foxy on the Limo Pro): no ``X | Y`` types at runtime, no ``zip(strict=)``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

#: Gate outcomes, in order of precedence. Anything but OK and ARENA outputs a zero twist.
ESTOP = "estop"
STALE = "stale"
NO_POSE = "no_pose"
ARENA = "arena"
OK = "ok"


class Limits(NamedTuple):
    max_v: float = 0.4  # [m/s], the trained action bound
    max_w: float = 1.5  # [rad/s]
    cmd_timeout: float = 0.3  # [s] a command older than this is dropped
    arena_half: float = 2.0  # [m] half side of the arena square; <= 0 disables the arena stop
    arena_margin: float = 0.25  # [m] how far past the arena the centre may go before the stop
    pose_timeout: float = 0.5  # [s] with the arena stop on, an older pose stops the robot


def clamp(x: float, lim: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(-lim, min(lim, x))


def gate(
    cmd: Optional[Tuple[float, float]],
    cmd_age: float,
    estop: bool,
    pose: Optional[Tuple[float, float, float]],
    pose_age: float,
    lim: Limits,
) -> Tuple[float, float, str]:
    """``(v, w, reason)`` to send to the chassis for the latest command ``cmd = (v, w)``.

    * e-stop latched, or no command within ``cmd_timeout`` -> zero;
    * the command is clamped to ``|v| <= max_v``, ``|w| <= max_w`` (non-finite -> 0);
    * with the arena stop on (``arena_half > 0``): no pose within ``pose_timeout`` -> zero;
      the centre past ``arena_half + arena_margin`` on some axis -> the linear velocity is
      zeroed if it points outward along that axis, so the robot can still turn and back in.
    """
    if estop:
        return 0.0, 0.0, ESTOP
    if cmd is None or not cmd_age <= lim.cmd_timeout:
        return 0.0, 0.0, STALE
    v, w = clamp(cmd[0], lim.max_v), clamp(cmd[1], lim.max_w)
    if lim.arena_half <= 0.0:
        return v, w, OK
    if pose is None or not pose_age <= lim.pose_timeout:
        return 0.0, 0.0, NO_POSE
    x, y, yaw = pose
    edge = lim.arena_half + lim.arena_margin
    vx, vy = v * math.cos(yaw), v * math.sin(yaw)
    outward = (abs(x) > edge and vx * x > 0.0) or (abs(y) > edge and vy * y > 0.0)
    if outward:
        return 0.0, w, ARENA
    return v, w, OK


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
