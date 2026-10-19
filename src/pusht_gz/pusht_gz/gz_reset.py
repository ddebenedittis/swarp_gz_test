"""Teleport Gazebo models (robots, T, goal marker) through the world's ``set_pose`` service.

Uses the gz-transport Python bindings (``gz.transport13``, shipped by ROS Jazzy's
gz_transport_vendor) and falls back to the ``gz service`` CLI when they cannot be imported. No
ROS node is needed, so this works from any process inside the container::

    from pusht_gz.gz_reset import GzWorld
    world = GzWorld("pusht")
    world.set_pose("tee", 0.3, -0.2, 1.0)
    world.set_poses({"robot_1": (-1.0, 0.0, 0.0), "robot_2": (1.0, 0.0, 3.14)})

Teleporting keeps a body's velocity (gz-physics only rewrites the pose). For a reset at rest,
stop the robots first (zero ``cmd_vel``) and use :meth:`GzWorld.set_poses_at_rest`, which
teleports, lets friction stop whatever is still sliding, and teleports again.
"""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Mapping

try:
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.clock_pb2 import Clock
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.msgs10.world_control_pb2 import WorldControl
    from gz.transport13 import Node

    HAVE_GZ_PY = True
except ImportError:
    HAVE_GZ_PY = False

#: ``(x, y, yaw)`` or ``(x, y, yaw, z)``.
Pose2D = tuple[float, ...]


def _fill_pose(msg, name: str, pose: Pose2D) -> None:
    x, y, yaw = pose[:3]
    z = pose[3] if len(pose) > 3 else 0.0
    msg.name = name
    msg.position.x, msg.position.y, msg.position.z = float(x), float(y), float(z)
    msg.orientation.x = msg.orientation.y = 0.0
    msg.orientation.z, msg.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class GzWorld:
    """Pose and run-state control of one running Gazebo world."""

    def __init__(self, world: str = "pusht", timeout_ms: int = 2000):
        self.world = world
        self.timeout_ms = timeout_ms
        self._node = Node() if HAVE_GZ_PY else None
        self._sim_time = None

    def _on_clock(self, msg) -> None:
        self._sim_time = msg.sim.sec + 1e-9 * msg.sim.nsec

    def _request(self, service: str, req, req_type, rep_type=None) -> bool:
        rep_type = rep_type or Boolean
        ok, rep = self._node.request(service, req, req_type, rep_type, self.timeout_ms)
        return bool(ok and rep.data)

    def set_pose(self, name: str, x: float, y: float, yaw: float, z: float = 0.0) -> bool:
        """Teleport model ``name`` to ``(x, y, z)`` with heading ``yaw``. True on success."""
        service = f"/world/{self.world}/set_pose"
        if self._node is None:
            return self._cli_set_pose(service, name, (x, y, yaw, z))
        req = Pose()
        _fill_pose(req, name, (x, y, yaw, z))
        return self._request(service, req, Pose)

    def set_poses(self, poses: Mapping[str, Pose2D]) -> bool:
        """Teleport several models in one request (applied in the same simulation step)."""
        if self._node is None:
            return all(self.set_pose(n, *p[:3], *p[3:]) for n, p in poses.items())
        req = Pose_V()
        for name, pose in poses.items():
            _fill_pose(req.pose.add(), name, pose)
        return self._request(f"/world/{self.world}/set_pose_vector", req, Pose_V)

    def pause(self, paused: bool = True) -> bool:
        """Pause or resume the simulation."""
        if self._node is None:
            return self._cli(f"/world/{self.world}/control", "gz.msgs.WorldControl",
                             f"pause: {str(paused).lower()}")
        req = WorldControl()
        req.pause = paused
        return self._request(f"/world/{self.world}/control", req, WorldControl)

    def last_sim_time(self) -> float | None:
        """Simulation time at the end of the last :meth:`wait_sim` (None before one ran)."""
        return self._sim_time

    def wait_sim(self, dt: float, timeout: float = 30.0) -> bool:
        """Block until the simulation clock advances by ``dt`` seconds (False on timeout).

        The clock is subscribed to only while waiting: the Python bindings keep the GIL during
        ``request()``, so a live Python subscription callback stalls the transport thread and
        every service reply times out.
        """
        if self._node is None:
            return False
        topic = f"/world/{self.world}/clock"
        self._sim_time = None
        self._node.subscribe(Clock, topic, self._on_clock)
        try:
            deadline = time.monotonic() + timeout
            while self._sim_time is None:
                if time.monotonic() > deadline:
                    return False
                time.sleep(0.005)
            t0 = self._sim_time
            while self._sim_time - t0 < dt:
                if time.monotonic() > deadline:
                    return False
                time.sleep(0.002)
            return True
        finally:
            self._node.unsubscribe(topic)

    def set_poses_at_rest(self, poses: Mapping[str, Pose2D], settle: float = 0.3) -> bool:
        """Teleport, wait ``settle`` s of sim time for friction to stop residual motion, repeat.

        Needs the simulation running. A T sliding at 0.5 m/s on mu = 0.3 stops in ~0.17 s; robots
        stop as fast as their wheels, so command zero velocity before calling this.
        """
        if not self.set_poses(poses):
            return False
        if not self.wait_sim(settle):
            return False
        return self.set_poses(poses)

    # gz CLI fallback (no Python bindings) ---------------------------------------------------

    def _cli(self, service: str, reqtype: str, req: str) -> bool:
        out = subprocess.run(
            ["gz", "service", "-s", service, "--reqtype", reqtype, "--reptype", "gz.msgs.Boolean",
             "--timeout", str(self.timeout_ms), "--req", req],
            capture_output=True, text=True,
        )
        return out.returncode == 0 and "data: true" in out.stdout

    def _cli_set_pose(self, service: str, name: str, pose: Pose2D) -> bool:
        x, y, yaw = pose[:3]
        z = pose[3] if len(pose) > 3 else 0.0
        req = (
            f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
            f"orientation: {{x: 0, y: 0, z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}"
        )
        return self._cli(service, "gz.msgs.Pose", req)


_default: GzWorld | None = None


def set_pose(name: str, x: float, y: float, yaw: float, z: float = 0.0, world: str = "pusht") -> bool:
    """Module-level shortcut: teleport ``name`` in ``world`` (shares one gz-transport node)."""
    global _default
    if _default is None or _default.world != world:
        _default = GzWorld(world)
    return _default.set_pose(name, x, y, yaw, z)


def main() -> None:
    """CLI: ``ros2 run pusht_gz set_pose <name> <x> <y> <yaw> [--world pusht]``."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("name")
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("yaw", type=float)
    parser.add_argument("--world", default="pusht")
    args = parser.parse_args()
    ok = set_pose(args.name, args.x, args.y, args.yaw, world=args.world)
    print("ok" if ok else "failed")
    raise SystemExit(0 if ok else 1)
