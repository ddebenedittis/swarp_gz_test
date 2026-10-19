"""Render the simulation's xacro models to URDF / SDF strings.

Visibility bits: robot ``i`` puts its visuals on bit ``i`` and its lidar masks that bit out, so a
robot never sees its own body; bit 27 holds overlays (goal marker, arena outline) that no lidar
sees. gz-sim spawns the robot's URDF directly (sdformat converts it).
"""

from __future__ import annotations

import os
import tempfile

import xacro
from ament_index_python.packages import get_package_share_directory

MAX_ROBOTS = 26  # bits 1..26; 27 is the overlay bit, 28-31 are gz-rendering's own


def _xacro(path: str, **args) -> str:
    mappings = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in args.items()}
    return xacro.process_file(path, mappings=mappings).toprettyxml(indent="  ")


def limo(robot_name: str, robot_index: int, **args) -> str:
    """URDF of one Limo; ``args`` are the xacro args of limo.urdf.xacro."""
    if not 1 <= robot_index <= MAX_ROBOTS:
        raise ValueError(f"robot_index must be in [1, {MAX_ROBOTS}], got {robot_index}")
    path = os.path.join(
        get_package_share_directory("limo_gz_description"), "urdf", "limo.urdf.xacro"
    )
    return _xacro(path, robot_name=robot_name, robot_index=robot_index, **args)


def tee(name: str = "tee", goal: bool = False, **args) -> str:
    """SDF of the T (or, with ``goal=True``, of the visual-only goal marker)."""
    path = os.path.join(get_package_share_directory("pusht_gz"), "models", "tee.sdf.xacro")
    return _xacro(path, name=name, goal=goal, **args)


def world_file(**args) -> str:
    """Render pusht.sdf.xacro to a temporary .sdf file and return its path."""
    path = os.path.join(get_package_share_directory("pusht_gz"), "worlds", "pusht.sdf.xacro")
    with tempfile.NamedTemporaryFile("w", prefix="pusht_", suffix=".sdf", delete=False) as f:
        f.write(_xacro(path, **args))
    return f.name
