"""The pose source behind ``/robot_i/pose`` and ``/robot_i/vel`` (and ``/tee/pose``).

    ros2 launch limo_real poses.launch.py pose_source:=mocap|amcl n_robots:=2 [map:=...] [amcl_onboard:=true]

* ``mocap``: ``mocap_relay`` on ``/rigid_bodies`` (``mocap_params``, default
  ``config/mocap.yaml``), which also gives ``/tee/pose`` for scoring;
* ``amcl``: one namespaced nav2 AMCL per robot on ``map`` (unless ``amcl_onboard``, when the
  robots run it, see limo_bridge robot.launch.py) and ``amcl_relay`` -> ``/robot_i/<amcl_out>``;
* ``gt``: nothing (Gazebo's gt_pose_relay already publishes the poses).

With ``compare:=true`` (mocap) AMCL runs too, relayed to ``/robot_i/amcl/pose`` instead, and
``pose_error`` logs it against the mocap pose into ``compare_out``.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    "pose_source": ("mocap", "mocap | amcl | gt"),
    "n_robots": ("2", "robots robot_1..robot_N"),
    "use_sim_time": ("false", "true in Gazebo"),
    "mocap_params": ("", "mocap_relay params; default limo_real/config/mocap.yaml"),
    "map": ("", "AMCL map yaml (arena frame)"),
    "amcl_params": ("", "AMCL params yaml; default limo_bridge/config/amcl.yaml"),
    "amcl_scan": ("scan_loc", "AMCL's scan under /robot_i/ (scan_loc: arena contents removed by robot_bridge)"),
    "amcl_onboard": ("false", "AMCL runs on the robots: only relay it here"),
    "map_to_arena": ("0,0,0", "map origin x,y,yaw in the arena frame"),
    "initial_poses": ("", "'x,y,yaw;x,y,yaw' AMCL initial poses (default all zero)"),
    "compare": ("false", "mocap: also run AMCL and log its error against mocap"),
    "compare_out": ("out/amcl_vs_mocap", "pose_error output directory"),
}


def _true(s: str) -> bool:
    return s.lower() in ("1", "true", "yes")


def _amcl(a, robots, sim, out):
    poses = [p.split(",") for p in a["initial_poses"].split(";") if p.strip()] or [["0", "0", "0"]] * len(robots)
    acts = []
    if not _true(a["amcl_onboard"]):
        launch = os.path.join(get_package_share_directory("limo_bridge"), "launch", "amcl.launch.py")
        for r, (x, y, yaw) in zip(robots, poses):
            acts.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(launch),
                launch_arguments={"robot_name": r, "map": a["map"], "params": a["amcl_params"], "scan": a["amcl_scan"],
                                  "use_sim_time": str(sim).lower(),
                                  "x": x, "y": y, "yaw": yaw}.items(),
            ))
    acts.append(Node(
        package="limo_real", executable="amcl_relay",
        parameters=[{"robots": robots, "out": out, "out_vel": "vel" if out == "pose" else out.replace("pose", "vel"),
                     "map_to_arena": [float(v) for v in a["map_to_arena"].split(",")], "use_sim_time": sim}],
        output="screen",
    ))
    return acts


def _setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    robots = [f"robot_{i}" for i in range(1, int(a["n_robots"]) + 1)]
    sim = _true(a["use_sim_time"])
    src = a["pose_source"]
    if src == "gt":
        return []
    if src == "amcl":
        return _amcl(a, robots, sim, "pose")
    if src != "mocap":
        raise ValueError(f"pose_source must be gt, mocap or amcl, got {src!r}")
    params = a["mocap_params"] or os.path.join(get_package_share_directory("limo_real"), "config", "mocap.yaml")
    acts = [Node(package="limo_real", executable="mocap_relay", parameters=[params, {"use_sim_time": sim}],
                 output="screen")]
    if _true(a["compare"]):
        acts += _amcl(a, robots, sim, "amcl/pose")
        acts.append(Node(
            package="limo_real", executable="pose_error",
            parameters=[{"robots": robots, "est": "amcl/pose", "ref": "pose", "ref_vel": "vel", "out": a["compare_out"],
                         "use_sim_time": sim}],
            output="screen",
        ))
    return acts


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=_setup)]
    )
