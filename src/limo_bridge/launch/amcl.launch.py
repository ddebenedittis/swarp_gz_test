"""nav2 AMCL for one robot, in its namespace: map_server + amcl + lifecycle_manager.

    ros2 launch limo_bridge amcl.launch.py robot_name:=robot_1 map:=/path/arena.yaml x:=-1 y:=0 yaw:=0

Reads ``/<robot>/scan_loc`` (robot_bridge's scan without the arena's contents) and the TF ``<robot>/odom`` -> ``<robot>/base_footprint`` ->
``<robot>/laser_link`` (robot_bridge, or the Gazebo sim), publishes ``/<robot>/amcl_pose``
and TF ``map`` -> ``<robot>/odom``. The map frame is the arena frame: the map's origin is
the arena centre, x along the arena's +x (see limo_real's amcl_relay ``map_to_arena`` for a
map built in some other frame). Set the pose later on ``/<robot>/initialpose``.

Runs on Foxy (onboard), Humble and Jazzy: the only distro difference handled here is
``robot_model_type``.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    "robot_name": ("robot_1", "namespace and frame prefix"),
    "map": ("", "map yaml (nav2 map_server format), in the arena frame"),
    "params": ("", "amcl params yaml; default limo_bridge/config/amcl.yaml"),
    "use_sim_time": ("false", "true in Gazebo"),
    "scan": ("scan_loc", "scan topic under /<robot>/: robot_bridge's scan_loc has the arena's T and robots removed"),
    "x": ("0.0", "initial x [m]"),
    "y": ("0.0", "initial y [m]"),
    "yaw": ("0.0", "initial yaw [rad]"),
}


def _setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    robot = a["robot_name"]
    if not a["map"]:
        raise ValueError("map:=<yaml> is required")
    params = a["params"] or os.path.join(get_package_share_directory("limo_bridge"), "config", "amcl.yaml")
    sim = a["use_sim_time"].lower() in ("1", "true", "yes")
    foxy = os.environ.get("ROS_DISTRO", "") == "foxy"
    amcl = {
        "use_sim_time": sim,
        "base_frame_id": f"{robot}/base_footprint",
        "odom_frame_id": f"{robot}/odom",
        "global_frame_id": "map",
        "scan_topic": f"/{robot}/{a['scan']}",
        "map_topic": f"/{robot}/map",
        "robot_model_type": "differential" if foxy else "nav2_amcl::DifferentialMotionModel",
        "set_initial_pose": True,
        "initial_pose.x": float(a["x"]),
        "initial_pose.y": float(a["y"]),
        "initial_pose.z": 0.0,
        "initial_pose.yaw": float(a["yaw"]),
    }
    return [
        Node(
            package="nav2_map_server", executable="map_server", name="map_server", namespace=robot,
            parameters=[{"yaml_filename": a["map"], "use_sim_time": sim, "topic_name": f"/{robot}/map",
                         "frame_id": "map"}],
            output="screen",
        ),
        Node(
            package="nav2_amcl", executable="amcl", name="amcl", namespace=robot,
            parameters=[params, amcl],
            remappings=[("map", f"/{robot}/map")],
            output="screen",
        ),
        Node(
            package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_amcl",
            namespace=robot,
            parameters=[{"autostart": True, "node_names": ["map_server", "amcl"], "use_sim_time": sim,
                         "bond_timeout": 0.0}],
            output="screen",
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=_setup)]
    )
