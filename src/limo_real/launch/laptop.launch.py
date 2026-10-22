"""Laptop side of the real setup (in this repo's container): the pose source and one policy
node per robot. The operator console runs in its own terminal (``ros2 run limo_real operator``).

    ros2 launch limo_real laptop.launch.py checkpoint:=/opt/swarp/runs/<run>/<ckpt>.pt pose_source:=mocap \
        lidar_fov:=<measured> [record:=true] [compare:=true map:=...] [rviz:=true]

Each robot runs ``limo_bridge robot.launch.py``; the policies publish ``/robot_i/cmd_vel``
into its bridge. Stopping this launch (Ctrl-C) makes every policy publish a zero twist;
if that is lost, the bridges' 0.3 s watchdog zeroes the chassis anyway.
"""

from __future__ import annotations

import os

import torch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    "checkpoint": ("", "swarp pusht_lidar checkpoint (*.pt)"),
    "pose_source": ("mocap", "mocap | amcl"),
    "lidar_fov": ("6.283185307179586", "the robots' measured lidar field of view [rad]"),
    "lidar_mount_x": ("0.103", "lidar origin ahead of the robot centre [m]"),
    "max_pose_age": ("0.05", "max |pose stamp - scan stamp| for a control step [s]"),
    "sample": ("false", "sample actions instead of greedy"),
    "record": ("false", "per-step npz records under out_dir/rec"),
    "out_dir": (os.path.join(os.environ.get("WS_DIR", "."), "out", "real"), "records root"),
    "mocap_params": ("", "mocap_relay params yaml"),
    "map": ("", "AMCL map yaml (pose_source:=amcl or compare:=true)"),
    "amcl_onboard": ("true", "AMCL runs on the robots (robot.launch.py amcl:=true)"),
    "map_to_arena": ("0,0,0", "map origin x,y,yaw in the arena frame"),
    "initial_poses": ("", "'x,y,yaw;...' AMCL initial poses when AMCL runs here"),
    "compare": ("false", "with mocap: also relay AMCL to /robot_i/amcl/pose and log its error"),
    "rviz": ("false", "RViz (config/real.rviz) and its markers (viz)"),
}


def _setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    ckpt = os.path.abspath(a["checkpoint"])
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint:={a['checkpoint']!r} is not a file")
    n = int(torch.load(ckpt, map_location="cpu", weights_only=False)["n_agents"])
    poses = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("limo_real"), "launch", "poses.launch.py")),
        launch_arguments={
            "pose_source": a["pose_source"], "n_robots": str(n), "mocap_params": a["mocap_params"],
            "map": a["map"], "amcl_onboard": a["amcl_onboard"], "map_to_arena": a["map_to_arena"],
            "initial_poses": a["initial_poses"], "compare": a["compare"],
            "compare_out": os.path.join(a["out_dir"], "amcl_vs_mocap"),
        }.items(),
    )
    record = a["record"].lower() in ("1", "true", "yes")
    policies = [
        Node(
            package="pusht_policy", executable="policy_node", name=f"policy_robot_{i}",
            parameters=[{
                "checkpoint": ckpt, "robot_index": i, "n_robots": n,
                "lidar_fov": float(a["lidar_fov"]), "lidar_mount_x": float(a["lidar_mount_x"]),
                "max_pose_age": float(a["max_pose_age"]),
                "sample": a["sample"].lower() in ("1", "true", "yes"),
                "record_dir": os.path.join(a["out_dir"], "rec") if record else "",
            }],
            output="screen",
        )
        for i in range(1, n + 1)
    ]
    viz = []
    if a["rviz"].lower() in ("1", "true", "yes"):
        viz = [
            Node(package="limo_real", executable="viz", parameters=[{"checkpoint": ckpt}], output="screen"),
            Node(package="rviz2", executable="rviz2", output="log", arguments=[
                "-d", os.path.join(get_package_share_directory("limo_real"), "config", "real.rviz")]),
        ]
    return [poses, *policies, *viz]


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=_setup)]
    )
