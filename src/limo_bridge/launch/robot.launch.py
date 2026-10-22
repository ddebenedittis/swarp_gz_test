"""Robot side: run on each Limo (Foxy on the Limo Pro, Humble on the Limo ROS2).

    ros2 launch limo_bridge robot.launch.py robot_name:=robot_1 [dry_run:=true] [max_v:=0.2] [amcl:=true map:=...]

Starts AgileX's ``limo_base`` driver and the T-mini Pro (``ydlidar_ros2_driver``, full
circle at 10 Hz) with their absolute topics remapped under ``/<robot>/limo/`` (so nothing
else in the shared ROS domain can drive this robot through a bare ``/cmd_vel``), and
:mod:`limo_bridge.robot_bridge` between them and the laptop. With ``driver:=false`` /
``lidar:=false`` AgileX's own bringup runs instead and the bridge uses its plain
``/cmd_vel``, ``/scan``, ``/odom``. With ``amcl:=true`` AMCL localizes onboard on ``map``.

Out: ``/<robot>/scan``, ``/<robot>/odom``, ``/<robot>/cmd_vel_gated``, ``/<robot>/bridge_state``,
TF ``<robot>/odom`` -> ``<robot>/base_footprint`` -> ``<robot>/laser_link``.
In: ``/<robot>/cmd_vel``, ``/<robot>/pose`` (arena stop), ``/estop``.
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
    "robot_name": ("robot_1", "robot_i: namespace, frame prefix, and the policy's agent index"),
    "driver": ("true", "start limo_base here (false: AgileX's bringup is already running)"),
    "lidar": ("true", "start the T-mini Pro driver here"),
    "port_name": ("ttylimo", "limo_base serial port (under /dev)"),
    "lidar_port": ("/dev/ydlidar", "T-mini Pro serial port"),
    "lidar_params": ("", "ydlidar params yaml; default limo_bridge/config/ydlidar_tmini_pro.yaml"),
    "dry_run": ("false", "shadow mode: never send anything to the chassis"),
    "max_v": ("0.4", "linear speed cap [m/s] (trained bound 0.4)"),
    "max_w": ("1.5", "angular speed cap [rad/s] (trained bound 1.5)"),
    "cmd_timeout": ("0.3", "watchdog: zero the chassis when no command for this long [s]"),
    "arena_half": ("2.0", "arena half side [m]; 0 disables the arena stop"),
    "arena_margin": ("0.25", "allowed overshoot past the arena edge [m]"),
    "pose_timeout": ("0.5", "with the arena stop on, stop when /<robot>/pose is older [s]"),
    "scan_yaw_offset": ("0.0", "added to the scan angles so that 0 is the robot's heading [rad]"),
    "loc_half": ("2.35", "/<robot>/scan_loc (AMCL's input) blanks returns inside this square around the "
                         "arena centre [m]; 0: off"),
    "laser_xyz": ("0.103,0.0,0.116", "lidar origin from base_footprint [m]"),
    "amcl": ("false", "run AMCL onboard"),
    "map": ("", "map yaml for amcl (arena frame)"),
    "amcl_params": ("", "amcl params yaml; default limo_bridge/config/amcl.yaml"),
    "x": ("0.0", "AMCL initial x"),
    "y": ("0.0", "AMCL initial y"),
    "yaw": ("0.0", "AMCL initial yaw"),
}


def _true(s: str) -> bool:
    return s.lower() in ("1", "true", "yes")


def _setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    robot = a["robot_name"]
    share = get_package_share_directory("limo_bridge")
    limo = f"/{robot}/limo"
    actions = []
    if _true(a["driver"]):
        actions.append(Node(
            package="limo_base", executable="limo_base", name="limo_base_node", namespace=robot,
            parameters=[{"port_name": a["port_name"], "odom_frame": f"{robot}/odom",
                         "base_frame": f"{robot}/base_footprint", "pub_odom_tf": False,
                         "use_mcnamu": False}],
            # AgileX's limo_base versions disagree on absolute vs relative names (cmd_vel and odom
            # are relative on the Limo Pro's Foxy build): remap both forms, or a relative cmd_vel
            # lands on /<robot>/cmd_vel, the policy's ungated output.
            remappings=[(n, f"{limo}/{t}") for t in ("cmd_vel", "odom", "tf") for n in (t, f"/{t}")]
            + [(n, f"/{robot}/{t}") for t in ("imu", "limo_status") for n in (t, f"/{t}")],
            output="screen",
        ))
    if _true(a["lidar"]):
        actions.append(Node(
            package="ydlidar_ros2_driver", executable="ydlidar_ros2_driver_node",
            name="ydlidar_ros2_driver_node", namespace=robot,
            parameters=[a["lidar_params"] or os.path.join(share, "config", "ydlidar_tmini_pro.yaml"),
                        {"port": a["lidar_port"], "frame_id": f"{robot}/laser_link"}],
            remappings=[("scan", f"{limo}/scan"), ("/scan", f"{limo}/scan")],
            output="screen",
        ))
    actions.append(Node(
        package="limo_bridge", executable="robot_bridge", name="robot_bridge", namespace=robot,
        parameters=[{
            "robot_name": robot,
            "dry_run": _true(a["dry_run"]),
            "max_v": float(a["max_v"]),
            "max_w": float(a["max_w"]),
            "cmd_timeout": float(a["cmd_timeout"]),
            "arena_half": float(a["arena_half"]),
            "arena_margin": float(a["arena_margin"]),
            "pose_timeout": float(a["pose_timeout"]),
            "scan_yaw_offset": float(a["scan_yaw_offset"]),
            "loc_half": float(a["loc_half"]),
            "laser_xyz": [float(v) for v in a["laser_xyz"].split(",")],
            "driver_cmd_vel": f"{limo}/cmd_vel" if _true(a["driver"]) else "/cmd_vel",
            "driver_odom": f"{limo}/odom" if _true(a["driver"]) else "/odom",
            "driver_scan": f"{limo}/scan" if _true(a["lidar"]) else "/scan",
        }],
        output="screen",
    ))
    if _true(a["amcl"]):
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "amcl.launch.py")),
            launch_arguments={"robot_name": robot, "map": a["map"], "params": a["amcl_params"], "x": a["x"],
                              "y": a["y"], "yaw": a["yaw"]}.items(),
        ))
    return actions


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=_setup)]
    )
