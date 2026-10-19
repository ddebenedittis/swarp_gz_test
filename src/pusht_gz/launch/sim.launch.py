"""Push-T in Gazebo Harmonic: world, N Limos, the T, the goal marker, the ROS bridge.

    ros2 launch pusht_gz sim.launch.py n_robots:=2 headless:=true rtf:=0

Per robot ``robot_i`` (i = 1..N): /robot_i/{cmd_vel, odom, scan, gt_odom, joint_states, pose, vel}.
T: /tee/{gt_odom, pose, vel}. Ground-truth TF world -> robot_i/base_footprint and world -> tee.
Teleport anything with the /world/pusht/set_pose service (ROS, bridged) or pusht_gz.gz_reset.
"""

from __future__ import annotations

import math
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pusht_gz import models

WORLD = "pusht"

ARGS = {
    "n_robots": ("2", "number of Limos, named robot_1..robot_N"),
    "headless": ("false", "server only, headless EGL rendering (no GUI)"),
    "rtf": ("1.0", "target real-time factor; 0 runs as fast as possible"),
    "step": ("0.001", "physics step [s]"),
    "physics_engine": ("dart", "dart or bullet-featherstone"),
    "arena_half": ("2.0", "half side of the drawn arena outline [m] (no walls)"),
    "robot_poses": ("", "'x,y,yaw;x,y,yaw;...' one per robot; default: circle of radius 1.4"),
    "use_mesh": ("true", "Limo meshes as visuals (false: boxes)"),
    "max_linear_acceleration": ("1.0", "diff-drive linear acceleration limit [m/s^2]"),
    "max_angular_acceleration": ("3.0", "diff-drive angular acceleration limit [rad/s^2]"),
    "wheel_mu": ("1.0", "wheel friction along the rolling direction"),
    "wheel_mu_lateral": ("0.2", "wheel friction along the axle (skid steer needs it low to turn)"),
    "body_mu": ("0.1", "robot body friction (robot-T contact)"),
    "lidar_noise_stddev": ("0.01", "lidar Gaussian range noise [m]"),
    "tee_bar_len": ("1.0", "T crossbar length [m]"),
    "tee_bar_width": ("0.3", "T crossbar width [m]"),
    "tee_stem_len": ("0.7", "T stem length [m]"),
    "tee_stem_width": ("0.3", "T stem width [m]"),
    "tee_height": ("0.25", "T height [m]"),
    "tee_mass": ("0.5", "T mass [kg]"),
    "tee_mu": ("0.3", "T friction coefficient (floor and robots; the floor itself has mu 1)"),
    "tee_floor_friction": ("coulomb", "coulomb (isotropic, PlanarCoulombFriction) or contact (DART's)"),
    "tee_contact_sensor": ("false", "publish the T bar contacts on gz (diagnostics)"),
    "tee_x": ("0.0", "T initial x [m]"),
    "tee_y": ("0.0", "T initial y [m]"),
    "tee_yaw": ("0.0", "T initial yaw [rad]"),
    "goal_x": ("0.8", "goal marker x [m]"),
    "goal_y": ("0.8", "goal marker y [m]"),
    "goal_yaw": ("0.5", "goal marker yaw [rad]"),
    "rviz": ("false", "start RViz"),
}


def _robot_poses(spec: str, n: int) -> list[tuple[float, float, float]]:
    if spec.strip():
        poses = [tuple(float(v) for v in p.split(",")) for p in spec.split(";") if p.strip()]
        if len(poses) != n or any(len(p) != 3 for p in poses):
            raise ValueError(f"robot_poses needs {n} 'x,y,yaw' entries, got {spec!r}")
        return poses
    # On a circle around the T, facing the centre (robot_1 at -x).
    poses = []
    for i in range(n):
        a = math.pi + 2.0 * math.pi * i / n
        poses.append((1.4 * math.cos(a), 1.4 * math.sin(a), a + math.pi))
    return poses


def _bridge_entry(ros: str, ros_type: str, gz_type: str, direction: str, gz: str | None = None):
    return {
        "ros_topic_name": ros,
        "gz_topic_name": gz or ros,
        "ros_type_name": ros_type,
        "gz_type_name": gz_type,
        "direction": direction,
    }


def _bridge_config(robots: list[str]) -> list[dict]:
    odom = ("nav_msgs/msg/Odometry", "gz.msgs.Odometry", "GZ_TO_ROS")
    tf = ("tf2_msgs/msg/TFMessage", "gz.msgs.Pose_V", "GZ_TO_ROS")
    cfg = [_bridge_entry("/clock", "rosgraph_msgs/msg/Clock", "gz.msgs.Clock", "GZ_TO_ROS")]
    for r in robots:
        cfg += [
            _bridge_entry(f"/{r}/cmd_vel", "geometry_msgs/msg/Twist", "gz.msgs.Twist", "ROS_TO_GZ"),
            _bridge_entry(f"/{r}/odom", *odom),
            _bridge_entry(f"/{r}/gt_odom", *odom),
            _bridge_entry("/tf", *tf, gz=f"/{r}/gt_tf"),
            _bridge_entry(f"/{r}/scan", "sensor_msgs/msg/LaserScan", "gz.msgs.LaserScan", "GZ_TO_ROS"),
            _bridge_entry(f"/{r}/joint_states", "sensor_msgs/msg/JointState", "gz.msgs.Model", "GZ_TO_ROS"),
        ]
    cfg += [_bridge_entry("/tee/gt_odom", *odom), _bridge_entry("/tf", *tf, gz="/tee/gt_tf")]
    cfg.append({
        "service_name": f"/world/{WORLD}/set_pose",
        "ros_type_name": "ros_gz_interfaces/srv/SetEntityPose",
        "gz_req_type_name": "gz.msgs.Pose",
        "gz_rep_type_name": "gz.msgs.Boolean",
    })
    return cfg


def _spawn(name: str, model: str, x: float, y: float, yaw: float, z: float = 0.0) -> Node:
    return Node(
        package="ros_gz_sim",
        executable="create",
        name=f"spawn_{name}",
        arguments=["-world", WORLD, "-name", name, "-string", model,
                   "-x", str(x), "-y", str(y), "-z", str(z), "-Y", str(yaw)],
        output="screen",
    )


def _setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    n = int(a["n_robots"])
    headless = a["headless"].lower() in ("1", "true", "yes")
    robots = [f"robot_{i}" for i in range(1, n + 1)]
    poses = _robot_poses(a["robot_poses"], n)

    world = models.world_file(
        world_name=WORLD, rtf=a["rtf"], step=a["step"], arena_half=a["arena_half"])
    engines = {"dart": "", "bullet-featherstone": "--physics-engine gz-physics-bullet-featherstone-plugin"}
    gz_args = f"-r {engines[a['physics_engine']]} {world}"
    if headless:
        gz_args = "-s --headless-rendering " + gz_args
    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
            launch_arguments={"gz_args": gz_args, "on_exit_shutdown": "true"}.items(),
        )
    ]

    for i, (robot, (x, y, yaw)) in enumerate(zip(robots, poses, strict=True), start=1):
        urdf = models.limo(
            robot, i,
            use_mesh=a["use_mesh"],
            max_linear_acceleration=a["max_linear_acceleration"],
            max_angular_acceleration=a["max_angular_acceleration"],
            wheel_mu=a["wheel_mu"],
            wheel_mu_lateral=a["wheel_mu_lateral"],
            body_mu=a["body_mu"],
            lidar_noise_stddev=a["lidar_noise_stddev"],
        )
        actions += [
            _spawn(robot, urdf, x, y, yaw, z=0.002),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                namespace=robot,
                parameters=[{"robot_description": urdf, "use_sim_time": True}],
                output="log",
            ),
        ]

    tee_args = {k[4:]: a[k] for k in ARGS if k.startswith("tee_") and k[4:] not in ("x", "y", "yaw")}
    actions.append(_spawn("tee", models.tee("tee", **tee_args),
                          float(a["tee_x"]), float(a["tee_y"]), float(a["tee_yaw"])))
    goal_args = {k: tee_args[k] for k in ("bar_len", "bar_width", "stem_len", "stem_width")}
    actions.append(_spawn("goal", models.tee("goal", goal=True, **goal_args),
                          float(a["goal_x"]), float(a["goal_y"]), float(a["goal_yaw"])))

    with tempfile.NamedTemporaryFile("w", prefix="pusht_bridge_", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(_bridge_config(robots), f, sort_keys=False)
    actions += [
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="pusht_bridge",
            parameters=[{"config_file": f.name}],
            output="screen",
        ),
        Node(
            package="pusht_gz",
            executable="gt_pose_relay",
            parameters=[{"bodies": robots + ["tee"], "use_sim_time": True}],
            output="screen",
        ),
    ]

    if a["rviz"].lower() in ("1", "true", "yes"):
        actions.append(Node(package="rviz2", executable="rviz2", parameters=[{"use_sim_time": True}]))
    return actions


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=_setup)]
    )
