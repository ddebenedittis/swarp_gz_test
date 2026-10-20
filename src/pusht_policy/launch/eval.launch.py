"""Evaluate a swarp pusht_lidar checkpoint in Gazebo: sim + one policy node per robot + the
episode manager. Exits when the manager has run all episodes.

    ros2 launch pusht_policy eval.launch.py checkpoint:=/path/ckpt.pt episodes:=50 seed:=0

The robot count and the T geometry come from the checkpoint. Results:
``<out_dir>/<run_name>/episodes.csv`` (and ``rec/`` per-step npz files with record:=true).
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pusht_policy.task import Task

ARGS = {
    "checkpoint": ("", "swarp checkpoint (*.pt) of a pusht_lidar recipe"),
    "episodes": ("10", "number of episodes"),
    "seed": ("0", "episode sampling seed"),
    "out_dir": (os.path.join(os.environ.get("WS_DIR", "."), "out"), "results root"),
    "run_name": ("", "results subdirectory; default <checkpoint stem>_s<seed>"),
    "record": ("false", "write per-step obs/actions/scans to <run>/rec/*.npz"),
    "sample": ("false", "draw actions from the policy distribution instead of greedy"),
    "headless": ("true", "no Gazebo GUI"),
    "rtf": ("0", "real-time factor; 0 runs as fast as possible"),
    "record_video": ("false", "film the episodes with the world's video camera into one mp4"),
    "video": ("", "mp4 path; default <run>/video.mp4"),
    "video_title": ("", "text in the video's top-right corner"),
    "pose_source": ("gt", "gt | amcl | mocap: what /robot_i/pose is (limo_real poses.launch.py)"),
    "landmarks": ("auto", "walls and boxes outside the arena; auto: with pose_source:=amcl"),
    "lidar_fov": ("6.283185307179586", "lidar field of view [rad]: gz sensor crop and the policy's blanking"),
    "lidar_noise_stddev": ("0.015", "lidar range noise [m]"),
    "safety_bridge": ("true", "commands through limo_bridge robot_bridge, as on the robots"),
    "amcl_params": ("", "AMCL params yaml (pose_source:=amcl); default limo_bridge/config/amcl.yaml"),
    "amcl_scan": ("scan_loc", "AMCL's scan topic under /robot_i/"),
    "pose_error": ("auto", "log /robot_i/pose against ground truth into <run>/pose_error; auto: unless gt"),
    # Physical parameters forwarded to pusht_gz sim.launch.py (sim-to-sim knobs).
    "max_linear_acceleration": ("1.0", "diff-drive linear acceleration limit [m/s^2]"),
    "max_angular_acceleration": ("3.0", "diff-drive angular acceleration limit [rad/s^2]"),
    "wheel_mu_lateral": ("0.2", "wheel friction along the axle"),
    "tee_mass": ("0.5", "T mass [kg]"),
    "tee_mu": ("0.3", "T friction coefficient"),
}


def _setup(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ARGS}
    ckpt = os.path.abspath(a["checkpoint"])
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint:={a['checkpoint']!r} is not a file")
    task = Task.from_checkpoint(ckpt)
    run_name = a["run_name"] or f"{os.path.splitext(os.path.basename(ckpt))[0]}_s{a['seed']}"
    run_dir = os.path.join(a["out_dir"], run_name)
    record = a["record"].lower() in ("1", "true", "yes")
    src = a["pose_source"]
    landmarks = src == "amcl" if a["landmarks"] == "auto" else a["landmarks"].lower() in ("1", "true", "yes")
    pose_error = src != "gt" if a["pose_error"] == "auto" else a["pose_error"].lower() in ("1", "true", "yes")
    sample = a["sample"].lower() in ("1", "true", "yes")

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("pusht_gz"), "launch", "sim.launch.py")
        ),
        launch_arguments={
            "n_robots": str(task.n_agents),
            "headless": a["headless"],
            "rtf": a["rtf"],
            "arena_half": str(task.world_size),
            "tee_bar_len": str(task.bar_len),
            "tee_bar_width": str(task.bar_width),
            "tee_stem_len": str(task.stem_len),
            "tee_stem_width": str(task.stem_width),
            "record_video": a["record_video"],
            "pose_source": a["pose_source"],
            "landmarks": str(landmarks).lower(),
            "lidar_fov": a["lidar_fov"],
            "lidar_noise_stddev": a["lidar_noise_stddev"],
            "safety_bridge": a["safety_bridge"],
            **{k: a[k] for k in ("max_linear_acceleration", "max_angular_acceleration",
                                 "wheel_mu_lateral", "tee_mass", "tee_mu")},
        }.items(),
    )
    policies = [
        Node(
            package="pusht_policy",
            executable="policy_node",
            name=f"policy_robot_{i}",
            parameters=[{
                "checkpoint": ckpt,
                "robot_index": i,
                "n_robots": task.n_agents,
                "record_dir": os.path.join(run_dir, "rec") if record else "",
                "sample": sample,
                # Distinct per robot and per shard seed.
                "sample_seed": 1000 * int(a["seed"]) + i,
                "lidar_fov": float(a["lidar_fov"]),
                "use_sim_time": True,
            }],
            output="screen",
        )
        for i in range(1, task.n_agents + 1)
    ]
    manager = Node(
        package="pusht_policy",
        executable="episode_manager",
        parameters=[{
            "checkpoint": ckpt,
            "episodes": int(a["episodes"]),
            "seed": int(a["seed"]),
            "out_dir": a["out_dir"],
            "run_name": run_name,
            "initialpose": src == "amcl",
            "use_sim_time": True,
        }],
        output="screen",
    )
    done = RegisterEventHandler(
        OnProcessExit(target_action=manager, on_exit=[EmitEvent(event=Shutdown(reason="episodes done"))])
    )
    robots = [f"robot_{i}" for i in range(1, task.n_agents + 1)]
    extra = []
    if src in ("amcl", "mocap"):
        extra.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("limo_real"), "launch", "poses.launch.py")
            ),
            launch_arguments={
                "pose_source": src,
                "n_robots": str(task.n_agents),
                "use_sim_time": "true",
                "map": os.path.join(get_package_share_directory("pusht_gz"), "maps", "landmarks.yaml"),
                "amcl_params": a["amcl_params"],
                "amcl_scan": a["amcl_scan"],
            }.items(),
        ))
    if src == "mocap":
        extra.append(Node(
            package="limo_real", executable="fake_mocap",
            parameters=[{"bodies": robots + ["tee"],
                         "names": [f"limo_{i}" for i in range(1, task.n_agents + 1)] + ["tee"],
                         "seed": int(a["seed"]), "use_sim_time": True}],
            output="screen",
        ))
    if pose_error:
        extra.append(Node(
            package="limo_real", executable="pose_error",
            parameters=[{"robots": robots, "est": "pose", "ref": "gt_pose",
                         "out": os.path.join(run_dir, "pose_error"), "use_sim_time": True}],
            output="screen",
        ))
    if a["record_video"].lower() in ("1", "true", "yes"):
        os.makedirs(run_dir, exist_ok=True)
        extra.append(Node(
            package="pusht_policy",
            executable="video_recorder",
            parameters=[{
                "output": a["video"] or os.path.join(run_dir, "video.mp4"),
                "title": a["video_title"],
                "goal_tolerance": task.goal_tolerance,
                "angle_tolerance": task.angle_tolerance,
                "use_sim_time": True,
            }],
            output="screen",
            sigterm_timeout="30",
        ))
    return [sim, *policies, manager, done, *extra]


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [OpaqueFunction(function=_setup)]
    )
