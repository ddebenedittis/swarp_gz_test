# CLAUDE.md

## Project Overview

ROS 2 Jazzy + Gazebo Harmonic workspace that tests, in simulation, AgileX Limo robots pushing a T to a goal pose with policies trained in the sibling GPU simulator `../swarp` (`PushTLidarScenario`).
Gazebo stands in for the real robots: the policy sees the same topics a mocap + lidar setup would give.

## Build & Run

Everything runs in Docker; the repo is mounted at the same absolute path, `../swarp` read-only at `/opt/swarp` (on `PYTHONPATH`).

```bash
docker/build.bash                                   # writes docker/.env, builds image swarp_gz_test
docker/run.bash                                     # dev shell, X11 + GPU
docker compose -f docker/docker-compose.yaml run --rm eval [cmd ...]   # headless, rebuilds ws first

# inside the container
colcon build --symlink-install
ros2 launch pusht_gz sim.launch.py n_robots:=2 headless:=true rtf:=0
ros2 run pusht_gz set_pose tee 0 0 0
```

- `docker/entrypoint.sh` sources ROS, builds per `COLCON_BUILD` (`auto` | `always` | `never`), sources `install/`.
- venv `/opt/venv` (CPU torch, warp-lang, numpy<2) is first on `PATH` and its site-packages are on `PYTHONPATH`, because ament_python scripts get a `/usr/bin/python3` shebang.
- gz Python bindings (`gz.transport13`, `gz.msgs10`) come from the ROS gz vendor packages, not from packages.osrfoundation.org.
- Rendering: NVIDIA EGL when `DOCKER_RUNTIME=nvidia` (set by `build.bash`), Mesa on `/dev/dri`, llvmpipe without `/dev/dri`. Never `LIBGL_ALWAYS_SOFTWARE=1` with `/dev/dri` mounted (ogre2 segfault).

## Architecture

```
limo_gz_description
    ↑
pusht_gz
    ↑
pusht_policy   (deployment: obs builder, policy node, episode manager; separate owner)
```

- `limo_gz_description` (ament_cmake): `urdf/limo.urdf.xacro`, spawned as URDF (sdformat converts it). Frames prefixed `robot_i/`, topics under `/robot_i/`. DiffDrive (4 joints), OdometryPublisher ground truth, JointStatePublisher, `gpu_lidar`.
- `pusht_gz` (ament_python):
  - `launch/sim.launch.py`: renders world + models with xacro, spawns via `ros_gz_sim create`, writes the `parameter_bridge` YAML to a temp file, one `robot_state_publisher` per robot.
  - `models/tee.sdf.xacro`: T (origin at area centroid on the floor, same construction as `swarp.scenarios.shapes.tee`) and, with `goal:=true`, the visual-only goal marker.
  - `worlds/pusht.sdf.xacro`: world `pusht`, DART, 1 ms step, no walls.
  - `pusht_gz/models.py`: xacro rendering helpers used by the launch file.
  - `pusht_gz/gz_reset.py`: `GzWorld.set_pose / set_poses / set_poses_at_rest / pause` through `/world/pusht/set_pose[_vector]`.
  - `pusht_gz/gt_pose_relay.py`: `/b/gt_odom` → `/b/pose` (PoseStamped, `world`) and `/b/vel` (world-frame TwistStamped).

## Notes

- Lidar self-occlusion: robot `i`'s visuals are on visibility bit `1 << i` and its lidar masks that bit out; bit 31 is for overlays (goal marker, arena outline) that no lidar sees.
- A `lidar_shell` box visual (0.322×0.220, around the lidar plane) makes other robots appear exactly as swarp's box robots.
- Do not name `<collision>`/`<visual>` in the URDF: sdformat then silently drops the `<gazebo reference>` friction and visibility blocks.
- Wheel friction is anisotropic (`fdir1` = axle, `mu1` lateral 0.2, `mu2` rolling 1.0). With isotropic mu the 4WD skid steer cannot turn in place (scrub torque about the wheelbase beats drive torque about the track). Commanded ω = 1.0 gives 0.79 rad/s.
- gz Python bindings hold the GIL in `request()`: a live Python subscription callback in the same process makes every service call time out. `GzWorld` subscribes to the clock only inside `wait_sim`.
- Teleporting keeps velocities; `set_poses_at_rest` teleports, waits sim time, teleports again.
- OdometryPublisher twist is body-frame; `gt_pose_relay` rotates it to world frame (swarp observes world-frame velocity).
- `rtf:=0` reaches ~1.6× real time with 2 robots on this machine (physics + 2 lidars).
