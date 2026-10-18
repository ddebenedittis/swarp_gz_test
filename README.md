# swarp_gz_test

Gazebo Harmonic + ROS 2 Jazzy testbed for AgileX Limo robots pushing a T to a goal pose (Push-T), with policies trained in [swarp](../swarp).
Everything runs in Docker.

## Build

```bash
docker/build.bash        # writes docker/.env (paths, UID/GID, nvidia runtime if present), builds the image
```

The image is `osrf/ros:jazzy-desktop` plus `ros-jazzy-ros-gz` (Gazebo Harmonic), and a `/opt/venv` venv (`--system-site-packages`) with CPU torch, `warp-lang`, and pytest.
`../swarp` is bind-mounted read-only at `/opt/swarp` and is on `PYTHONPATH`, so `import swarp` works without a rebuild.

## Run

```bash
docker/run.bash          # interactive shell with X11 (Gazebo GUI, RViz), GPU if available
ros2 launch pusht_gz sim.launch.py n_robots:=2

# headless, one-shot (rebuilds the workspace on start, command overridable)
docker compose -f docker/docker-compose.yaml run --rm eval
docker compose -f docker/docker-compose.yaml run --rm eval ros2 launch pusht_gz sim.launch.py headless:=true rtf:=0
```

The entrypoint runs `colcon build --symlink-install` when `install/` is missing (`COLCON_BUILD=auto`, the `dev` default) or on every start (`always`, the `eval` default).
For `docker compose run dev`, create `/tmp/.docker.xauth` first (`docker/run.bash` does it).

Rendering for the `gpu_lidar` works on NVIDIA through EGL (`--headless-rendering`), on Intel/AMD through Mesa with `/dev/dri`, and in pure software (llvmpipe) when `/dev/dri` is not mounted.
Do not set `LIBGL_ALWAYS_SOFTWARE=1` with `/dev/dri` mounted: ogre2 then segfaults.

## Launch arguments (`pusht_gz sim.launch.py`)

| Argument | Default | Meaning |
|---|---|---|
| `n_robots` | 2 | Limos `robot_1..robot_N` (max 30) |
| `headless` | false | server only, headless EGL rendering |
| `rtf` | 1.0 | target real-time factor, `0` = as fast as possible |
| `step` | 0.001 | physics step [s] |
| `robot_poses` | circle r=1.4 | `'x,y,yaw;x,y,yaw'` |
| `max_linear_acceleration`, `max_angular_acceleration` | 1.0, 3.0 | diff-drive limits [m/s², rad/s²] |
| `wheel_mu`, `wheel_mu_lateral` | 1.0, 0.2 | wheel friction along / across the rolling direction |
| `lidar_noise_stddev` | 0.01 | lidar range noise [m] |
| `tee_bar_len`, `tee_bar_width`, `tee_stem_len`, `tee_stem_width`, `tee_height` | 1.0, 0.3, 0.7, 0.3, 0.25 | T geometry [m] |
| `tee_mass`, `tee_mu` | 0.5, 0.3 | T mass [kg] and friction |
| `tee_x`, `tee_y`, `tee_yaw` | 0, 0, 0 | T initial pose |
| `goal_x`, `goal_y`, `goal_yaw` | 0.8, 0.8, 0.5 | goal marker pose |
| `use_mesh` | true | Limo meshes (false: boxes) |
| `arena_half` | 2.0 | half side of the drawn arena outline (no walls) |
| `rviz` | false | start RViz |

## Topics

| Topic | Type | Notes |
|---|---|---|
| `/robot_i/cmd_vel` | `geometry_msgs/Twist` | diff drive (4WD skid steer), acceleration-limited |
| `/robot_i/scan` | `sensor_msgs/LaserScan` | 360 rays, -π..π step 1°, 0.10–8 m, 10 Hz, frame `robot_i/laser_link` |
| `/robot_i/odom` | `nav_msgs/Odometry` | wheel odometry, `robot_i/odom` → `robot_i/base_footprint` |
| `/robot_i/gt_odom`, `/tee/gt_odom` | `nav_msgs/Odometry` | exact ground truth, world frame, body-frame twist, 50 Hz |
| `/robot_i/pose`, `/tee/pose` | `geometry_msgs/PoseStamped` | ground truth, frame `world` (mocap on the real robots) |
| `/robot_i/vel`, `/tee/vel` | `geometry_msgs/TwistStamped` | world-frame linear velocity, yaw rate |
| `/robot_i/joint_states` | `sensor_msgs/JointState` | wheel joints |
| `/tf` | | `world` → `robot_i/base_footprint`, `world` → `tee` (ground truth) |
| `/clock` | `rosgraph_msgs/Clock` | sim time |
| `/world/pusht/set_pose` | `ros_gz_interfaces/srv/SetEntityPose` | teleport a model (service) |

## Reset

```python
from pusht_gz.gz_reset import GzWorld
world = GzWorld("pusht")
world.set_pose("tee", 0.3, -0.2, 1.0)                                    # one model
world.set_poses({"robot_1": (-1.0, 0.0, 0.0), "goal": (0.8, 0.8, 0.5)})  # one step, many models
world.set_poses_at_rest({"tee": (0.0, 0.0, 0.0)})                        # teleport, settle, teleport
```

From a shell: `ros2 run pusht_gz set_pose tee 0 0 0`.
Teleporting keeps a body's velocity; `set_poses_at_rest` removes it.
