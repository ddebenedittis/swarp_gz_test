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
| `tee_floor_friction` | coulomb | `coulomb`: isotropic Coulomb floor friction from `pusht_gz_plugins` (the T on frictionless supports); `contact`: DART's contact friction, which barely turns the T under off-centre pushes |
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

## Evaluate a checkpoint

```bash
# inside the container (checkpoints under ../swarp are at /opt/swarp)
ros2 launch pusht_policy eval.launch.py checkpoint:=/opt/swarp/runs/<run>/<scenario>_final.pt episodes:=50 seed:=0 record:=true
ros2 run pusht_policy aggregate out/<run name>

# on the host: N parallel headless containers (own ROS_DOMAIN_ID, GZ_PARTITION and seed each)
src/pusht_policy/scripts/eval_shards.bash /opt/swarp/runs/<run>/<ckpt>.pt <shards> <episodes per shard> [run name] [base seed] [launch args ...]
```

`eval.launch.py` reads the robot count and the T geometry from the checkpoint, starts the sim, one `policy_node` per robot and the `episode_manager`, and exits after the last episode.
The manager draws initial states like swarp's reset at full difficulty (seeded), teleports the bodies, publishes `/goal` (PoseStamped) and `/episode` (Int32 id, `-1` = stopped), and scores success exactly as swarp (`|p - g| < goal_tolerance` and `|Δθ| < angle_tolerance`, checked every `dt`, timeout `max_steps·dt`).
Each episode is a row of `out/<run name>/episodes.csv`; with `record:=true`, `rec/ep<id>_robot_<i>.npz` holds every step's obs, action, raw scan, poses and T pose.

Results for the Limo-scale checkpoints, with failure modes and how to reproduce them: [docs/results.md](docs/results.md).
Per-run failure classes, stalls and squeeze locks: `PYTHONPATH=src/pusht_policy python3 -m pusht_policy.failure_modes out/<run>`.

## Sim-to-sim system identification

The same scripted `(v, ω)` sequences from the same initial states in swarp and in Gazebo (`pusht_policy/sysid/experiments.py`): robot steps and arcs, pushes, the lidar against the T, and swarp's greedy policy from recorded Gazebo episode starts.

```bash
# Gazebo (container): sim at real time, then the runner
ros2 launch pusht_gz sim.launch.py n_robots:=2 headless:=true rtf:=1 &
ros2 run pusht_policy sysid_gz --ros-args -p use_sim_time:=true -p checkpoint:=/opt/swarp/runs/<run>/<ckpt>.pt -p out:=out/sysid

# swarp (host, swarp's venv; imports swarp read-only)
cd ../swarp && uv run --no-sync python ../swarp_gz_test/src/pusht_policy/scripts/sysid_swarp.py \
    --checkpoint runs/<run>/<ckpt>.pt --out ../swarp_gz_test/out/sysid --paired ../swarp_gz_test/out/<eval run>

# swarp without a checkpoint: a recipe's kwargs, overrides, one group, a tag per config
uv run --no-sync python ../swarp_gz_test/src/pusht_policy/scripts/sysid_swarp.py --recipe pusht_lidar_limo_dr \
    --tag swarp-L3b --group contact --scen-kwarg cmd_gain_w=0.79 --out ../swarp_gz_test/out/sysid/contact
# Gazebo contact forces: add tee_contact_sensor:=true to the sim launch (bridged to /tee/contacts)

# quasi-static references (host, need cvxpy)
python3 src/pusht_policy/scripts/pushing_reference.py --viscous      # pushes -> out/sysid/reference.md
python3 src/pusht_policy/scripts/pressed_turn_reference.py           # turning flush against the T

# tables -> out/sysid/report.md
PYTHONPATH=src/pusht_policy python3 -m pusht_policy.sysid.compare out/sysid
```

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
