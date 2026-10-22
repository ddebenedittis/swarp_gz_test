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
limo_bridge   limo_gz_description   pusht_gz_plugins   mocap_msgs
   ↑    ↑               ↑         ↑                      ↑
   │    └──────────  pusht_gz                         limo_real (←limo_bridge)
   │                    ↑                                ↑
   └──────────────  pusht_policy ────────────────────────┘
```

- `limo_bridge` (ament_python, runs ONBOARD on Foxy/py3.8 and Humble: rclpy + stdlib only, no `X | Y` at runtime, no `zip(strict=)`, no `rclpy.try_shutdown`): `robot_bridge` (clamp to 0.4/1.5, 0.3 s watchdog publishing zeros at 20 Hz, latched `/estop`, arena stop on `/robot_i/pose`, `dry_run`, scan/odom relays with prefixed frames + TF), `safety.gate` (pure, `test/test_safety.py`), `spin.py` (spin without rclpy's signal handlers so SIGINT/SIGTERM can still publish a zero; used by `policy_node` too), `launch/robot.launch.py` (limo_base + ydlidar remapped under `/robot_i/limo/`), `launch/amcl.launch.py` (namespaced map_server + amcl + lifecycle manager), `config/amcl.yaml`, `config/ydlidar_tmini_pro.yaml`.
- `limo_real` (ament_python, laptop): `mocap_relay`, `amcl_relay`, `fake_mocap` (gz gt → `/rigid_bodies`), `operator` (interactive console: goal, start/stop, estop, scoring csv), `viz` (RViz markers: robot footprints on their TF frames, goal T, arena; the goal is also an interactive marker whose release goes on `/goal_request` (transformed to `arena` from RViz's fixed frame), which `operator` applies at any time, logging mid-episode changes in `events`; `config/real.rviz`, `laptop.launch.py rviz:=true`), `pose_error` (est vs ref, bias/jitter summary on SIGINT), `measure/*` (cmd timeout, step response, lidar static/rate), `launch/poses.launch.py` (pose_source), `launch/laptop.launch.py`, `config/mocap.yaml`, zenoh configs. Real-robot procedure: `docs/real_robot.md`.
- `mocap_msgs`: copy of mocap4ros2's interfaces (`RigidBodies`).
- `limo_gz_description` (ament_cmake): `urdf/limo.urdf.xacro`, spawned as URDF (sdformat converts it). Frames prefixed `robot_i/`, topics under `/robot_i/`. DiffDrive (4 joints) on `/robot_i/limo/cmd_vel`, OdometryPublisher ground truth, JointStatePublisher, `gpu_lidar` (T-mini Pro: 400 samples, 0.02–12 m, σ 0.015, `lidar_fov` crop). 4.8 kg.
- `pusht_gz_plugins` (ament_cmake, C++): `PlanarCoulombFriction` gz-sim system (isotropic Coulomb floor friction of a uniform pressure over a footprint, applied as a wrench); an env hook puts its `lib/` on `GZ_SIM_SYSTEM_PLUGIN_PATH`.
- `pusht_gz` (ament_python):
  - `launch/sim.launch.py`: renders world + models with xacro, spawns via `ros_gz_sim create`, writes the `parameter_bridge` YAML to a temp file, one `robot_state_publisher` and (`safety_bridge`) one `robot_bridge` per robot; `pose_source` decides who publishes `/robot_i/pose` and whether gt or odometry TF goes on `/tf`.
  - `pusht_gz/landmarks.py`: walls/boxes outside the arena (SDF) and the map rasterized from them (`maps/landmarks.{pgm,yaml}`, `make_map`).
  - `models/tee.sdf.xacro`: T (origin at area centroid on the floor, same construction as `swarp.scenarios.shapes.tee`) and, with `goal:=true`, the visual-only goal marker.
  - `worlds/pusht.sdf.xacro`: world `pusht`, DART, 1 ms step, no walls.
  - `pusht_gz/models.py`: xacro rendering helpers used by the launch file.
  - `pusht_gz/gz_reset.py`: `GzWorld.set_pose / set_poses / set_poses_at_rest / pause` through `/world/pusht/set_pose[_vector]`.
  - `pusht_gz/gt_pose_relay.py`: `/b/gt_odom` → `/b/gt_pose`, `/b/gt_vel` (always) and `/b/pose`, `/b/vel` (bodies in `pose_bodies`).
- `pusht_policy` (ament_python), runs swarp checkpoints (`/opt/swarp` must be on the branch the checkpoint was trained on):
  - `obs.py`: pure-numpy rebuild of swarp's `pusht_lidar` obs (`ObsConfig.from_checkpoint`, `ObsBuilder`, `scan_to_rays`, `label_rays`); `test/test_obs_parity.py` checks it bit-for-bit against `PushTLidarScenario`. Do not change its semantics without rerunning `python3 -m pytest src/pusht_policy/test`.
  - `task.py`: `Task.from_checkpoint` (geometry, tolerances, horizon via the real scenario object) and `sample_episode` (swarp's reset kernels in numpy, overlapping spawns redrawn).
  - `policy_node.py`: one per robot; a step per scan whose stamp is `dt` past the last (not a timer), poses at the nearest stamp (a due scan waits for a lagging pose), `actor.greedy` or, with `sample`, the trainer's tanh-Gaussian draw.
  - `episode_manager.py`: reset, `/goal` + `/episode`, scoring on `gt_pose`, `episodes.csv`; with `initialpose` (AMCL) re-initialises each robot's localizer after the teleport.
  - `video_recorder.py`: with `eval.launch.py record_video:=true`, gz camera frames (gz-transport, not bridged) + overlay -> ffmpeg pipe -> one mp4.
  - `launch/eval.launch.py`, `scripts/eval_shards.bash` (staggered shards, restarts a shard that never started; `DOMAIN_BASE`/`BUILD=0` for concurrent runs), `aggregate.py` (Wilson interval, stdlib only), `failure_modes.py` (failure classes and squeeze locks from `rec/`); swarp-side references: `scripts/swarp_reference.py` (host venv). Results: `docs/results.md`.
  - `sysid/`: shared experiment definitions, `gz_runner.py` (`sysid_gz`), `compare.py`; the swarp side is `scripts/sysid_swarp.py` (host venv), which pins swarp's state by overriding `PushTLidarScenario._launch_spawn`.

## Notes

- Lidar self-occlusion: robot `i`'s visuals are on visibility bit `1 << i` and its lidar masks that bit out; bit 27 is for overlays (goal marker, arena outline) that no lidar sees. Bits 28-31 are reserved by gz-rendering: cameras (default mask 0x0FFFFFFF) never render them.
- A `lidar_shell` box visual (0.322×0.220, around the lidar plane) makes other robots appear exactly as swarp's box robots.
- Do not name `<collision>`/`<visual>` in the URDF: sdformat then silently drops the `<gazebo reference>` friction and visibility blocks.
- Wheel friction is anisotropic (`fdir1` = axle, `mu1` lateral 0.2, `mu2` rolling 1.0). With isotropic mu the 4WD skid steer cannot turn in place (scrub torque about the wheelbase beats drive torque about the track). Commanded ω = 1.0 gives 0.79 rad/s.
- gz Python bindings hold the GIL in `request()`: a live Python subscription callback in the same process makes every service call time out. `GzWorld` subscribes to the clock only inside `wait_sim`.
- DART's (and bullet-featherstone's) contact friction is a pyramid with decoupled axes: a T sliding along one axis keeps the full static friction of the other and resists turning, so an off-centre flat push barely rotates it (-0.4 deg/m vs -64 deg/m quasi-static). The T therefore stands on three frictionless spheres and gets its floor friction from `PlanarCoulombFriction` (`tee_floor_friction:=coulomb`, the default; `contact` restores DART's).
- swarp's T drag is viscous (equal linear/angular decay 40/s, i.e. a uniform viscous floor): it rotates the T ~35% less than Coulomb at a 0.2 m offset push (`scripts/pushing_reference.py --viscous`).
- `scan_to_rays` drops only points more than 3 cm inside the own footprint (`own_margin=-0.03`), so a T touching the side/rear survives 1 cm range noise. A T against the front bumper is 0.058 m from the lidar, below its 0.10 m min range, and stays invisible.
- Two robots driving into opposite faces of the T cannot turn (the corner must swing into a T the other robot holds); Gazebo and the quasi-static reference (`scripts/pressed_turn_reference.py`) agree, swarp turns them anyway (kinematic heading, no contact torque on agents). Single robots flush against a face turn in all three.
- Teleporting keeps velocities; `set_poses_at_rest` teleports, waits sim time, teleports again.
- OdometryPublisher twist is body-frame; `gt_pose_relay` rotates it to world frame (swarp observes world-frame velocity).
- `rtf:=0` reaches ~1.6× real time with 2 robots on this machine (physics + 2 lidars), ~1× with the policies and manager running.
- `/goal` and `/episode` are transient-local (latched); the policy ignores scans stamped before it received the episode start.
- Odometry: the real `limo_base` integrates the IMU yaw and the chassis speed along it; in Gazebo `robot_bridge` (`imu_topic`) re-integrates the DiffDrive odometry the same way from the gz IMU (gyro noise 0.002 rad/s, bias 0.001 rad/s), because wheel odometry overestimates skid-steer turns by ~26%.
- AMCL reads `/robot_i/scan_loc` (`robot_bridge` `loc_half`): returns inside the arena (T, robots: not on the map) removed at the robot's latest `/robot_i/pose`; published unmasked until the first pose, or AMCL never starts.
- `policy_node` republishes its held command on every scan between steps (10 Hz) to feed the 0.3 s watchdog; a step parked waiting for a pose republishes nothing, so a dead pose source stops the robot.
- The policy drops returns closer than the checkpoint's `lidar_min_range` (0.1 m from the mount) even though the T-mini Pro sees from 0.02 m, and blanks rays outside its `lidar_fov` param (`obs.blank_outside_fov`, swarp's rule).
- Testing scripts that SIGINT a background job need `set -m`: non-interactive bash starts `&` jobs with SIGINT ignored. `ros2 run` does not forward a SIGINT sent to its own pid; signal the process group.
- Cross-distro DDS (tested between containers): Humble ↔ Jazzy works; Foxy's Fast DDS aborts (`std::bad_alloc`) on discovering Jazzy, Foxy with `rmw_cyclonedds_cpp` works. Same-host containers need `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` on the Jazzy side. The ros2 daemon is shared over host networking: `ros2 daemon stop` in a container of another distro.
