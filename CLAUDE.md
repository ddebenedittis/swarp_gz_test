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
limo_gz_description   pusht_gz_plugins
            ↑         ↑
             pusht_gz
                ↑
           pusht_policy
```

- `limo_gz_description` (ament_cmake): `urdf/limo.urdf.xacro`, spawned as URDF (sdformat converts it). Frames prefixed `robot_i/`, topics under `/robot_i/`. DiffDrive (4 joints), OdometryPublisher ground truth, JointStatePublisher, `gpu_lidar`.
- `pusht_gz_plugins` (ament_cmake, C++): `PlanarCoulombFriction` gz-sim system (isotropic Coulomb floor friction of a uniform pressure over a footprint, applied as a wrench); an env hook puts its `lib/` on `GZ_SIM_SYSTEM_PLUGIN_PATH`.
- `pusht_gz` (ament_python):
  - `launch/sim.launch.py`: renders world + models with xacro, spawns via `ros_gz_sim create`, writes the `parameter_bridge` YAML to a temp file, one `robot_state_publisher` per robot.
  - `models/tee.sdf.xacro`: T (origin at area centroid on the floor, same construction as `swarp.scenarios.shapes.tee`) and, with `goal:=true`, the visual-only goal marker.
  - `worlds/pusht.sdf.xacro`: world `pusht`, DART, 1 ms step, no walls.
  - `pusht_gz/models.py`: xacro rendering helpers used by the launch file.
  - `pusht_gz/gz_reset.py`: `GzWorld.set_pose / set_poses / set_poses_at_rest / pause` through `/world/pusht/set_pose[_vector]`.
  - `pusht_gz/gt_pose_relay.py`: `/b/gt_odom` → `/b/pose` (PoseStamped, `world`) and `/b/vel` (world-frame TwistStamped).
- `pusht_policy` (ament_python), runs swarp checkpoints (`/opt/swarp` must be on the branch the checkpoint was trained on):
  - `obs.py`: pure-numpy rebuild of swarp's `pusht_lidar` obs (`ObsConfig.from_checkpoint`, `ObsBuilder`, `scan_to_rays`, `label_rays`); `test/test_obs_parity.py` checks it bit-for-bit against `PushTLidarScenario`. Do not change its semantics without rerunning `python3 -m pytest src/pusht_policy/test`.
  - `task.py`: `Task.from_checkpoint` (geometry, tolerances, horizon via the real scenario object) and `sample_episode` (swarp's reset kernels in numpy, overlapping spawns redrawn).
  - `policy_node.py`: one per robot; a step per scan whose stamp is `dt` past the last (not a timer), poses at the nearest stamp (a due scan waits for a lagging pose), `actor.greedy` or, with `sample`, the trainer's tanh-Gaussian draw.
  - `episode_manager.py`: reset, `/goal` + `/episode`, scoring, `episodes.csv`.
  - `launch/eval.launch.py`, `scripts/eval_shards.bash` (staggered shards, restarts a shard that never started; `DOMAIN_BASE`/`BUILD=0` for concurrent runs), `aggregate.py` (Wilson interval, stdlib only), `failure_modes.py` (failure classes and squeeze locks from `rec/`); swarp-side references: `scripts/swarp_reference.py` (host venv). Results: `docs/results.md`.
  - `sysid/`: shared experiment definitions, `gz_runner.py` (`sysid_gz`), `compare.py`; the swarp side is `scripts/sysid_swarp.py` (host venv), which pins swarp's state by overriding `PushTLidarScenario._launch_spawn`.

## Notes

- Lidar self-occlusion: robot `i`'s visuals are on visibility bit `1 << i` and its lidar masks that bit out; bit 31 is for overlays (goal marker, arena outline) that no lidar sees.
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
