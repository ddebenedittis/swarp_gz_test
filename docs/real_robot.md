# Real Limo bringup

How to run the swarp Push-T policies on the real Limos, step by step: measure the robots first, then run the policy in shadow mode, then single pushes at a reduced speed cap, then full episodes.
Do not skip ahead: every stage checks an assumption the next one relies on.

## The setup

```
 robot_i (Foxy or Humble)                                   laptop (this repo's Jazzy container)
 ─────────────────────────                                  ───────────────────────────────────────
 limo_base ──/robot_i/limo/odom──┐                          mocap4ros2 ─/rigid_bodies─► mocap_relay ─┐
 ydlidar   ──/robot_i/limo/scan──┤                                                   or amcl_relay ──┤
                                 ▼                                                                   ▼
                       robot_bridge ── /robot_i/scan, /robot_i/odom, TF ─────────────► policy_node (one per robot)
 limo_base ◄─/robot_i/limo/cmd_vel── robot_bridge ◄──────────── /robot_i/cmd_vel ───────────┘
                       ▲      ▲                                     /robot_i/pose, /robot_i/vel, /tee/pose
                       │      └── /robot_i/pose (arena stop)        /goal, /episode ◄── operator (console)
                       └───────── /estop (latched) ◄─────────────────────────────────────── operator
```

- `limo_bridge` (onboard, rclpy and the standard library only, Python 3.8+): `robot.launch.py` starts AgileX's `limo_base` and the T-mini Pro driver with their absolute topics remapped under `/robot_i/limo/`, and `robot_bridge` between them and the network.
- `robot_bridge` also publishes `/robot_i/scan_loc` for AMCL: the scan with every return inside the square `|x|, |y| ≤ loc_half` (2.35 m, at the robot's latest `/robot_i/pose`) removed, since the arena holds only the T and the robots, which are not on the map.
- `robot_bridge` clamps the policy's command to the trained `|v| ≤ 0.4 m/s`, `|ω| ≤ 1.5 rad/s`, and publishes zeros to the chassis at 20 Hz whenever no command arrived for 0.3 s, `/estop` is true, or (arena stop on) its `/robot_i/pose` is older than 0.5 s; past the arena edge plus 0.25 m it zeroes outward motion.
- `dry_run:=true` (shadow mode) never sends anything to the chassis; the gated command still goes to `/robot_i/cmd_vel_gated`.
- `limo_real` (laptop): the pose sources (`mocap_relay`, `amcl_relay`), `laptop.launch.py` (pose source and policies), the `operator` console (goal, start/stop, e-stop, scoring), `pose_error`, and the measurement scripts.
- Gazebo runs the same `robot_bridge` between the policies and the DiffDrive (`sim.launch.py safety_bridge:=true`, the default), so the sim stops the robots the way the real stack does.

### What stops the robots

- Ctrl-C on `laptop.launch.py`: each `policy_node` publishes a zero twist before exiting (rclpy runs without its own signal handler, so the context is still alive; verified in Gazebo with the bridge off).
- The laptop crashes, the launch is killed with SIGKILL, or the WiFi drops: the bridge's watchdog zeroes the chassis 0.3 s after the last command (verified in Gazebo with SIGKILL, and on Foxy and Humble against a fake driver: first zero 0.26–0.31 s after the last command, then 20 Hz of zeros).
- The pose source dies: a policy step waits for a pose and republishes nothing, so the watchdog stops the robot; the bridge's own pose timeout (0.5 s) stops it too.
- `estop` in the operator console, or from any shell: `ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable /estop std_msgs/msg/Bool "{data: true}"`.
  It is latched, so a bridge that restarts still sees it; `release` (or `data: false`) releases it.
- Ctrl-C on `robot.launch.py`: the bridge sends zeros as it exits.
- `limo_base` itself has no command timeout and forwards every Twist unclamped; whether the firmware stops on its own is what `measure_cmd_timeout` finds out.
- Killing the bridge with SIGKILL leaves the chassis on its last command if the firmware has no timeout: keep the remote's stop button at hand.

### Conventions

- The arena frame is the frame the policy was trained in: origin at the arena centre, the arena spanning `[-2, 2]²` m, yaw counter-clockwise from +x.
  Every `/robot_i/pose`, `/tee/pose` and `/goal` is in it.
- AMCL's `map` frame is the arena frame: build the map with the robot at the arena origin facing +x, or set `map_to_arena` (the map origin's `x, y, yaw` in the arena frame) on `amcl_relay`.
- Mocap: `mocap_to_arena` in `limo_real/config/mocap.yaml` is the mocap origin's pose in the arena frame; `yaw_offsets` / `offsets_xy` put each rigid body's frame on the robot's centre and heading (and on the T's area centroid, x along the T's axis as in `pusht_gz/models/tee.sdf.xacro`).
- `robot_i` is the policy's agent `i - 1`: the robot named `robot_1` runs agent 0 of the checkpoint.
- Topics: `/robot_i/scan` (frame `robot_i/laser_link`, 0 rad = the robot's heading after `scan_yaw_offset`), `/robot_i/odom` (`robot_i/odom` → `robot_i/base_footprint`), `/robot_i/cmd_vel`, `/robot_i/cmd_vel_gated`, `/robot_i/bridge_state`, `/robot_i/pose`, `/robot_i/vel`, `/tee/pose`, `/goal`, `/episode`, `/estop`.

## Networking

- One `ROS_DOMAIN_ID` per fleet, the same on every robot and in the laptop container (`ROS_DOMAIN_ID=<n> docker compose ... run ...`; `docker/docker-compose.yaml` passes it through).
  Use a number no one else on the network uses.
- A dedicated WiFi router for the robots and the laptop, with multicast allowed (DDS discovery uses it).
- Synchronise the clocks: the policy pairs each scan (robot clock) with the pose nearest its stamp (laptop or mocap clock) within 50 ms.
  Run chrony on the laptop as a server (`allow <subnet>`, `local stratum 8`) and on each robot as its client (`server <laptop-ip> iburst`), then check with `measure_lidar_rate`: `receive_minus_stamp_s.mean` should be a few ms and positive.
- The robots' distros differ from the laptop's Jazzy.
  Tested in containers on one host (the laptop's Jazzy Gazebo eval with each robot's `robot_bridge` in a Foxy or Humble container, 2 episodes each, both solved):
  - Humble (Fast DDS, default) ↔ Jazzy (Fast DDS): works.
  - Foxy (Fast DDS 2.1, the Limo Pro's default) ↔ Jazzy: the Foxy process aborts with `std::bad_alloc` during discovery.
  - Foxy with CycloneDDS ↔ Jazzy (Fast DDS): works.
    On each Limo Pro: `sudo apt install ros-foxy-rmw-cyclonedds-cpp` and `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` in `~/.bashrc` before launching anything (the onboard driver nodes then use Cyclone too).
  - Containers on the same host need `FASTDDS_BUILTIN_TRANSPORTS=UDPv4` on the Jazzy side (Fast DDS shared memory between versions drops everything); between machines this does not matter.
- If DDS across distros or across the WiFi still misbehaves, use `zenoh-bridge-ros2dds`: `limo_real/config/zenoh-robot.json5` on each robot (onboard ROS with `ROS_LOCALHOST_ONLY=1`), `zenoh-laptop.json5` on the laptop.
  These configs are untested.

## Install

On each robot (Foxy or Humble), in a workspace next to AgileX's (which provides `limo_base`, `ydlidar_ros2_driver` and, for onboard AMCL, nav2):

```bash
cd ~/ws/src && cp -r <this repo>/src/limo_bridge . && cd ~/ws && colcon build --packages-select limo_bridge
source install/setup.bash
ros2 launch limo_bridge robot.launch.py robot_name:=robot_1 dry_run:=true
```

- Stop AgileX's own bringup first (`limo_start.launch.py` or its systemd service): two `limo_base` instances fight over the serial port.
- With AgileX's bringup kept running instead, pass `driver:=false lidar:=false`: the bridge then reads `/scan`, `/odom` and writes `/cmd_vel`, which every robot in the domain shares, so do that with one robot only.
- The T-mini Pro parameters are `limo_bridge/config/ydlidar_tmini_pro.yaml` (full ±180°, 10 Hz; AgileX's crop to ±90° is undone); the serial ports default to `/dev/ttylimo` and `/dev/ydlidar`, AgileX's udev names.

On the laptop: `docker/build.bash`, then `docker/run.bash` and `colcon build --symlink-install`; the Qualisys driver (mocap4ros2) runs wherever the lab runs it and publishes `/rigid_bodies` in the same domain.

## Checklist

Every command below except the robot ones runs in the laptop container; `<ck>` is the checkpoint, e.g. `/opt/swarp/runs/remote-davide-cf/L5s-wheelstall-limo-s0-fromL5r/pusht_lidar_limo_dr_wheel_best.pt`.

### 0. Before anything moves

1. Charge the batteries, clear the arena, mark its corners and the origin on the floor, and put each Limo in diff-drive mode (no Mecanum wheels, the policy assumes a skid steer).
2. Start the robot side in shadow mode on each robot: `ros2 launch limo_bridge robot.launch.py robot_name:=robot_i dry_run:=true`.
3. On the laptop, check that `ros2 topic hz /robot_i/scan` reads ~10 Hz and `ros2 topic echo /robot_i/bridge_state` reads `dry_run:stale` for every robot.
4. Calibrate the mocap: define rigid bodies `limo_1`, `limo_2`, `tee` (and optionally `goal_marker`), measure `mocap_to_arena`, `yaw_offsets` and `offsets_xy`, and write them into a copy of `limo_real/config/mocap.yaml`.
5. Run `ros2 run limo_real mocap_relay --ros-args --params-file <your mocap.yaml>` and check, with each robot placed on a floor mark facing +x, that `/robot_i/pose` reads the mark's coordinates and a yaw of 0.

### 1. Measurements

Each script writes JSON (and raw data) under `out/measure`.
Keep the numbers: they go to the recipe author (FOV, pose noise, latency) and into the launch parameters (`lidar_fov`, `scan_yaw_offset`).

1. **Firmware command timeout** (robot on blocks, bridge in `dry_run`): `ros2 run limo_real measure_cmd_timeout --ros-args -p robot:=robot_i`.
   It drives 0.1 m/s for 1 s straight to the driver, goes silent and reports how long the wheels keep turning.
   "No firmware timeout" means `robot_bridge` is the only watchdog: never run the chassis without it.
2. **Static lidar scan** (robot in open space, one narrow object 1 m straight ahead): `ros2 run limo_real measure_lidar_static --ros-args -p robot:=robot_i`.
   It reports the self-return and blind sectors, the unobstructed field of view `fov_rad`, and the object's bearing `probe_bearing_deg`.
   If the bearing is not ~0, restart the robot with `scan_yaw_offset:=<minus the bearing, in rad>` and repeat.
   Use `fov_rad` (the smallest over the robots) as `lidar_fov` everywhere below and send it to the recipe author; `policy_node` refuses a `lidar_fov` below the checkpoint's narrowest trained one (240° for L6) unless `allow_fov_below_trained`.
   This FOV is measured from the mount, while swarp's `lidar_fov` (through L6) blanks the robot-centre rays: a physical 240° hides a T against the robot's side that a 240° policy expects to see, which in Gazebo takes L6 from 11/20 (full scan, policy blanking at 240°) to 1/20 (lidar cropped to 240°; `docs/results.md`); until the recipe blanks by the bearing from the mount, a robot needs about 290° at the mount to serve a 240° policy.
   Repeat with `-p checkpoint:=<ck>`: `policy_rays_hit` lists the policy's binned rays that read a hit in open space; it must be empty, or the robot sees its own wheels or mount as obstacles (the checkpoint's minimum range is 0.02 m for L6, so returns within 2 cm of the lidar are all that is dropped besides the footprint).
3. **Lidar rate and latency**: `ros2 run limo_real measure_lidar_rate --ros-args -p robot:=robot_i`.
   Expect ~10 Hz, few gaps, and `receive_minus_stamp_s.mean` of a few ms; a large or negative value means the clocks are not synchronised (fix chrony before going on).
4. **Step response** (bridge running normally, ~1.5 m free ahead, mocap on): restart the robot side without `dry_run`, then `ros2 run limo_real measure_step_response --ros-args -p robot:=robot_i`.
   It runs v steps of ±0.1, ±0.2, ±0.4 m/s and ω steps of ±0.5, ±1.0, ±1.5 rad/s and reports per step the gain (measured / commanded), dead time, rise time and acceleration, from mocap and from odometry.
   Compare with the training ranges (`cmd_gain_v` 0.9–1.05, `cmd_gain_w` 0.7–1.0, `servo_accel` 0.6–2.0 m/s², `servo_ang_accel` 2–6 rad/s²); the turn gain is the turn ratio (Gazebo's skid steer gives 0.79).
   The odometry's heading comes from the chassis IMU (`limo_base` integrates the IMU yaw and the chassis' forward speed along it), so its pose turns correctly even though its twist's ω (wheel-based) overestimates turns; check that the odometry's ω gain from the pose differs from the mocap one by only the IMU drift.
5. **Mocap vs AMCL** (only for `pose_source:=amcl`): build a map (AgileX's cartographer, or slam_toolbox) with the robot starting at the arena origin facing +x, save it, then start the robots with `amcl:=true map:=<map yaml> x:=.. y:=.. yaw:=..` and on the laptop `ros2 launch limo_real laptop.launch.py checkpoint:=<ck> pose_source:=mocap compare:=true lidar_fov:=<fov>` with the robots' bridges in `dry_run`.
   Drive the robots around by hand (the remote, with the bridge stopped) or let the policy run in shadow mode for a few minutes, then Ctrl-C: `out/real/amcl_vs_mocap/summary.json` holds the mean / p95 position and heading error and the per-episode bias, the same statistics as the Gazebo numbers below.

### 2. Shadow mode

1. Robots: `ros2 launch limo_bridge robot.launch.py robot_name:=robot_i dry_run:=true`.
2. Laptop: `ros2 launch limo_real laptop.launch.py checkpoint:=<ck> pose_source:=mocap lidar_fov:=<fov> record:=true`.
3. Second laptop terminal: `ros2 run limo_real operator --ros-args -p checkpoint:=<ck> -p out:=out/real/shadow`, then `goal 0.8 0.8 30` and `start`.
4. Move the T and the robots by hand and watch `/robot_i/cmd_vel_gated`: the commands should point the robots at the T and change sensibly when you move it.
5. Check the policy logs: no repeated "no pose within 0.05 s" warnings (pose latency or clock offset), one step every 0.2 s.
6. Inspect a record (`out/real/rec/ep0000_robot_1.npz`): the lidar part of `obs` should show the T where it is, and the ray classes T (3) / robot (1) / none (0) should be right.

### 3. Single pushes at a reduced speed cap

1. Restart one robot without `dry_run` and with `max_v:=0.2 max_w:=0.8`; keep the other robot in `dry_run` and out of the arena (its policy still needs its pose: leave it tracked by mocap, parked outside the arena).
2. Put the T in the middle, set a goal 0.5 m away with the same heading, `start`, and keep a hand on `estop`.
3. Check that the robot drives to the T, pushes it without climbing it, and that `stop` (or a solved episode) stops it at once.
4. Test the stops for real once: Ctrl-C on `laptop.launch.py` mid-push, unplug the laptop's WiFi mid-push, and `estop` mid-push; the robot should stop within 0.3 s each time.
5. Test the arena stop: start the robot near an edge with a goal that drags the T out, and check that it stops at the edge plus 0.25 m with `bridge_state` reading `arena`.

### 4. Full episodes

1. Both robots without `dry_run`, first at `max_v:=0.3`, then at the trained 0.4 / 1.5.
2. Place the T and the robots as the Gazebo episode manager would (robots 1–1.5 m from the T, facing it), set a goal, `start`.
3. The operator scores every episode as swarp does (0.15 m, 0.25 rad, 450 steps = 90 s) and appends it to `out/real/<run>/episodes.csv` with the initial poses and every gate event (`estop`, `arena`, `stale`).
4. Record the per-step data (`record:=true`) and a `ros2 bag record /tf /robot_1/scan /robot_2/scan /robot_1/pose /robot_2/pose /tee/pose /robot_1/cmd_vel_gated /robot_2/cmd_vel_gated /goal /episode` per session.

## Gazebo stand-ins

- `eval.launch.py pose_source:=gt|amcl|mocap` swaps the pose source behind `/robot_i/pose` (`amcl`: walls and boxes outside the arena from `pusht_gz/landmarks.py`, the map generated from the same geometry in `pusht_gz/maps/`, one AMCL per robot, re-initialised at the true pose after every teleport; `mocap`: `fake_mocap` turns ground truth into `/rigid_bodies` for the real `mocap_relay`).
- `pose_error` logs `/robot_i/pose` against ground truth into `<run>/pose_error/`.
- In Gazebo `mocap_relay` (fed by `fake_mocap` with 1 mm / 3 mrad noise) is within 1.3 mm of the truth in position and 0.02 m/s in velocity, and the operator console ran a solved episode through it.
- AMCL during pushing, L6 (`L6-real-limo-s0-fromL5s` best, greedy, the gz lidar and the policy at a 240° field of view), 20 episodes × 2 robots, ~165k samples each; "raw" is AMCL on the full scan, "tuned" is the shipped setup (`limo_bridge/config/amcl.yaml` on `/robot_i/scan_loc`, the odometry's heading from the IMU):

  | | raw: mean / median / p95 / max | tuned: mean / median / p95 / max |
  |---|---|---|
  | position error [m] | 0.257 / 0.098 / 1.298 / 7.21 | 0.162 / 0.091 / 0.579 / 2.79 |
  | heading error [rad] | 0.067 / 0.015 / 0.106 / 3.14 | 0.022 / 0.013 / 0.060 / 0.57 |
  | per-episode bias, position (mean / std / max) [m] | 0.154 / 0.331 / 1.92 | 0.101 / 0.206 / 1.14 |
  | per-episode bias, heading (mean abs / std / max abs) [rad] | 0.049 / 0.225 / 1.44 | 0.010 / 0.016 / 0.078 |
  | jitter about the bias (position per axis, heading) | 0.373 m, 0.161 rad | 0.170 m, 0.038 rad |
  | decorrelation time (position, heading) | 3.7 s, 1.8 s | 3.6 s, 1.6 s |
  | excursions > 0.3 m: per robot-minute, median / p90 duration, median / p90 peak | 2.2, 0.3 / 8.1 s, 0.34 / 1.25 m | 1.2, 0.5 / 5.8 s, 0.35 / 0.89 m |
  | free driving (79 % of the time): position median / p95 | | 0.083 / 0.187 m |
  | ≤ 5 s after a shove (slip > 0.15 m/s; 21 %): position median / p95 | | 0.146 / 1.46 m |

  89 % of the excursions above 0.3 m start within 5 s of a shove (the T or the teammate moving the robot faster than its wheels report).
  Even a robot driving a clean circle with no contact keeps a 5–12 cm offset with every AMCL setting tried (the scans match the walls at the true pose to 2.5 mm, so this is AMCL's own estimate); the 8–9 cm median above is that floor.
  Tried without gain: `update_min_d` / `update_min_a` 0 (nav2 AMCL still updates only when the odometry moves), a sharper likelihood field, the beam model, beam skipping, 5–10k particles.
- Success with those poses is not informative for L6: in Gazebo it solves 1/20 with ground-truth poses at 240° (`out/L6_gt_fov240`), 2/20 with tuned AMCL (`out/L6_amcl_fov240`) and 2/20 with raw AMCL; at the full field of view it solved 2/4 with ground truth where L5s solved 4/4.
- L5s before this tuning (wheel odometry, full scan, full FOV): 19/20 with ground truth and 15/20 with AMCL on the same 20 states (`out/L5s_gt_landmarks`, `out/L5s_amcl`; position error median 0.081 m, p95 0.469 m).

## What only the hardware can tell

- Whether the firmware has a command timeout, the real turn ratio and accelerations, the lidar's true FOV, mount yaw and self-returns, its timing, and the real AMCL error (the measurement scripts above).
- Whether Foxy with CycloneDDS talks to the laptop over the lab's WiFi, and whether multicast discovery works there (tested only between containers on one host).
- The zenoh fallback configs.
- `robot.launch.py` with the real `limo_base` and `ydlidar_ros2_driver` (their parameters and remappings follow their source; neither is in the test containers).
