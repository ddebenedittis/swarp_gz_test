---
name: real-limo-experiment
description: Start a Push-T experiment on the two real AgileX Limos (lm2591 = robot_1, lm2935 = robot_2) with a swarp policy - robot bringup, automatic initial localization on the saved map (scan matching, AMCL seeding, 360 deg spins, verification), T detection, arena and goal, RViz, shadow run, live episode, emergency stop. Optionally maps the room autonomously first (only when the user asks for mapping). Use whenever the user wants to run, restart or prepare an episode/experiment on the real robots.
---

# Real two-Limo Push-T experiment

Everything is driven from the laptop host with the scripts in `scripts/real/` (lab constants in `scripts/real/config.bash`).
ROS on the laptop runs in the Humble container `swarp_real`; `laptop.bash py ...` runs a script inside it.
Read `docs/real_robot.md` for the background; this skill is the operational procedure that worked on 2026-10-22.

## Safety rules (always)

- Before anything moves, tell the user the stop command and keep it ready: `scripts/real_stop.bash` (both robots' bridges send zeros and exit, latched `/estop`, laptop policies and scripts killed).
  Single robot: `ssh agilex@lm2591.local ~/swarp_ws/stop.sh` (or lm2935).
- Every motion goes through `robot_bridge` (clamp, 0.3 s watchdog, arena stop, `/estop`); never publish to `/robot_i/limo/cmd_vel`.
- Ask the user before the first motion of the session (the spins) and before `start` of every live episode; state the goal and the caps.
- If the user says stop: run `scripts/real_stop.bash` unless they say otherwise, then report the state.
- Monitor every live episode (`monitor.py`); a robot `PINNED` at the arena stop or gates flipping `arena` for >10 s: `laptop.bash op stop`, then ask.
- Never run `pkill -f` on a robot with a pattern that appears in your own ssh command line (it kills the ssh session); call `~/swarp_ws/stop.sh` by path.

## 0. Preflight

1. `scripts/real/robots.bash status`: both reachable, clock offsets within ~20 ms (the policy pairs poses within 50 ms).
   A large offset: ask the user to run `sudo systemctl restart systemd-timesyncd` on that robot.
   A robot with a 0-minute uptime was rebooted: its pose is unknown, redo localization (step 2).
2. Container: `docker ps | grep swarp_real`; if missing, `scripts/real/laptop.bash container` (image `swarp_real_humble`, build with `docker build -f docker/Dockerfile.humble --build-arg USERNAME=$USER -t swarp_real_humble docker`).
3. First time on a robot, or after changing `src/limo_bridge`: `scripts/real/robots.bash install`.
4. The map: `out/real/maps/<name>.yaml` on the laptop (default `map_corridor`, `MAP=` overrides). AMCL runs on the laptop, so the robots need no copy.
   If the user asks for mapping, do step 1 first; otherwise skip it.

## 1. Autonomous mapping (only when the user asks)

UNTESTED on the robots so far: say so, use robot_2 (robot_1's lidar is blind from +8 to +92 deg), and keep the user next to it.
1. `robots.bash start map 2` (0.15 m/s, 0.6 rad/s, arena stop and scan masking off); `robots.bash stop 1` or leave robot_1 parked out of the way.
2. `laptop.bash slam robot_2`, then `laptop.bash rviz` and switch the Map display topic to `/map` so the user sees it grow.
3. Ask for approval, then `laptop.bash py scripts/real/explore.py robot_2 --duration 300` (Ctrl-C / `real_stop.bash` stop it).
   The explorer treats missing returns as open space: black or glass obstacles are a risk at 0.12 m/s; watch it.
4. When the map covers the room (closed walls, no grey holes in the working area): `laptop.bash slam-save <name>`, `laptop.bash kill slam`, show the user `out/real/maps/<name>.pgm` (Read it), and use `MAP=out/real/maps/<name>.yaml` from now on.

## 2. Initial localization (automatic)

1. Robots in shadow mode: `robots.bash start dry`; wait ~25 s (robot_2's unicast Cyclone discovery is slow).
2. `laptop.bash py scripts/real/localize.py scan` then `... localize.py match`: global match of each robot on the map.
   Read `out/real/state/overlay.png`: each robot's points must lie on the map's walls (in its colour); green points are off the map (the T, the other robot, clutter).
   A LOW SCORE or AMBIGUOUS match: ask the user to turn the robot to face a distinctive part of the room (or give a rough pose), rescan.
3. `... localize.py tee`: fits the checkpoint's T to the off-map points, proposes the arena (centre on the corridor axis through the T, x along the walls) and prints the robots and the T in the arena frame and the wall half-width.
   A POOR FIT: the T is not seen well (too far, occluded); ask the user where it is.
4. Restart the robots at the manoeuvre caps: `robots.bash start slow` (this resets their odometry), wait ~25 s.
5. AMCL on the laptop, started only after the robot stacks are up (it stalls for good if a robot's TF starts after it):
   `read P M < <(laptop.bash py scripts/real/localize.py args | paste -sd' ')` and `laptop.bash amcl "$P" "$M"`; wait ~15 s.
6. `... localize.py seed`: seeds both AMCLs and checks they answered. NOT ALIVE: restart AMCL (step 5), seed again.
7. Ask the user for the go (first motion), then `... localize.py spin robot_2` and `... localize.py spin robot_1` (360 deg in place, 0.12 m forward and back).
   Check the clearance first: the T and the other robot more than ~0.4 m away (overlay).
8. `... localize.py check` (1-sigma around 5-12 cm, 2-4 deg is normal) and `... localize.py verify` (AMCL vs a fresh scan match: OK below 12 cm / 4 deg).
   Then `... localize.py scan` and `... localize.py tee --poses amcl` for the final T pose with the converged poses.

## 3. Arena, goal, visualization

1. The arena is fixed by `tee`'s `map_to_arena`; if you changed it after step 5, restart AMCL with the new value (`laptop.bash amcl ...`, then `seed --poses amcl`).
2. Goal: propose one to the user in the arena frame (`goal X Y YAW_DEG`) and let them choose; defaults:
   - the T's heading, moved along the corridor axis (arena x) toward the side with room, keeping |y| small and the goal at least 0.5 m inside the arena stop (|x|, |y| <= 1.25 with `ARENA_HALF=1.5`);
   - the user has asked for goals farther than 0.6 m: offer ~1.0-1.5 m.
   The policy labels every return within 2.5 m of the arena centre as the T: walls closer than that (the corridor's are ~2.1 m) look like T to a robot within 1.5 m of them; robots pinned at the walls was the failure of the first run.
3. `laptop.bash rviz`: map, both scans, robot footprints, AMCL covariance, goal T with tolerance circle, arena square.
   To check it, screenshot it: `xwininfo -root -tree | grep -i rviz`, then `magick import -window <id> /tmp/.../rviz.png` and Read the image.
   If the user says RViz is frozen or closed: `laptop.bash rviz` again (it restarts it).

## 4. Shadow run, then the live episode

1. Restart the robots at the episode caps (`robots.bash start live`), then AMCL (`laptop.bash amcl "$(... localize.py args --poses amcl | head -1)" "$M"`) and `seed --poses amcl`; or keep `slow` caps for a first try.
2. `laptop.bash operator`, `laptop.bash op release` (a previous `real_stop.bash` latched `/estop`), `laptop.bash op "goal X Y DEG"`.
3. `laptop.bash policies shadow`, wait ~20 s, `laptop.bash op start`, `laptop.bash py scripts/real/monitor.py 15 3` (commands go to `/robot_i/cmd_vel_shadow`; gates stay `stale`), `laptop.bash op stop`.
   Check the commands make sense (turning toward / driving to the T) and `out/real/logs/policy_*.log` has no "no pose" warnings.
4. Ask for the go, then: `laptop.bash policies live`, `laptop.bash bag ep<N>`, wait ~20 s, `laptop.bash op status` (both gates `ok`), `laptop.bash op start`, `laptop.bash py scripts/real/monitor.py 100 3`.
   No mocap tracks the T, so the episode ends at the 90 s timeout or with `laptop.bash op "stop ok"` / `"stop fail"` (ask the user whether the T reached the goal).
5. After it: `laptop.bash kill policies`, `laptop.bash kill bag`; optionally `localize.py scan` + `tee --poses amcl` to measure the final T error against the goal.
   Data: `RUN_DIR` (`out/real/run_<date>/`): `episodes.csv`, `rec/` (per-step obs), `ep<N>/` bags; logs in `out/real/logs/`.

## Known facts and pitfalls (from the first sessions)

- robot_1 (lm2591): Limo ROS2, Humble, Fast DDS; YD lidar model 151 at 6.1 Hz (SDK 1.1.3 cannot change it), no returns between +8 and +92 deg; its onboard nav2 AMCL (built in agilex_ws) segfaults: AMCL runs on the laptop.
- robot_2 (lm2935): Limo Pro on an Orin Nano, Foxy; chassis on `/dev/ttyTHS0`; Cyclone DDS unicast to the laptop only (`~/swarp_ws/cyclonedds.xml`); T-mini Pro at 10 Hz.
- Both lidars are cropped to -116..+108 deg (`limo_bridge/config/ydlidar_tmini_pro.yaml`); L8b runs with `lidar_fov 4.18879` and simply gets no returns beyond.
- The laptop must be Humble with Fast DDS over UDPv4 only (`docker/fastdds_udp.xml`): any Jazzy participant crashes every robot process; shared-memory locators crash robot_2.
- Another Fast DDS participant lives on 10.42.0.1 in domain 28 (a colleague's machine); it is why robot_2 cannot use multicast.
- Restarting a robot stack resets its odometry: always restart the laptop AMCL afterwards and seed (`seed --poses amcl` if the robot did not move, else rescan + match).
- A battery running flat reboots the robot: check uptimes when something stops answering.
