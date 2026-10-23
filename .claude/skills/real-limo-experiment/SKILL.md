---
name: real-limo-experiment
description: Start a Push-T experiment on the two real AgileX Limos (lm2591 = robot_1, lm2935 = robot_2) with a swarp policy - robot bringup, automatic initial localization on the saved map (scan matching, AMCL seeding, verification, 360 deg spins only if needed), T detection, arena and goal, RViz, optional shadow run, live episode, emergency stop. Optionally maps the room autonomously first (only when the user asks for mapping). Use whenever the user wants to run, restart or prepare an episode/experiment on the real robots.
---

# Real two-Limo Push-T experiment

Everything is driven from the laptop host with the scripts in `scripts/real/` (lab constants in `scripts/real/config.bash`).
ROS on the laptop runs in the Humble container `swarp_real`; `laptop.bash py ...` runs a script inside it.
Read `docs/real_robot.md` for the background; this skill is the operational procedure that worked on 2026-10-23 (setup to ready in ~3 min of commands).

The policy is `CHECKPOINT` in `config.bash`: L9c (`pusht_lidar_limo_slim240_arena`, the default, run live on 2026-10-23) or L8b (`pusht_lidar_limo_slim240_s2`; set `CHECKPOINT` and `LIDAR_FOV=4.18879` to use it).
L9c observes the arena: `ARENA_HALF` is its input as well as the bridges' arena stop, so it must stay within 1.0–2.5 m, and after changing it restart the policies and `viz` and run `robots.bash caps <mode>` (the bridges take `arena_half` live).

## Speed: how to drive the scripts

- The robots start ONCE per session, in `dry`. Caps and dry/live switch at runtime with `robots.bash caps MODE` (~2 s, odometry and AMCL kept): never restart a robot stack to change caps.
  Modes: `dry` (nothing forwarded, 0.15/0.6) | `slow` (0.15/0.6) | `shadow` (nothing forwarded, episode caps) | `live` (`EPISODE_MAX_V`/`_W`).
- The scripts wait for readiness themselves: `robots.bash start` until the scans/odometry reach the laptop, `laptop.bash amcl` until AMCL is active (then it seeds and checks), `policies` until both are loaded (~3 s), `operator` until its prompt, `bag` until it records robot_2. Do not add sleeps or "wait ~N s" steps.
- `localize.py` takes several commands in one process: `scan match tee`, `check verify tee`, `spin robot_1 robot_2` (both at once).
- Chain independent laptop steps in one Bash call (see section 3). Run the episode monitor in the background and follow its output.
- Arguments that start with `-` must be glued: `"'--arena-centre=-0.803,-0.219'"`, `"'--from=...'"` (the inner quotes survive `laptop.bash py`).

## Safety rules (always)

- Before anything moves, tell the user the stop command and keep it ready: `scripts/real_stop.bash` (both robots' bridges send zeros and exit, latched `/estop`, laptop policies and scripts killed).
  Single robot: `ssh agilex@10.42.0.152 ~/swarp_ws/stop.sh` (robot_1) or `agilex@lm2935.local` (robot_2).
- Every motion goes through `robot_bridge` (clamp, 0.3 s watchdog, arena stop, `/estop`); never publish to `/robot_i/limo/cmd_vel`.
- The bridges stay in `dry`/`shadow` except during an approved motion: `caps slow` only for the spins, `caps live` only right before `op start`; back to `caps dry` right after.
  `caps` refuses to leave dry while an episode runs (`FORCE=1` overrides: do not use it without the user).
- If the user says the robots are not on the floor (on a table, being carried): nothing may leave dry; also `laptop.bash kill policies`.
- Ask the user before every motion: the spins and `caps live` + `start` of every episode; state the goal and the caps.
- If the user says stop: run `scripts/real_stop.bash` unless they say otherwise (a soft "you can stop the episode": `op stop` + `robots.bash caps dry` keeps the localization), then report the state.
- Monitor every live episode (`monitor.py`); a robot `PINNED` at the arena stop or gates flipping `arena` for >10 s: `laptop.bash op stop`, then ask.
- Never run `pkill -f` on a robot with a pattern that appears in your own ssh command line (it kills the ssh session); call `~/swarp_ws/stop.sh` by path.

## 0. Preflight (one call each, seconds)

1. `scripts/real/robots.bash status` (persistent ssh connections, <1 s): both reachable, clock offsets within ~20 ms (the policy pairs poses within 50 ms).
   A large offset: ask the user to run `sudo systemctl restart systemd-timesyncd` on that robot.
   A robot with a few-minute uptime was rebooted: its pose is unknown (step 2 does it anyway), and check robot_1's ports (pitfalls).
2. Container: `docker ps | grep swarp_real`; if missing, `scripts/real/laptop.bash container` (image `swarp_real_humble`, build with `docker build -f docker/Dockerfile.humble --build-arg USERNAME=$USER -t swarp_real_humble docker`).
3. After changing `src/limo_bridge`: `scripts/real/robots.bash install` (~30 s, does not restart the stacks; restart them to load a new bridge).
4. The map: `MAP` in `config.bash` (`out/real/maps/map_corridor2.yaml`, made 2026-10-23; `MAP=` overrides). AMCL runs on the laptop.
   If the user asks for mapping, do step 1 first; otherwise skip it.

## 1. Autonomous mapping (only when the user asks)

UNTESTED on the robots so far: say so, use robot_2 (robot_1's lidar is blind from +8 to +92 deg), and keep the user next to it.
1. `robots.bash start map 2` (0.15 m/s, 0.6 rad/s, arena stop and scan masking off); `robots.bash stop 1` or leave robot_1 parked out of the way.
2. `laptop.bash slam robot_2`, then `laptop.bash rviz` and switch the Map display topic to `/map` so the user sees it grow.
3. Ask for approval, then `laptop.bash py scripts/real/explore.py robot_2 --duration 300` (Ctrl-C / `real_stop.bash` stop it).
   The explorer treats missing returns as open space: black or glass obstacles are a risk at 0.12 m/s; watch it.
4. When the map covers the room (closed walls, no grey holes in the working area): `laptop.bash slam-save <name>`, `laptop.bash kill slam`, show the user `out/real/maps/<name>.pgm` (Read it), and use `MAP=out/real/maps/<name>.yaml` from now on.

## 2. Localization (automatic; usually no motion)

1. `robots.bash start dry` (~10–20 s; exits 1 listing the missing topics if a robot does not come up: read its `~/swarp_ws/robot.log`).
2. `laptop.bash py scripts/real/localize.py scan match tee` (~45 s), then Read `out/real/state/overlay.png`: each robot's points on the map's walls (in its colour), the T fit (orange) on its green points, the arena (grey) and its stop (magenta, `ARENA_HALF` + 0.25).
   A LOW SCORE or AMBIGUOUS match: ask the user to turn the robot to face a distinctive part of the room (or give a rough pose), rescan.
   A POOR FIT: the T is not seen well (too far, occluded); ask the user where it is.
   The proposed arena is often wrong: it walks from the T to the walls along the arena's y and passes through gaps in the map (then `walls at +-3.45 m` or more, robots far outside).
   Use the lab arena instead unless the user says otherwise: on `map_corridor2` centre map (-0.803, -0.219), x at 12.284 deg: rerun `tee "'--arena-centre=-0.803,-0.219'" --arena-yaw 12.284` (fast, no scan).
   The T and both robots must be inside `|x|, |y| <= ARENA_HALF` (the policy labels the T only out to `ARENA_HALF` + 0.5); if not, ask the user to move the T or the arena (or raise `ARENA_HALF`).
3. AMCL, started after the robot stacks are up: `read P M < <(scripts/real/laptop.bash py scripts/real/localize.py args | paste -sd' ')`, then `scripts/real/laptop.bash amcl "$P" "$M"` (~8 s: waits until active, seeds, checks both answered). NOT ALIVE: run it again.
4. `localize.py check verify` (~20 s): both `OK` (AMCL vs a fresh scan match below 12 cm / 4 deg) means localized: skip the spins.
5. Only if verify fails, or AMCL's 1-sigma stays large: the spins. Clearance first (the T and the other robot more than ~0.4 m from each robot, overlay), ask the user for the go, then in one call
   `robots.bash caps slow && laptop.bash py scripts/real/localize.py spin robot_1 robot_2; robots.bash caps dry` (~20 s, both robots at once: 360 deg in place, 0.12 m forward and back),
   then `localize.py check verify tee --poses amcl` plus the arena flags (`verify`'s fresh scans feed `tee`).
6. robot_2's AMCL can answer the seed yet never track (no `map -> robot_2/odom` on `/tf`; RViz shows no robot_2): check it from robot_2 itself (laptop-side tools are subject to the discovery delay, pitfalls):
   `ssh agilex@lm2935.local 'source ~/swarp_ws/env.sh; timeout 8 ros2 topic echo /tf 2>/dev/null | grep -A1 "frame_id: map" | grep child_frame_id | sort | uniq -c'` must list both `robot_1/odom` and `robot_2/odom`.
   Missing: `laptop.bash amcl "$(... localize.py args --poses amcl | head -1)" "$M"` and check again after ~30 s.

## 3. Laptop side and goal (one call, ~11 s; nothing moves)

1. Goal: propose one in the arena frame (`goal X Y YAW_DEG`) and let the user choose (or drag it in RViz); default: the T's heading, ~1 m from the T along arena x toward the side with room, |y| small, at least 0.5 m inside the arena stop.
2. ```
   scripts/real/robots.bash caps shadow; scripts/real/laptop.bash rviz; scripts/real/laptop.bash operator; \
   scripts/real/laptop.bash policies; scripts/real/laptop.bash op release; scripts/real/laptop.bash op "goal X Y DEG"
   ```
   The policies publish on `/robot_i/cmd_vel`; the bridges in `shadow` forward nothing. `op release` clears the `/estop` a previous `real_stop.bash` latched.
3. `laptop.bash op status`: both poses fresh, both gates `dry_run:ok` (a single `stale` is a missed command at robot_1's 6 Hz lidar; recheck), the goal. `grep -ci "no pose" out/real/logs/policy_*.log` should be 0.
4. RViz: map, both scans, footprints, AMCL covariance, goal T with tolerance circle, arena square. A restarted RViz shows robot_2 only after up to ~30 s (discovery).
   To check it, screenshot it: `xwininfo -root -tree | grep -i rviz`, then `magick import -window <id> <scratchpad>/rviz.png` and Read the image.
   If the user says RViz is frozen or closed: `laptop.bash rviz` again.

## 4. Episodes

1. Shadow episode: OPTIONAL, skip it unless the user asks or something changed since the last live run (checkpoint, `ARENA_HALF`, map, lidar config). With the bridges in `shadow`: `op start`, `laptop.bash py scripts/real/monitor.py 15 3`, `op stop`; the commands should turn toward / drive to the T, the gates read the real gate behind `dry_run:`.
2. Ask for the go (goal, caps 0.3 m/s / 1.0 rad/s, arena stop at `ARENA_HALF` + 0.25), then `laptop.bash bag ep<N>` (returns once it records robot_2), and in the background
   `robots.bash caps live && laptop.bash op start && laptop.bash py scripts/real/monitor.py 100 3`; follow its output (`tail -f` with a timeout) for PINNED robots.
   No mocap tracks the T, so the episode ends at the 90 s timeout or with `laptop.bash op "stop ok"` / `"stop fail"` (ask the user whether the T reached the goal).
3. After it: `robots.bash caps dry`, `laptop.bash kill bag`; optionally `localize.py scan tee --poses amcl` (plus the arena flags) for the final T error.
   The next episode needs no setup: new goal (`op "goal ..."` or RViz), bag, `caps live`, `op start`.
4. Data: `RUN_DIR` (`out/real/run_<date>/`): `episodes.csv`, `rec/` (per-step obs), `ep<N>/` bags; logs in `out/real/logs/`.
   A restarted operator numbers episodes from 0 again and the policies overwrite `rec/ep0000_*`: use a new `RUN_DIR=` for a new operator session when the records matter.

## Known facts and pitfalls

- robot_1 (10.42.0.152; lm2591.local's mDNS stopped answering): Limo ROS2, Humble, Fast DDS; YD lidar model 151 at 6.1 Hz (SDK 1.1.3 cannot change it), no returns between +8 and +92 deg; its onboard nav2 AMCL segfaults: AMCL runs on the laptop.
- robot_1's udev rules match the USB bus number (`KERNELS=="1-7*"` chassis, `"1-3*"` lidar): after some reboots the devices enumerate on bus 3, `/dev/ttylimo` and `/dev/ydlidar` do not exist and `robot.log` says `Failed to open: '/dev/ttylimo'`.
  Workaround: `ROBOT_EXTRA_1="port_name:=ttyUSB1 lidar_port:=/dev/serial/by-path/pci-0000:00:14.0-usb-0:3:1.0-port0" robots.bash start dry 1` (check `udevadm info -q property -n /dev/ttyUSB1 | grep ID_PATH` reads `usb-0:7.4`, the chassis; the ttyUSB number can change).
  Fix (ask the user, sudo on robot_1): `KERNELS=="*-7*"` and `"*-3*"` in `/etc/udev/rules.d/{limo-usb,ydlidar}.rules`, `sudo udevadm control --reload && sudo udevadm trigger`.
- robot_2 (lm2935): Limo Pro on an Orin Nano, Foxy; chassis on `/dev/ttyTHS0`; Cyclone DDS unicast to the laptop only (`~/swarp_ws/cyclonedds.xml`); T-mini Pro at 10 Hz.
  Every NEW laptop process (AMCL, RViz, a script, the bag) sees robot_2 only at robot_2's next unicast announcement, up to ~30 s later; robot_2's own new processes discover the laptop at once.
  Services do not cross between the laptop's Fast DDS and robot_2's Cyclone (topics do): `caps` therefore runs onboard over ssh (`python3 -m limo_bridge.caps`, environment in `~/swarp_ws/env.sh`).
- Both lidars are cropped to -116..+108 deg (`limo_bridge/config/ydlidar_tmini_pro.yaml`); L9c runs with `lidar_fov 3.83972` (220 deg, missing only +108..+110) and L8b with `4.18879` (240 deg), and they simply get no returns beyond.
- The laptop must be Humble with Fast DDS over UDPv4 only (`docker/fastdds_udp.xml`): any Jazzy participant crashes every robot process; shared-memory locators crash robot_2.
- Another Fast DDS participant lives on 10.42.0.1 in domain 28 (a colleague's machine); it is why robot_2 cannot use multicast.
- Restarting a robot stack resets its odometry: restart the laptop AMCL afterwards and seed (`args --poses amcl` if the robot did not move, else rescan + match). `caps` does not restart anything.
- A battery running flat reboots the robot: check uptimes when something stops answering.
