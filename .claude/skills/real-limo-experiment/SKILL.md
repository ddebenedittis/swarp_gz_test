---
name: real-limo-experiment
description: Start a Push-T experiment on the two real AgileX Limos (lm2591 = robot_1, lm2935 = robot_2) with a swarp policy - robot bringup, automatic initial localization on the saved map (scan matching, AMCL seeding, verification, 360 deg spins only if needed), T detection, arena and goal, RViz, optional shadow run, live episode, emergency stop. Optionally maps the room autonomously first (only when the user asks for mapping). Use whenever the user wants to run, restart or prepare an episode/experiment on the real robots.
effort: medium
---

# Real two-Limo Push-T experiment

Everything is driven from the laptop host with the scripts in `scripts/real/` (lab constants in `scripts/real/config.bash`).
ROS on the laptop runs in the Humble container `swarp_real`; `laptop.bash py ...` runs a script inside it.
Read `docs/real_robot.md` for the background; this skill is the operational procedure that worked on 2026-10-23.

The policy is `CHECKPOINT` in `config.bash`: L9c (`pusht_lidar_limo_slim240_arena`, the default, run live on 2026-10-23) or L8b (`pusht_lidar_limo_slim240_s2`; set `CHECKPOINT` and `LIDAR_FOV=4.18879` to use it).
L9c observes the arena: `ARENA_HALF` is its input as well as the bridges' arena stop, so it must stay within 1.0–2.5 m, and after changing it restart the policies and `viz` and run `robots.bash caps <mode>` (the bridges take `arena_half` live).
The map and the lab arena in it are `MAP`, `ARENA_CENTRE`, `ARENA_YAW` in `config.bash` (`map_corridor2_r1`, centre (1.934, 0.354), 12.5 deg); `export` overrides, set them once per Bash call that needs them.

## Speed: how to drive the scripts

A whole session is a handful of calls: `robots.bash status` → `bringup.bash start` → Read the overlay → goal → `episode.bash` per episode.
- Never `sleep` to wait for something: every script waits for its own readiness and returns.
  robot_2 announces itself every 1 s (Cyclone `SPDPInterval` in `~/swarp_ws/cyclonedds.xml`, written by `robots.bash install`), so new laptop processes see it within ~1 s; a robot_2 installed before this change still takes up to 30 s: run `robots.bash install 2` and restart its stack once.
- Do not screenshot RViz or check robot_2's TF over ssh unless something looks wrong (the user says RViz is wrong, `verify` or `op status` fails).
- Do not re-read scripts or explore the code during a session: the commands below are complete.
- The robots start ONCE per session, in `dry`. Caps and dry/live switch at runtime with `robots.bash caps MODE` (~2 s, odometry and AMCL kept): never restart a robot stack to change caps.
  Modes: `dry` (nothing forwarded, 0.15/0.6) | `slow` (0.15/0.6) | `shadow` (nothing forwarded, episode caps) | `live` (`EPISODE_MAX_V`/`_W`).
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
- Monitor every live episode: `episode.bash` runs `monitor.py --stop-pinned`, which stops the episode itself when a robot stays at the arena stop ~9 s, and always returns the bridges to dry; follow its output and ask the user after a PINNED stop.
- Never run `pkill -f` on a robot with a pattern that appears in your own ssh command line (it kills the ssh session); call `~/swarp_ws/stop.sh` by path.

## 0. Preflight (one call each, seconds)

1. `scripts/real/robots.bash status` (persistent ssh connections, <1 s): both reachable, clock offsets within ~20 ms (the policy pairs poses within 50 ms).
   A large offset: ask the user to run `sudo systemctl restart systemd-timesyncd` on that robot.
   A robot with a few-minute uptime was rebooted: its pose is unknown (step 2 does it anyway), and check robot_1's ports (pitfalls).
2. Container: `docker ps | grep swarp_real`; if missing, `scripts/real/laptop.bash container` (image `swarp_real_humble`, build with `docker build -f docker/Dockerfile.humble --build-arg USERNAME=$USER -t swarp_real_humble docker`).
3. After changing `src/limo_bridge`: `scripts/real/robots.bash install` (~30 s, does not restart the stacks; restart them to load a new bridge).
4. The map: `MAP` in `config.bash` (`out/real/maps/map_corridor2_r1.yaml`, 2026-10-23; `MAP=` overrides, with its `ARENA_CENTRE`/`ARENA_YAW`). AMCL runs on the laptop.
   If the user asks for mapping, do step 1 first; otherwise skip it.

## 1. Autonomous mapping (only when the user asks)

UNTESTED on the robots so far: say so, use robot_2 (robot_1's lidar is blind from +8 to +92 deg), and keep the user next to it.
1. `robots.bash start map 2` (0.15 m/s, 0.6 rad/s, arena stop and scan masking off); `robots.bash stop 1` or leave robot_1 parked out of the way.
2. `laptop.bash slam robot_2`, then `laptop.bash rviz` and switch the Map display topic to `/map` so the user sees it grow.
3. Ask for approval, then `laptop.bash py scripts/real/explore.py robot_2 --duration 300` (Ctrl-C / `real_stop.bash` stop it).
   The explorer treats missing returns as open space: black or glass obstacles are a risk at 0.12 m/s; watch it.
4. When the map covers the room (closed walls, no grey holes in the working area): `laptop.bash slam-save <name>`, `laptop.bash kill slam`, show the user `out/real/maps/<name>.pgm` (Read it), and use `MAP=out/real/maps/<name>.yaml` from now on.

## 2. Bringup and localization (one call, ~30 s; nothing moves)

1. `scripts/real/bringup.bash start` (`bringup.bash` alone when the stacks already run; `bringup.bash localize` after the user moved robots or the T by hand).
   It starts both stacks in dry (exits 1 listing the missing topics if a robot does not come up: read its `~/swarp_ws/robot.log`), then in parallel: the laptop side (caps shadow, RViz + viz, operator, policies, op release) and the localization (`localize.py scan match tee`, AMCL seeded there, `check verify`).
   It prints the matches, the T and both robots in the arena frame, AMCL vs a fresh scan match per robot (`OK`), and both gates (`dry_run:ok`; a single `stale` is a missed command at robot_1's 6 Hz lidar).
2. Read `out/real/state/overlay.png`: each robot's points on the map's walls (in its colour), the T fit (orange) on its green points, the arena (grey) and its stop (magenta, `ARENA_HALF` + 0.25).
   A LOW SCORE or AMBIGUOUS match, or a verify `CHECK`: ask the user to turn the robot to face a distinctive part of the room (or give a rough pose), then `bringup.bash localize`.
   T NOT FOUND or a POOR FIT: the T is not seen well (too far, occluded); ask the user where it is (the arena is written anyway).
   The T and both robots must be inside `|x|, |y| <= ARENA_HALF` (the policy labels the T only out to `ARENA_HALF` + 0.5); if not, ask the user to move the T or the arena (`ARENA_CENTRE`/`ARENA_YAW`, or `auto` to propose one from the walls, often wrong: it passes through gaps in the map) or raise `ARENA_HALF`.
3. AMCL NOT ALIVE (the seed resends for 40 s): the robot's TF does not reach the laptop; check `robots.bash status`, then `bringup.bash localize`.
4. Only if verify keeps failing or AMCL's 1-sigma stays large: the spins. Clearance first (the T and the other robot more than ~0.4 m from each robot, overlay), ask the user for the go, then in one call
   `robots.bash caps slow && laptop.bash py scripts/real/localize.py spin robot_1 robot_2; robots.bash caps shadow` (~20 s, both robots at once: 360 deg in place, 0.12 m forward and back),
   then `laptop.bash py scripts/real/localize.py check verify tee --poses amcl`.

## 3. Goal (nothing moves)

Propose a goal in the arena frame (`goal X Y YAW_DEG`) and let the user choose (or drag it in RViz); default: the T's heading, ~1 m from the T along arena x toward the side with room, |y| small, at least 0.5 m inside the arena stop.
Set it with `laptop.bash op "goal X Y DEG"`.
RViz shows map, both scans, footprints, AMCL covariance, goal T with tolerance circle, arena square; if the user says it is frozen or closed: `laptop.bash rviz`.

## 4. Episodes

1. Shadow episode: OPTIONAL, skip it unless the user asks or something changed since the last live run (checkpoint, `ARENA_HALF`, map, lidar config). With the bridges in `shadow`: `op start`, `laptop.bash py scripts/real/monitor.py 15 3`, `op stop`; the commands should turn toward / drive to the T, the gates read the real gate behind `dry_run:`.
2. Ask for the go (goal, caps 0.3 m/s / 1.0 rad/s, arena stop at `ARENA_HALF` + 0.25), then run `EXP=<exp_name> VIDEO=<n> scripts/real/episode.bash ep<N>` in the background (bag, caps live, op start, monitor; on exit caps dry, bag stopped, episode saved: step 5; `VIDEO` only if the user gave one) and follow its output with the Monitor tool (lines with `PINNED`, `ep=-1`, `episode` and `over`), not with `sleep` + `tail` polls.
   No mocap tracks the T, so the episode ends at the 90 s timeout or with `laptop.bash op "stop ok"` / `"stop fail"` (ask the user whether the T reached the goal); the script then finishes by itself.
3. Optionally `laptop.bash py scripts/real/localize.py scan tee --poses amcl` for the final T error.
   The next episode needs no setup: new goal (`op "goal ..."` or RViz), `episode.bash ep<N+1>`. After the user moved things by hand: `bringup.bash localize` first.
4. Raw data: `RUN_DIR` (`out/real/run_<date>/`): `episodes.csv`, `rec/` (per-step obs), `ep<N>/` bags; logs in `out/real/logs/`.
   A restarted operator numbers episodes from 0 again and the policies overwrite `rec/ep0000_*`: use a new `RUN_DIR=` for a new operator session when the records matter.
5. Save every live episode (the user wants all of them archived on this laptop):
   - At the start of the session ask for the experiment name (short, e.g. `l9c_a125`; keep it while the setup is unchanged) and, per episode, the video number if the user films it.
   - `EXP=<exp_name> [VIDEO=<n>] episode.bash` saves by itself at the end; otherwise (or for an episode run before) `scripts/real/save_exp.bash <exp_name> ep<N> [<video>|-] [<episode index>]`.
   - It writes `~/Documents/swarp_exp/<yyyy-mm-dd>/<exp_name>_<id>/` (id 01, 02, ... per day: bag, `rec/`, `episode.csv`, overlay and poses, logs, `meta.txt`) and appends an entry to `~/Documents/swarp_exp/<yyyy-mm-dd>/summary.md` (video, result, goal, starts) ending in `- Synthesis: TODO`.
   - Replace that `TODO` (Edit) with 1–3 sentences on the outcome: did the T reach the goal (the user's verdict), how the robots behaved (approach, pushing, getting stuck, PINNED or arena stops, localization trouble) and why it ended; from the monitor output and what the user said.
   - A video number given later: update the entry's `- Video:` line and `video` in its `meta.txt`.

## Known facts and pitfalls

- robot_1 (10.42.0.152; lm2591.local's mDNS stopped answering): Limo ROS2, Humble, Fast DDS; YD lidar model 151 at 6.1 Hz (SDK 1.1.3 cannot change it), no returns between +8 and +92 deg; its onboard nav2 AMCL segfaults: AMCL runs on the laptop.
- robot_1's udev rules match the USB bus number (`KERNELS=="1-7*"` chassis, `"1-3*"` lidar): after some reboots the devices enumerate on bus 3, `/dev/ttylimo` and `/dev/ydlidar` do not exist and `robot.log` says `Failed to open: '/dev/ttylimo'`.
  Workaround: `ROBOT_EXTRA_1="port_name:=ttyUSB1 lidar_port:=/dev/serial/by-path/pci-0000:00:14.0-usb-0:3:1.0-port0" robots.bash start dry 1` (check `udevadm info -q property -n /dev/ttyUSB1 | grep ID_PATH` reads `usb-0:7.4`, the chassis; the ttyUSB number can change).
  Fix (ask the user, sudo on robot_1): `KERNELS=="*-7*"` and `"*-3*"` in `/etc/udev/rules.d/{limo-usb,ydlidar}.rules`, `sudo udevadm control --reload && sudo udevadm trigger`.
- robot_2 (lm2935): Limo Pro on an Orin Nano, Foxy; chassis on `/dev/ttyTHS0`; Cyclone DDS unicast to the laptop only (`~/swarp_ws/cyclonedds.xml`); T-mini Pro at 10 Hz.
  Every NEW laptop process (AMCL, RViz, a script, the bag) sees robot_2 only at robot_2's next unicast announcement (every 1 s with `SPDPInterval`, 30 s without); robot_2's own new processes discover the laptop at once.
  robot_2's AMCL tracking can be checked from robot_2 itself: `ssh agilex@lm2935.local 'source ~/swarp_ws/env.sh; timeout 8 ros2 topic echo /tf 2>/dev/null | grep -A1 "frame_id: map" | grep child_frame_id | sort | uniq -c'` must list both `robot_1/odom` and `robot_2/odom`.
  Services do not cross between the laptop's Fast DDS and robot_2's Cyclone (topics do): `caps` therefore runs onboard over ssh (`python3 -m limo_bridge.caps`, environment in `~/swarp_ws/env.sh`).
- Both lidars are cropped to -116..+108 deg (`limo_bridge/config/ydlidar_tmini_pro.yaml`); L9c runs with `lidar_fov 3.83972` (220 deg, missing only +108..+110) and L8b with `4.18879` (240 deg), and they simply get no returns beyond.
- The laptop must be Humble with Fast DDS over UDPv4 only (`docker/fastdds_udp.xml`): any Jazzy participant crashes every robot process; shared-memory locators crash robot_2.
- Another Fast DDS participant lives on 10.42.0.1 in domain 28 (a colleague's machine); it is why robot_2 cannot use multicast.
- Restarting a robot stack resets its odometry: restart the laptop AMCL afterwards and seed (`args --poses amcl` if the robot did not move, else rescan + match). `caps` does not restart anything.
- A battery running flat reboots the robot: check uptimes when something stops answering.
