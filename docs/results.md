# Gazebo results: Limo-scale lidar Push-T policies

How often swarp `pusht_lidar` policies trained at Limo scale solve Push-T in Gazebo, compared with swarp itself.

## Setup

Gazebo side, all rows on commit `6caede2`:
- Two Limos in a 4×4 m arena with no walls.
- The T is 1.0×0.3 m bar plus 0.7×0.3 m stem, 0.5 kg, with isotropic Coulomb floor friction μ 0.3 (`PlanarCoulombFriction`).
- Lidar: 360 samples, 0.10–8 m, 10 Hz, 0.01 m noise, mounted 0.103 m ahead of the robot centre.
- One `policy_node` per robot steps every 0.2 s on the latest scan and holds the command in between.
- Poses and velocities are ground truth, the stand-in for mocap.

Episodes:
- `episode_manager` draws the initial states as swarp's reset does at full difficulty.
- It scores success as swarp does: T within 0.15 m and 0.25 rad of the goal.
- An episode times out after 450 steps (90 s).

Runs:
- Each row is 60 episodes: 3 shards of 20.
- Shard seeds are 0–2 for greedy rows and 100–102 for sampled rows.
- Every run is recorded (`record:=true`).
- "Sampled" draws `squash(loc + scale·ε)` as the trainer's rollout does; "greedy" is `squash(loc)`.

swarp references:
- Computed with `scripts/swarp_reference.py`: 512 envs, seed 123, one un-reset 450-step episode per env, each checkpoint with its own stored scenario kwargs.
- swarp was at the head of `pusht-lidar-limo` when each reference was computed (`4cc542e` to `db4ee06`).
- L2's greedy reference reproduces the 0.857 from its training-time eval.

Intervals are Wilson 95%.
The plan's transfer criterion is a Gazebo rate of at least 80% of swarp's.

## Results

| checkpoint | mode | swarp | Gazebo | Gazebo / swarp | median solve time, Gazebo (swarp) | failures: left arena / near goal, angle off / near goal, position off / T far | squeeze lock: episodes > 10 s (share of steps) |
|---|---|---|---|---|---|---|---|
| L2 | greedy | 0.857 [0.824, 0.885] | 34/60 = 0.567 [0.441, 0.684] | 0.66 | 46.6 s (25.8 s) | 2 / 6 / 2 / 16 | 1 (1.2%) |
| L3b | greedy | 0.986 [0.972, 0.993] | 21/60 = 0.350 [0.242, 0.476] | 0.35 | 23.8 s (19.4 s) | 3 / 17 / 3 / 16 | 32 (32.4%) |
| L3b | sampled | 0.998 [0.989, 1.000] | 20/60 = 0.333 [0.227, 0.459] | 0.33 | 21.5 s (22.2 s) | 8 / 13 / 1 / 18 | 32 (24.7%) |
| L4 | greedy | 0.973 [0.955, 0.984] | 53/60 = 0.883 [0.778, 0.942] | 0.91 | 25.0 s (20.8 s) | 1 / 1 / 0 / 5 | 0 (0.2%) |
| L4 | sampled | 0.996 [0.986, 0.999] | 57/60 = 0.950 [0.863, 0.983] | 0.95 | 26.4 s (24.6 s) | 2 / 0 / 0 / 1 | 0 (0.0%) |
| L5r | greedy | 0.992 [0.980, 0.997] | 52/60 = 0.867 [0.758, 0.931] | 0.87 | 22.9 s (19.6 s) | 1 / 1 / 0 / 6 | 1 (0.6%) |
| L5r | sampled | 0.998 [0.989, 1.000] | 51/60 = 0.850 [0.739, 0.919] | 0.85 | 27.0 s (22.2 s) | 1 / 3 / 0 / 5 | 0 (0.0%) |
| L5s | greedy | 0.986 [0.972, 0.993] | 55/60 = 0.917 [0.819, 0.964] | 0.93 | 26.6 s (20.6 s) | 0 / 4 / 1 / 0 | 0 (0.0%) |
| L5s | sampled | 0.998 [0.989, 1.000] | 55/60 = 0.917 [0.819, 0.964] | 0.92 | 31.4 s (22.6 s) | 4 / 0 / 0 / 1 | 0 (0.0%) |
| L5 | greedy | 0.959 [0.938, 0.973] | 48/60 = 0.800 [0.682, 0.882] | 0.83 | 23.9 s (19.4 s) | 1 / 4 / 5 / 2 | 0 (0.0%) |

Failure classes are checked in this order:
- "left arena": the T's centroid went past ±2 m at some step.
- "near goal, angle off": the final position error is below 0.3 m and the angle error is at least 0.25 rad.
- "near goal, position off": the final position error is below 0.3 m and the angle error is below 0.25 rad.
- "T far": any other failure.

"Squeeze lock" means both robots are flush against opposite faces of the T, facing each other, and neither turns although each commands |ω| > 0.3 rad/s.

L2 sampled and L5 sampled were not run on `6caede2`.
A partial L2 sampled run on `a381ee8` solved 13/20.
L2 greedy before the Coulomb floor friction (DART friction, `a1f6f30`) solved 31/50 = 0.620 [0.482, 0.741].

## What each checkpoint embodies

All are seed 0, two pushers, 5 Hz control (dt 0.2 s), 450 steps.

- **L2** (`L2-H1-limo-s0`, swarp `92c1282`): the Limo-scale task at 5 Hz, with swarp's viscous T drag, frictionless contacts, ideal actuation and an ideal centre-mounted lidar.
- **L3b** (`L3b-DRmass-limo-s0-ext`, `4cc542e`): adds randomised actuation (servo acceleration and command-gain ranges) and the lidar model (0.103 m mount offset, 0.1 m minimum range, 0.01 m noise), with a robot mass of 6.7 kg. Iterations 4001–6000.
- **L4** (`L4-DRcoul-limo-s0-ext`, `64998fc`): replaces the T's viscous drag with Coulomb floor friction (μ 0.2–0.5), with a robot mass of 2.0 kg. Iterations 8001–16000.
- **L5r** (`L5r-wheel-limo-s0-fromL4`, `c467889`): wheel-faithful robot contacts, so a robot cannot spin while it is squeezed (lateral wheel friction 0.2, turn torque 3.5 N·m). Fine-tuned from L4's best checkpoint.
- **L5s** (`L5s-wheelstall-limo-s0-fromL5r`, `db4ee06`): adds a turn stall when the contact resists harder than the drive. Fine-tuned from L5r; best checkpoint.
- **L5** (`L5-wheel-limo-s0`, `c467889`): L5r's physics trained from scratch, as a second data point.

## Findings

- L4, L5r, L5s and L5 meet the 80% transfer criterion in every mode run.
- L2 and L3b do not.
- L5s transfers best: 0.917 in both modes, 0.93 and 0.92 of swarp.

**L3b's collapse is the squeeze lock.**
- In 32 of 60 episodes in each mode, the two robots drive into opposite faces of the T and stay locked for more than 10 s. Every one of those episodes failed.
- A box robot pressed flush against a face that the other robot holds cannot swing its corner, so it cannot turn. Gazebo and a quasi-static reference agree on this (`scripts/pressed_turn_reference.py`).
- swarp before `c467889` integrated the heading kinematically and turned the robots anyway, so this squeeze cost nothing in training.
- The wheel-faithful contacts of L5r and L5s bring the lock to 0–1 episodes per run.
- L4 also avoids it (no episode locked for more than 10 s), although it was trained without that physics.

**L2 is slow and stalls.**
- Its median solve time is 46.6 s against 25.8 s in swarp.
- Its robots command |v| < 0.04 m/s in a median 15% of steps.
- Most of its failures are "T far".

**The arena differs between the two simulators.**
- swarp clamps the whole T inside the arena, keeping its centroid within ±(2 − 0.661) = ±1.34 m.
- Gazebo has no walls.
- In 12–26 of 60 episodes per run, the T goes past that clamp, into states swarp never produces.
- The T then leaves the ±2 m square in 0–8 failures per run, including 4 of L5s sampled's 5 failures (final position errors 0.97–2.10 m).
- Walls in Gazebo, or a matching clamp, would remove this mismatch.

**L5s greedy's 5 failures all end near the goal.**
- Four have the angle off by 0.31–0.40 rad against the 0.25 rad tolerance; the fifth has the angle in tolerance but ends 0.27 m from the goal position.
- Their robots stall in about 30% of steps.

**Control timing is clean.**
- Since `5c411eb` the policy waits for a pose near the scan stamp instead of skipping the step.
- Fewer than 0.1% of control steps are late, against 1.7% before that commit.

## Reproduce

```bash
# Gazebo: 3 shards x 20 episodes per mode (host). Use DOMAIN_BASE to run two batches at once, and BUILD=0 for the second.
src/pusht_policy/scripts/eval_shards.bash /opt/swarp/runs/remote-davide-cf/<run>/<ckpt>.pt 3 20 <name>_greedy 0 record:=true
BUILD=0 DOMAIN_BASE=70 src/pusht_policy/scripts/eval_shards.bash /opt/swarp/runs/remote-davide-cf/<run>/<ckpt>.pt 3 20 <name>_sampled 100 record:=true sample:=true

# failure classes, stalls, yaw and squeeze locks, T past swarp's clamp
PYTHONPATH=src/pusht_policy python3 -m pusht_policy.failure_modes out/<name>_greedy out/<name>_sampled

# swarp references (swarp's venv)
cd ../swarp && uv run --no-sync python ../swarp_gz_test/src/pusht_policy/scripts/swarp_reference.py runs/remote-davide-cf/<run>/<ckpt>.pt 512 123
```

The runs behind the table are in `out/{L2,L3b-ext,L4-ext,L5r,L5s,L5}_{greedy,sampled}_6caede2/` (gitignored); the pre-friction L2 run is `out/L2-H1-limo-s0_greedy_prefix/`.
Each holds `episodes.csv` and `rec/` per shard, the shard logs, and the command line in `cmd`.

## Pose source: ground truth vs AMCL

On the measured Limo specs (4.8 kg, T-mini Pro: 400 samples per turn, 0.02–12 m, σ 0.015 m), with the onboard safety bridge in the loop and the landmarks outside the arena, L5s greedy on seeds 0–1 (20 episodes, the same initial states in both rows):

| pose source | Gazebo | median solve time |
|---|---|---|
| ground truth | 19/20 = 0.950 [0.764, 0.991] | 26.0 s |
| AMCL (nav2, one per robot, `limo_bridge/config/amcl.yaml`) | 15/20 = 0.750 [0.531, 0.888] | 59.0 s |

AMCL's pose error over those episodes: position mean 0.161 m, median 0.081 m, p95 0.469 m; heading mean 0.033 rad, p95 0.136 rad; details and the mechanism in [real_robot.md](real_robot.md#gazebo-stand-ins).

L6 (`L6-real-limo-s0-fromL5s` best, trained with the real-Limo sensor, localisation and latency randomisation), greedy, the lidar and the policy at a 240° field of view, IMU-heading odometry, same seeds:

| pose source | Gazebo |
|---|---|
| ground truth | 1/20 = 0.050 [0.009, 0.236] |
| AMCL, full scan | 2/20 = 0.100 [0.028, 0.301] |
| AMCL, arena returns removed (shipped) | 2/20 = 0.100 [0.028, 0.301] |

At 240° these rows say nothing about localisation: the field of view itself breaks the policy (below). The AMCL error tables are in [real_robot.md](real_robot.md#gazebo-stand-ins).

### The field of view is defined at the robot centre in swarp, at the mount on a lidar

swarp blanks the robot-centre rays more than `fov/2` off the heading; a real lidar, and the cropped gz one, loses everything more than `fov/2` off the heading as seen from its mount, 0.103 m ahead.
A T against the robot's side at centre bearing 101–112° (rays k = ±9, ±10) is at ~125–145° from the mount: a physical 240° hides it while swarp shows it.
Ground-truth poses, seeds 0–1, 20 episodes each:

| checkpoint | mode | gz lidar | policy blanking | Gazebo | failures: left arena / near goal, angle off / T far |
|---|---|---|---|---|---|
| L6 | greedy | full | none | 17/20 = 0.850 | 0 / 1 / 2 |
| L6 | sampled | full | none | 15/20 = 0.750 | 0 / 2 / 3 |
| L6 | greedy | cropped to 240° at the mount | 240° | 1/20 = 0.050 | 7 / 2 / 10 |
| L6 | sampled | cropped to 240° | 240° | 0/20 | 5 / 1 / 14 |
| L5s | greedy | cropped to 240° | 240° | 1/20 = 0.050 | 7 / 2 / 10 |
| L6 | greedy | full | 240° | 11/20 = 0.550 | 3 / 1 / 5 |
| L6 | sampled | full | 240° | 15/20 = 0.750 | 4 / 0 / 1 |

Against analytic centre rays on the recorded true poses, 16% of the T-hit rays in the policy's frames are wrong with the crop, nearly all at k = ±9, ±10 (k = 10: 7680 of 8419), against 1.5% without it.
`eval.launch.py sim_lidar_fov:=` crops the gz lidar independently of the policy's `lidar_fov`.

The deployment pipeline itself is at parity: `test_obs_parity.py` checks the obs assembly on L6's configuration (240° per centre ray and cut at the mount) bit for bit, and `test_sensor_matches_swarp` feeds synthetic 400-sample mount scans of swarp's own states through `scan_to_rays`, `label_rays` and the field-of-view rule (61, 19 and 7 of 12800 ray classes differ at 360°, 240° per ray and 240° at the mount; p99 range error below 3.4 mm).
L6's checkpoint differs from L5s only in sensor and noise kwargs (obs_dim, roles, action bounds, dt, frame stack unchanged).
swarp's `lidar_fov_at_mount` (8aaa57d) cuts the field of view at the mount as the sensor does; `policy_node` reads it from the checkpoint and then crops the raw scan (`obs.crop_fov_at_mount`) instead of blanking centre rays.
The 20 initial states with every run's outcome, for paired swarp runs: `out/L6_paired_states.csv` (`scripts/paired_states.py`).

```bash
src/pusht_policy/scripts/eval_shards.bash /opt/swarp/runs/remote-davide-cf/L5s-wheelstall-limo-s0-fromL5r/pusht_lidar_limo_dr_wheel_best.pt 2 10 L5s_gt_landmarks 0 landmarks:=true record:=true
BUILD=0 DOMAIN_BASE=80 src/pusht_policy/scripts/eval_shards.bash /opt/swarp/runs/remote-davide-cf/L5s-wheelstall-limo-s0-fromL5r/pusht_lidar_limo_dr_wheel_best.pt 2 10 L5s_amcl 0 pose_source:=amcl record:=true
ros2 run limo_real pose_error_merge out/L5s_amcl/shard0/pose_error out/L5s_amcl/shard1/pose_error
```

## L7: field of view cut at the mount, AMCL-sized pose drift

L7 (`L7-amcl-mount-limo-s0-fromL6` best, recipe `pusht_lidar_limo_amcl`), greedy, seeds 0–3 (40 episodes per cell, 4 shards of 10 run 4 at a time), landmarks on, the policy cutting the raw scan at the mount (`lidar_fov_at_mount` from the checkpoint, no per-ray blanking) to the same field of view as the gz lidar:

| pose source | FOV | Gazebo | median solve time | failures: left arena / near goal, angle off / near goal, position off / T far |
|---|---|---|---|---|
| ground truth | 360° | 18/40 = 0.450 [0.307, 0.602] | 41.9 s | 1 / 5 / 3 / 13 |
| ground truth | 240° at the mount | 5/40 = 0.125 [0.055, 0.261] | 61.0 s | 3 / 7 / 3 / 22 |
| AMCL | 360° | 11/36 = 0.306 [0.180, 0.469] | 54.6 s | 3 / 9 / 0 / 13 |
| AMCL | 240° at the mount | 3/40 = 0.075 [0.026, 0.199] | 81.8 s | 5 / 6 / 2 / 24 |

The 360° AMCL cell has 36 episodes: its fourth shard's Gazebo failed to start twice (`/clock` never came) and its third attempt finished 6.
AMCL error during these runs:

| | 360° | 240° at the mount |
|---|---|---|
| position mean / median / p95 / max [m] | 0.096 / 0.069 / 0.297 / 1.27 | 0.141 / 0.094 / 0.457 / 1.92 |
| heading mean / median / p95 [rad] | 0.018 / 0.010 / 0.059 | 0.020 / 0.014 / 0.052 |
| per-episode bias, position mean / std / max [m] | 0.067 / 0.045 / 0.33 | 0.111 / 0.117 / 0.74 |
| jitter about the bias (position per axis, heading) | 0.085 m, 0.029 rad | 0.132 m, 0.029 rad |
| decorrelation time (position, heading) | 5.2 s, 2.7 s | 8.0 s, 3.9 s |
| position median / p95, free driving / ≤ 5 s after a shove | 0.067 / 0.243, 0.144 / 0.440 m | 0.091 / 0.369, 0.213 / 1.14 m |

The policy's lidar frames match analytic mount-cut rays on the true poses in 97.7% (360°) and 97.0% (240°) of the T-hit rays, so the weak ground-truth rows are the policy's behaviour in Gazebo, not its input.
Running the two AMCL cells at once (8 simulations) starved some localizers, which never processed a scan; the episode manager now refuses to start until every robot has a pose estimate and re-sends initial poses that did not converge (those runs are kept as `out/L7_amcl_*_overloaded` and not reported).
