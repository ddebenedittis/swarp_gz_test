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

```bash
src/pusht_policy/scripts/eval_shards.bash /opt/swarp/runs/remote-davide-cf/L5s-wheelstall-limo-s0-fromL5r/pusht_lidar_limo_dr_wheel_best.pt 2 10 L5s_gt_landmarks 0 landmarks:=true record:=true
BUILD=0 DOMAIN_BASE=80 src/pusht_policy/scripts/eval_shards.bash /opt/swarp/runs/remote-davide-cf/L5s-wheelstall-limo-s0-fromL5r/pusht_lidar_limo_dr_wheel_best.pt 2 10 L5s_amcl 0 pose_source:=amcl record:=true
ros2 run limo_real pose_error_merge out/L5s_amcl/shard0/pose_error out/L5s_amcl/shard1/pose_error
```
