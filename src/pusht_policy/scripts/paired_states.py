"""One CSV of the initial states several Gazebo runs share, with each run's outcome, for
paired comparisons (e.g. swarp episodes started from the same states).

    python3 src/pusht_policy/scripts/paired_states.py out/paired.csv out/<run> [out/<run> ...]

A run is a directory with ``episodes.csv`` or ``shard*/episodes.csv``. Rows are keyed by
``(seed, episode)``; the state columns (T, goal, robot poses; the T at its area centroid,
yaws in rad, arena frame) come from the first run and must agree across runs. Per run:
``<run>_success``, ``<run>_time``, ``<run>_final_pos_err``, ``<run>_final_ang_err``.
Stdlib only.
"""

from __future__ import annotations

import csv
import glob
import os
import sys

STATE = ["tee_x", "tee_y", "tee_yaw", "goal_x", "goal_y", "goal_yaw",
         "robot_1_x", "robot_1_y", "robot_1_yaw", "robot_2_x", "robot_2_y", "robot_2_yaw"]
OUTCOME = ["success", "time", "final_pos_err", "final_ang_err"]


def _rows(run: str) -> dict[tuple[int, int], dict]:
    files = sorted(glob.glob(os.path.join(run, "episodes.csv")) + glob.glob(os.path.join(run, "shard*", "episodes.csv")))
    out = {}
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                out[(int(r["seed"]), int(r["episode"]))] = r
    return out


def main() -> None:
    dst, runs = sys.argv[1], sys.argv[2:]
    data = {os.path.basename(os.path.normpath(r)): _rows(r) for r in runs}
    first = next(iter(data.values()))
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    for name, d in data.items():
        for k in keys:
            if any(abs(float(d[k][c]) - float(first[k][c])) > 1e-3 for c in STATE):
                raise SystemExit(f"{name} {k}: initial state differs from the first run")
    with open(dst, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "episode", *STATE, *(f"{n}_{c}" for n in data for c in OUTCOME)])
        for k in keys:
            w.writerow([*k, *(first[k][c] for c in STATE), *(data[n][k][c] for n in data for c in OUTCOME)])
    print(f"{len(keys)} states x {len(data)} runs -> {dst}")


if __name__ == "__main__":
    main()
