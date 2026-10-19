"""Solve rate and failure modes of a Gazebo eval run recorded with ``record:=true``.

    python3 -m pusht_policy.failure_modes out/<run> [out/<run> ...]

Per run: the solve rate (Wilson 95%), solve times, the control-step spacing, and each
failure classed by the T's final state (checked in this order):

* ``left_arena``: the T's centroid went past ``+-world_size`` at some step (Gazebo has no
  walls; swarp keeps the whole T inside the arena);
* ``near_goal_wrong_angle``: final ``|p - g| < 2 goal_tolerance`` with the angle out of
  tolerance;
* ``near_goal_pos_off``: final ``|p - g| < 2 goal_tolerance`` with the angle in tolerance;
* ``tee_far``: anything else.

Plus behaviour measured on the recordings:

* stalled: share of steps with ``|v_cmd| < 0.04`` m/s;
* yaw-locked: a robot not turning (``|dyaw| < 1e-3`` per step) while commanding
  ``|w| > 0.3`` rad/s, with its yaw equal to the T's modulo 90 degrees (flush against a
  face);
* squeeze lock: both robots yaw-locked at once and facing each other, i.e. pushing into
  opposite faces of the T, which Gazebo (and real robots) cannot turn out of;
* ``past_clamp``: episodes whose T centroid passed ``world_size - tee_radius``, which
  swarp's body clamp never lets it reach.

numpy only; the task constants come from the recorded ``dt`` and the arguments.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics

import numpy as np

from pusht_policy.aggregate import wilson

STALL_V = 0.04


def _wrap(x):
    return np.angle(np.exp(1j * np.asarray(x, dtype=float)))


def yaw_lock(rec: np.lib.npyio.NpzFile) -> np.ndarray:
    """``[T - 1]`` bool: the recorded robot is yaw-locked flush against the T."""
    a = int(rec["agent"])
    yaw = rec["pose"][:, a, 2].astype(float)
    rel = _wrap(4.0 * (yaw - rec["tee"][:, 2].astype(float))) / 4.0
    still = np.abs(_wrap(np.diff(yaw))) < 1e-3
    return (np.abs(rel[1:]) < 0.005) & still & (np.abs(rec["act"][1:, 1]) > 0.3)


def analyse(run: str, world_size: float, goal_tol: float, ang_tol: float, tee_radius: float) -> dict:
    rows = []
    for f in sorted(glob.glob(os.path.join(run, "**", "episodes.csv"), recursive=True)):
        shard = os.path.dirname(f)
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                pattern = os.path.join(shard, "rec", f"ep{int(r['episode']):04d}_robot_*.npz")
                r["recs"] = [np.load(p) for p in sorted(glob.glob(pattern))]
                rows.append(r)
    if not rows:
        raise SystemExit(f"no episodes under {run}")
    n = len(rows)
    k = sum(int(r["success"]) for r in rows)
    out = {"n": n, "k": k, "ci": wilson(k, n)}
    out["solve_t"] = [float(r["time"]) for r in rows if int(r["success"])]
    cats = {c: 0 for c in ("left_arena", "near_goal_wrong_angle", "near_goal_pos_off", "tee_far")}
    dts, stall, robot_lock, squeeze, past_clamp = [], [], [], [], 0
    for r in rows:
        recs = r["recs"]
        left = False
        if recs:
            tee = recs[0]["tee"][:, :2].astype(float)
            reach = np.abs(tee).max(1)
            left = bool((reach > world_size).any())
            past_clamp += bool((reach > world_size - tee_radius + 0.02).any())
            stall.append(float(np.mean(np.concatenate([np.abs(x["act"][:, 0]) < STALL_V for x in recs]))))
            locks = []
            for x in recs:
                dts.append(np.diff(x["t"]))
                lk = yaw_lock(x)
                robot_lock.append((lk.mean(), lk.sum() * float(x["dt"]) > 10.0))
                locks.append(lk)
            if len(recs) == 2:
                m = min(len(locks[0]), len(locks[1]))
                pose = recs[0]["pose"][1 : m + 1].astype(float)
                facing = np.abs(_wrap(pose[:, 0, 2] - pose[:, 1, 2] - math.pi)) < 0.1
                sq = locks[0][:m] & locks[1][:m] & facing
                squeeze.append((sq.mean(), sq.sum() * float(recs[0]["dt"]) > 10.0))
        if int(r["success"]):
            continue
        pe, ae = float(r["final_pos_err"]), float(r["final_ang_err"])
        if left:
            cats["left_arena"] += 1
        elif pe < 2 * goal_tol:
            cats["near_goal_wrong_angle" if ae >= ang_tol else "near_goal_pos_off"] += 1
        else:
            cats["tee_far"] += 1
    out.update(cats=cats, past_clamp=past_clamp, n_rec=sum(bool(r["recs"]) for r in rows))
    if dts:
        d = np.concatenate(dts)
        out["late_steps"] = float(np.mean(d > 1.25 * np.median(d)))
    out["stall"] = float(np.median(stall)) if stall else math.nan
    if robot_lock:
        lk = np.array(robot_lock)
        out["robot_lock"] = (float(lk[:, 0].mean()), int(lk[:, 1].sum()), len(lk))
    if squeeze:
        sq = np.array(squeeze)
        out["squeeze"] = (float(sq[:, 0].mean()), int(sq[:, 1].sum()), len(sq))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories (searched for episodes.csv)")
    ap.add_argument("--world-size", type=float, default=2.0)
    ap.add_argument("--goal-tolerance", type=float, default=0.15)
    ap.add_argument("--angle-tolerance", type=float, default=0.25)
    ap.add_argument("--tee-radius", type=float, default=0.6614)
    args = ap.parse_args(argv)
    for run in args.runs:
        o = analyse(run, args.world_size, args.goal_tolerance, args.angle_tolerance, args.tee_radius)
        lo, hi = o["ci"]
        print(f"== {run}: solved {o['k']}/{o['n']} = {o['k'] / o['n']:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
        if o["solve_t"]:
            print(f"   solve time median {statistics.median(o['solve_t']):.1f} s")
        print("   failures: " + ", ".join(f"{c} {v}" for c, v in o["cats"].items()))
        if "late_steps" in o:
            print(f"   control steps late by > 25%: {o['late_steps']:.4f}")
        print(f"   stalled share (median episode) {o['stall']:.3f}; T past swarp's clamp in "
              f"{o['past_clamp']}/{o['n_rec']} episodes")
        if "robot_lock" in o:
            s, c, m = o["robot_lock"]
            print(f"   yaw-locked: {s:.3f} of robot-steps, {c}/{m} robot-episodes > 10 s")
        if "squeeze" in o:
            s, c, m = o["squeeze"]
            print(f"   squeeze lock: {s:.3f} of episode-steps, {c}/{m} episodes > 10 s")


if __name__ == "__main__":
    main()
