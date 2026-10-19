"""Solve rate (Wilson 95% interval) over one or more ``episodes.csv`` files.

    python3 -m pusht_policy.aggregate out/<run>            # every episodes.csv below it
    ros2 run pusht_policy aggregate out/a/episodes.csv out/b/episodes.csv

Standard library only, so it also runs on the host.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def load(paths: list[str]) -> list[dict]:
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "**", "episodes.csv"), recursive=True)) if os.path.isdir(p) else [p]
    rows = []
    for f in files:
        with open(f, newline="") as fh:
            rows += [dict(r, _file=f) for r in csv.DictReader(fh)]
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="episodes.csv files or directories to search")
    args = ap.parse_args(argv)
    rows = load(args.paths)
    if not rows:
        raise SystemExit("no episodes found")
    n = len(rows)
    k = sum(int(r["success"]) for r in rows)
    lo, hi = wilson(k, n)
    files = sorted({r["_file"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    print(f"{len(files)} file(s), seeds {seeds}: solved {k}/{n} = {k / n:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")
    solved_t = [float(r["time"]) for r in rows if int(r["success"])]
    if solved_t:
        print(f"time to solve: median {statistics.median(solved_t):.1f}s, mean {statistics.mean(solved_t):.1f}s")
    fails = [r for r in rows if not int(r["success"])]
    if fails:
        print(
            "failures: median final error "
            f"{statistics.median(float(r['final_pos_err']) for r in fails):.3f} m / "
            f"{statistics.median(float(r['final_ang_err']) for r in fails):.3f} rad, median min error "
            f"{statistics.median(float(r['min_pos_err']) for r in fails):.3f} m"
        )


if __name__ == "__main__":
    main()
