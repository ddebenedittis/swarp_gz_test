"""The sysid experiments in swarp, and swarp's greedy policy from recorded Gazebo starts.

Runs in swarp's own venv on the host and imports swarp read-only::

    cd ../swarp && uv run --no-sync python ../swarp_gz_test/src/pusht_policy/scripts/sysid_swarp.py \
        --checkpoint runs/<run>/<ckpt>.pt --out ../swarp_gz_test/out/sysid \
        [--paired ../swarp_gz_test/out/<eval run>]

The scenario is built from the checkpoint's ``scen_kwargs`` (the geometry, contact law and
tolerances it trained with). Scripted experiments step at ``experiments.DT`` with the same
6.25 ms contact sub-step as the recipes (``--substeps``); the paired policy rollouts use the
checkpoint's own ``dt``/``substeps``.

State is set through a ``PushTLidarScenario`` subclass whose ``_launch_spawn`` (the reset
hook every ``reset_world`` goes through) overwrites the random draw in place with a pinned
state, so the fused or torch refresh that follows (lidar ring, caches) sees it.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # src/pusht_policy, for pusht_policy.sysid
from pusht_policy.sysid.experiments import DT, Experiment, all_experiments  # noqa: E402

from swarp import Environment  # noqa: E402
from swarp.rl.ppo import load_checkpoint  # noqa: E402
from swarp.scenarios.pusht_lidar import PushTLidarScenario  # noqa: E402


def parse(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


class Pinned(PushTLidarScenario):
    """``PushTLidarScenario`` whose reset lands on ``self.pin`` (per-env tensors)."""

    pin: dict | None = None

    def _launch_spawn(self, mask, use_mask) -> None:
        super()._launch_spawn(mask, use_mask)
        if self.pin is None:
            return
        p = self.pin
        with torch.no_grad():
            self.tee_pos.copy_(p["tee"][:, :2])
            self.tee_theta.copy_(p["tee"][:, 2])
            self.tee_vel.zero_()
            self.tee_ang_vel.zero_()
            self.goal_pos.copy_(p["goal"][:, :2])
            self.goal_theta.copy_(p["goal"][:, 2])
            r = p["robots"]
            self.world.write_state(
                pos=r[..., :2], theta=r[..., 2], vel=torch.zeros_like(r[..., :2]), speed=0.0, ang_vel=0.0
            )


def make_env(kw, n_envs, dt, substeps, pin, n_agents=2):
    scen = Pinned(n_agents=n_agents, **kw)
    env = Environment(scen, n_envs=n_envs, device="cpu", dt=dt, substeps=substeps, seed=0,
                      auto_reset=False, fused=False)
    scen.pin = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in pin.items()}
    env.reset()
    return env, scen


def snapshot(scen):
    s = scen.world.state
    return {
        "robots": torch.cat([s.pos, s.theta.unsqueeze(-1)], -1).numpy().copy(),
        "robots_vel": s.vel.numpy().copy(),
        "robots_w": s.ang_vel.numpy().copy(),
        "tee": torch.cat([scen.tee_pos, scen.tee_theta.unsqueeze(-1)], -1).numpy().copy(),
        "tee_vel": scen.tee_vel.numpy().copy(),
        "tee_w": scen.tee_ang_vel.numpy().copy(),
    }


def run_scripted(kw, exps: list[Experiment], substeps: int, out: str) -> None:
    """All experiments at once, one env each, padded with zero commands to the longest."""
    n = len(exps)
    steps = max(len(e.cmds) for e in exps)
    pin = {
        "robots": [e.robots for e in exps],
        "tee": [e.tee for e in exps],
        "goal": [(1.5, -1.5, 0.0)] * n,
    }
    env, scen = make_env(kw, n, DT, substeps, pin)
    lid = {"ranges": scen._lcache["ranges"].numpy().copy(), "cls": scen._lcache["hit_class"].numpy().copy()}
    traj = [snapshot(scen)]
    cmds = np.zeros((steps, n, 2, 2), dtype=np.float32)
    for i, e in enumerate(exps):
        cmds[: len(e.cmds), i] = e.cmds
    for k in range(steps):
        env.step(torch.from_numpy(cmds[k]))
        traj.append(snapshot(scen))
    for i, e in enumerate(exps):
        m = len(e.cmds)
        rec = {key: np.stack([t[key][i] for t in traj[: m + 1]]) for key in traj[0]}
        np.savez(
            os.path.join(out, f"swarp_{e.name}.npz"),
            t=np.arange(m + 1) * DT, cmds=e.cmds, rays=lid["ranges"][i, 0], cls=lid["cls"][i, 0], **rec,
        )
    print(f"swarp: {n} scripted experiments -> {out}")


def run_paired(ckpt_path: str, kw, meta, eval_dir: str, out: str) -> None:
    """swarp's greedy policy from each recorded Gazebo episode's initial state."""
    rows = []
    for f in sorted(glob.glob(os.path.join(eval_dir, "**", "episodes.csv"), recursive=True)):
        with open(f, newline="") as fh:
            rows += [dict(r, _dir=os.path.dirname(f)) for r in csv.DictReader(fh)]
    if not rows:
        raise SystemExit(f"no episodes.csv under {eval_dir}")
    n_agents = int(meta["n_agents"])
    robots = [f"robot_{j}" for j in range(1, n_agents + 1)]
    f3 = lambda r, b: tuple(float(r[f"{b}_{c}"]) for c in ("x", "y", "yaw"))  # noqa: E731
    pin = {
        "robots": [[f3(r, b) for b in robots] for r in rows],
        "tee": [f3(r, "tee") for r in rows],
        "goal": [f3(r, "goal") for r in rows],
    }
    actor, _, _ = load_checkpoint(ckpt_path, "cpu")
    env, scen = make_env(kw, len(rows), float(meta["dt"]), int(meta["substeps"]), pin, n_agents)
    roles = scen.agent_roles()
    obs = scen.observations()
    steps = int(meta["max_steps"])
    tee = [snapshot(scen)["tee"]]
    rob = [snapshot(scen)["robots"]]
    solved_at = np.full(len(rows), -1)
    for k in range(steps):
        with torch.no_grad():
            act = actor.greedy(obs, roles)
        obs, _, term, _, _ = env.step(act)
        s = snapshot(scen)
        tee.append(s["tee"])
        rob.append(s["robots"])
        newly = term.numpy() & (solved_at < 0)
        solved_at[newly] = k + 1
    tee = np.stack(tee, 1)  # [E, T, 3]
    rob = np.stack(rob, 1)
    np.savez(
        os.path.join(out, "swarp_paired.npz"),
        tee=tee, robots=rob, solved_at=solved_at, dt=float(meta["dt"]),
        seed=np.array([int(r["seed"]) for r in rows]), episode=np.array([int(r["episode"]) for r in rows]),
        run_dir=np.array([r["_dir"] for r in rows]),
    )
    print(f"swarp: {len(rows)} paired greedy rollouts, solved {int((solved_at > 0).sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--substeps", type=int, default=16, help="per DT=0.1 step (6.25 ms, as both recipes)")
    ap.add_argument("--paired", help="Gazebo eval run dir (episodes.csv) to replay with swarp's policy")
    ap.add_argument("--only", choices=("scripted", "paired"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    kw = {k: parse(v) for k, v in meta["scen_kwargs"].items()}
    if args.only != "paired":
        run_scripted(kw, all_experiments(), args.substeps, args.out)
    if args.paired and args.only != "scripted":
        run_paired(args.checkpoint, kw, meta, args.paired, args.out)


if __name__ == "__main__":
    main()
