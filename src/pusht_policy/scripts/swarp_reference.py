"""swarp's solve rate of a checkpoint, greedy and sampled, scored as ``examples/eval_mappo.py`` does.

The Gazebo results' reference numbers (``docs/results.md``). Host side, swarp's venv::

    cd ../swarp && uv run --no-sync python ../swarp_gz_test/src/pusht_policy/scripts/swarp_reference.py \
        runs/<run>/<ckpt>.pt [n_envs=512] [seed=123]

One un-reset episode of ``max_steps`` per env (solved if the goal is reached at any step);
the sampled arm draws ``squash(loc + scale * eps)`` as the trainer's rollout does.
"""
import sys

import torch

from swarp.rl.ppo import greedy_env_arm, load_checkpoint
from swarp.rl.recipes import SPECS, coerce_scen_kwargs, score

path = sys.argv[1]
n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 512
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 123
device = "cuda" if torch.cuda.is_available() else "cpu"
actor, _, ckpt = load_checkpoint(path, device)
scenario = ckpt["scenario"]
spec = SPECS[scenario]
kw = dict(
    n_agents=ckpt["n_agents"], n_envs=n_envs, steps=ckpt["max_steps"], device=device,
    seed=seed, dt=ckpt["dt"], substeps=ckpt["substeps"],
    scen_kwargs=coerce_scen_kwargs(scenario, [f"{k}={v}" for k, v in ckpt["scen_kwargs"].items()]),
    arms=("policy",),
)


def sampled_arm(env):
    """The trainer's rollout draw: squash(loc + scale * eps), on a dedicated generator."""
    gen = torch.Generator(device=device).manual_seed(seed + 1)

    @torch.no_grad()
    def act(obs):
        loc, scale = actor(obs, env.agent_roles())
        eps = torch.randn(loc.shape, generator=gen, device=loc.device, dtype=loc.dtype)
        return actor.squash(loc + scale * eps)

    return act


for name, arm in (("greedy", greedy_env_arm(actor)), ("sampled", sampled_arm)):
    m = score(scenario, policy_from_env=arm, **kw)["policy"]
    med = m.solve_step_median
    print(f"{path}: {name:7s} solved {m.solve_rate:.3f} [{m.ci[0]:.3f}, {m.ci[1]:.3f}] over "
          f"{n_envs} envs, median solve step {med} ({med * ckpt['dt'] if med else float('nan'):.1f} s)")
