"""The delayed messages of :mod:`pusht_policy.comm` against swarp's own greedy rollout of a
``--comm pose`` actor (``swarp.rl.ppo.greedy_env_arm``, what ``eval_mappo.py`` scores).

Needs swarp (torch, warp); skips without it.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

# Two Limos, one pusher and one observer, with the on-goal flag in the aux targets: the
# (1, 1) fine-tune's message is 2 * 5 = 10 floats.
KW = dict(
    n_agents=2, n_pushers=1, model="diff_drive", agent_shape="box", obs_level=1, frame_stack=4,
    world_size=2.0, agent_half_extents=(0.161, 0.110), max_speed=0.4, max_ang_vel=1.5,
    lidar_max_range=1.5, aux_goal=True,
)
DT = 0.2
T0 = 1.7e9  # an epoch-like stamp: the payload's split keeps it to the millisecond


def test_messages_match_swarp_rollout():
    """Two :class:`Messenger` s exchanging payloads act as ``greedy_env_arm`` does on the
    same observations (no message on step 0, every one heard after); a missing or
    out-of-step message is not heard."""
    torch = pytest.importorskip("torch")
    swarp = pytest.importorskip("swarp")
    from swarp.rl.ppo import build_arch, check_action_box, greedy_env_arm, network_archs
    from swarp.rl.recipes import SPECS

    from pusht_policy.comm import Messenger
    from pusht_policy.obs import pruned_kwargs

    pruned = pruned_kwargs()
    scen = swarp.PushTLidarScenario(**{k: v for k, v in KW.items() if k not in pruned})
    env = swarp.Environment(scen, n_envs=1, device="cpu", dt=0.05, substeps=8, seed=0,
                            auto_reset=False, fused=False)
    spec = replace(SPECS["pusht_lidar_limo"], comm="pose")
    torch.manual_seed(0)
    name, kwargs = network_archs(spec, env, *check_action_box(env, "pusht_lidar_limo"))[:2]
    actor = build_arch(name, kwargs).eval()
    assert actor.comm == "pose" and actor.msg_dim == 10

    gen = torch.Generator().manual_seed(1)
    obs, ref = [], []
    with torch.no_grad():
        o = env.reset(seed=0)
        roles = env.agent_roles()[0].clone()
        policy = greedy_env_arm(actor)(env)
        for _ in range(12):
            a = policy(o)
            obs.append(o.clone())
            ref.append(a.clone())
            # Some noise on top, so the robots move and the observations change.
            o = env.step(a + 0.3 * torch.randn(a.shape, generator=gen))[0]
    assert sorted(roles.tolist()) == [0, 1]

    ep = 7
    nodes = [Messenger(actor, roles[i : i + 1], [1 - i], DT) for i in range(2)]
    deaf = 0.0
    for t, (o, a) in enumerate(zip(obs, ref)):
        rows = [o[0, i : i + 1] for i in range(2)]
        for i, node in enumerate(nodes):
            heard = node.heard(ep, T0 + t * DT)
            assert heard[0, -1] == (t > 0)
            loc = node.step(rows[i], heard)[0]
            torch.testing.assert_close(actor.squash(loc)[0], a[0, i], atol=1e-5, rtol=0,
                                       msg=lambda m, t=t, i=i: f"step {t} robot {i}: {m}")
            silent = actor.greedy(rows[i], roles[i : i + 1])[0]
            deaf = max(deaf, float((silent - a[0, i]).abs().max())) if t else deaf
        for i in range(2):
            nodes[1 - i].receive(i, nodes[i].payload(ep, T0 + t * DT, rows[i]))
    assert deaf > 1e-3, "the messages never changed an action: the parity says nothing"


def _sent(ep: int, stamp: float, value: float) -> list[float]:
    from pusht_policy.comm import split_stamp

    return [float(ep), *split_stamp(stamp), value, -value]


def test_message_window():
    """A teammate's message is heard on the step ``dt`` after it, give or take half a
    ``dt``: from a teammate a scan out of phase every step, and never two steps old when
    the grids drift a whole step apart."""
    pytest.importorskip("torch")
    from pusht_policy.comm import Messenger

    node = Messenger(SimpleNamespace(delayed=True, msg_dim=2), None, [1], DT)
    ep, t = 3, T0 + 50.0

    def heard(at, e=ep):
        h = node.heard(e, at)[0].numpy()
        return float(h[0]) if h[-1] else None

    node.receive(1, _sent(ep, t - DT, 1.0))
    assert heard(t) == 1.0
    assert heard(t, ep + 1) is None
    assert heard(t + DT) is None and heard(t - DT) is None  # two steps old, or not yet sent
    node.receive(1, _sent(ep, t - DT, 1.0)[:-1])  # another checkpoint's width: dropped
    assert len(node.inbox[1].msgs) == 1

    # A teammate one scan (0.1 s) behind: each step hears its latest, and it hears ours.
    for k in range(5):
        node.receive(1, _sent(ep, t + k * DT + 0.1, 10.0 + k))
        assert heard(t + (k + 1) * DT) == 10.0 + k
    mate = Messenger(SimpleNamespace(delayed=True, msg_dim=2), None, [0], DT)
    for k in range(5):
        mate.receive(0, _sent(ep, t + k * DT, 20.0 + k))
        assert mate.heard(ep, t + k * DT + 0.1)[0, 0] == 20.0 + k

    # A full dt of drift: the teammate's step lands on ours. Its coincident message is
    # not the previous step; its previous one is, and with that lost nothing is heard.
    s = t + 100.0
    node.receive(1, _sent(ep, s - 2 * DT, 30.0))
    node.receive(1, _sent(ep, s, 32.0))
    assert heard(s) is None
    node.receive(1, _sent(ep, s - DT, 31.0))
    assert heard(s) == 31.0

