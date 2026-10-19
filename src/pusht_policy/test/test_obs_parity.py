"""``pusht_policy.obs`` against swarp's own ``PushTLidarScenario``, and the scan resampler
against analytic scans.

The parity tests need swarp (torch, warp) and skip without it; the scan tests are numpy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from pusht_policy.obs import (
    HIT_AGENT,
    HIT_MOVABLE,
    HIT_NONE,
    ObsBuilder,
    ObsConfig,
    label_rays,
    ray_angles,
    scan_to_rays,
)

# ------------------------------------------------------------------- swarp parity

U_H1 = dict(
    n_agents=2,
    n_pushers=2,
    model="diff_drive",
    agent_shape="box",
    obs_level=1,
    frame_stack=4,
    lidar_max_range=0.3,
)
# shape=None: before swarp 4ce60c8, PushTLidarScenario's default shape="tee" built the
# default-size T and ignored the bar_*/stem_* sizes; None honours them on every checkout.
LIMO = dict(
    U_H1,
    world_size=2.0,
    agent_half_extents=(0.161, 0.110),
    max_speed=0.4,
    max_ang_vel=1.5,
    shape=None,
    bar_len=1.0,
    bar_width=0.3,
    stem_len=0.7,
    stem_width=0.3,
    lidar_max_range=1.5,
)
# Observers, a disc fleet and world-axis rays: the role sign, teammate order and the
# holonomic layout (no heading block).
MIXED = dict(n_agents=3, n_pushers=1, obs_level=1, frame_stack=3, world_size=0.5)


def _config(kw: dict) -> ObsConfig:
    names = ObsConfig.__dataclass_fields__.keys()
    return ObsConfig(**{k: v for k, v in kw.items() if k in names})


def _env(kw: dict, fused: bool, n_envs: int):
    torch = pytest.importorskip("torch")
    swarp = pytest.importorskip("swarp")
    del torch
    scen = swarp.PushTLidarScenario(**kw)
    env = swarp.Environment(
        scen, n_envs=n_envs, device="cpu", dt=0.05, substeps=8, seed=0,
        auto_reset=False, fused=fused,
    )
    return env, scen


def _snapshot(scen, fused: bool) -> dict[str, np.ndarray]:
    s = scen.world.state
    if fused:
        ranges, cls = scen.lidar.ranges, scen.lidar.hit_class
    else:
        ranges, cls = scen._lcache["ranges"], scen._lcache["hit_class"]
    return {
        "pos": s.pos.cpu().numpy(),
        "vel": s.vel.cpu().numpy(),
        "theta": s.theta.cpu().numpy(),
        "roles": scen.agent_roles().cpu().numpy(),
        "goal_pos": scen.goal_pos.cpu().numpy(),
        "goal_theta": scen.goal_theta.cpu().numpy(),
        "ranges": ranges.cpu().numpy(),
        "cls": cls.cpu().numpy(),
    }


def _build(builders, snap, envs) -> np.ndarray:
    out = []
    for e in envs:
        out.append(
            [
                b.build(
                    snap["pos"][e], snap["vel"][e], snap["theta"][e], snap["roles"][e],
                    snap["goal_pos"][e], float(snap["goal_theta"][e]),
                    snap["ranges"][e, b.agent], snap["cls"][e, b.agent],
                )
                for b in builders[e]
            ]
        )
    return np.asarray(out)


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "torch"])
@pytest.mark.parametrize("kw", [U_H1, LIMO, MIXED], ids=["u_h1", "limo", "mixed"])
def test_obs_matches_swarp(kw, fused):
    torch = pytest.importorskip("torch")
    n_envs = 4
    env, scen = _env(kw, fused, n_envs)
    cfg = _config(kw)
    assert cfg.obs_dim == scen.obs_dim
    assert cfg.layout() == {k: v for k, v in scen.obs_layout().items() if k in cfg.layout()}
    assert cfg.max_range == pytest.approx(scen.lidar_max_range)
    assert cfg.body_frame == scen.lidar_body_frame
    builders = [[ObsBuilder(cfg, a) for a in range(cfg.n_agents)] for _ in range(n_envs)]
    gen = torch.Generator().manual_seed(1)
    everyone = range(n_envs)
    lid = cfg.layout()["lidar"]
    seen = {"ring_full": False, "hits": set()}

    def check(obs, envs, tag):
        ref = obs.cpu().numpy()[list(envs)]
        mine = _build(builders, _snapshot(scen, fused), envs)
        np.testing.assert_allclose(mine, ref, atol=1e-5, rtol=0, err_msg=tag)
        seen["ring_full"] |= bool((ref[..., lid.start : lid.start + 3 * cfg.n_rays] != 0).any())
        seen["hits"] |= set(np.unique(_snapshot(scen, fused)["cls"]).tolist())

    def act():
        return torch.empty(n_envs, cfg.n_agents, 2).uniform_(-1, 1, generator=gen)

    with torch.no_grad():
        check(env.reset(seed=0), everyone, "reset")
        for i in range(10):
            check(env.step(act())[0], everyone, f"step {i}")
        # A masked reset: the selected envs restart their ring, the others are untouched.
        mask = torch.tensor([True, False, True, False])
        prev = env.scenario.observations().cpu().numpy().copy()
        for e in np.flatnonzero(mask.numpy()):
            for b in builders[e]:
                b.reset()
        obs = env.reset_at(mask)
        check(obs, np.flatnonzero(mask.numpy()), "reset_at")
        np.testing.assert_array_equal(obs.cpu().numpy()[~mask.numpy()], prev[~mask.numpy()])
        for i in range(6):
            check(env.step(act())[0], everyone, f"after reset_at {i}")
        for row in builders:
            for b in row:
                b.reset()
        check(env.reset(), everyone, "second reset")
        for i in range(4):
            check(env.step(act())[0], everyone, f"after reset {i}")
    # The oldest frame was filled (the ring really shifted) and the scans saw something.
    assert seen["ring_full"]
    assert seen["hits"] - {HIT_NONE}, "no ray hit anything: the parity says nothing"


def test_config_from_checkpoint_meta():
    """U-H1's stored metadata (``scen_kwargs`` are strings) rebuilds its 398-wide obs."""
    meta = {
        "scenario": "pusht_lidar",
        "obs_dim": 398,
        "n_agents": 2,
        "role_counts": [2, 0],
        "scen_kwargs": {
            "pos_shaping_factor": "5.0", "rot_shaping_factor": "0.5", "rot_away_penalty": "1.0",
            "obs_level": "1", "model": "diff_drive", "agent_shape": "box", "frame_stack": "4",
            "lidar_max_range": "0.3", "n_pushers": "2",
        },
    }
    cfg = ObsConfig.from_checkpoint(meta)
    assert (cfg.n_agents, cfg.n_pushers, cfg.frame_stack, cfg.max_range) == (2, 2, 4, 0.3)
    assert cfg.heading_obs and cfg.body_frame
    assert cfg.half_extents == (0.025, 0.0125)
    limo = dict(meta, obs_dim=398, scenario="pusht_lidar_limo")
    limo["scen_kwargs"] = dict(
        meta["scen_kwargs"], world_size="2.0", agent_half_extents="(0.161, 0.11)"
    )
    assert ObsConfig.from_checkpoint(limo).half_extents == (0.161, 0.11)
    with pytest.raises(ValueError, match="not a pusht_lidar"):
        ObsConfig.from_checkpoint(dict(meta, scenario="pusht"))
    with pytest.raises(ValueError, match="obs_dim"):
        ObsConfig.from_checkpoint(dict(meta, obs_dim=117))
    with pytest.raises(NotImplementedError):
        ObsConfig.from_checkpoint(dict(meta, scen_kwargs={"obs_level": "2"}))


# ------------------------------------------------------------------ scan resampling


def _cast(origin, angles, boxes, miss=np.inf):
    """Exact thin-ray ranges and box indices from ``origin`` against oriented boxes
    ``(cx, cy, theta, hx, hy)``; -1 where nothing is hit."""
    best = np.full(len(angles), miss)
    idx = np.full(len(angles), -1)
    for k, (cx, cy, th, hx, hy) in enumerate(boxes):
        c, s = math.cos(th), math.sin(th)
        rx, ry = origin[0] - cx, origin[1] - cy
        ox, oy = c * rx + s * ry, -s * rx + c * ry
        dx = np.cos(angles) * c + np.sin(angles) * s
        dy = -np.cos(angles) * s + np.sin(angles) * c
        with np.errstate(divide="ignore", invalid="ignore"):
            t1x, t2x = (-hx - ox) / dx, (hx - ox) / dx
            t1y, t2y = (-hy - oy) / dy, (hy - oy) / dy
        tn = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
        tf = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
        t = np.where((tn <= tf) & (tn > 0), tn, np.inf)
        closer = t < best
        best, idx = np.where(closer, t, best), np.where(closer, k, idx)
    return best, idx


MOUNT = (0.103, 0.0)
LIMO_HALF = (0.161, 0.110)


def _scan(pose, boxes, n=360, noise=0.0, seed=0):
    """A full-circle LaserScan from the lidar mounted at ``MOUNT`` on a robot at ``pose``."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    origin = (x + c * MOUNT[0] - s * MOUNT[1], y + s * MOUNT[0] + c * MOUNT[1])
    angle_min, inc = -math.pi, 2.0 * math.pi / n
    r, _ = _cast(origin, th + angle_min + inc * np.arange(n), boxes)
    r = r + np.random.default_rng(seed).normal(0.0, noise, n)
    return dict(ranges=r, angle_min=angle_min, angle_increment=inc, range_min=0.1, range_max=8.0)


def _resample(scan, pose, n_rays, max_range, **kw):
    return scan_to_rays(
        scan["ranges"], scan["angle_min"], scan["angle_increment"], scan["range_min"],
        scan["range_max"], MOUNT, 0.0, n_rays, max_range, pose=pose, own_half=LIMO_HALF, **kw,
    )


def test_scan_to_rays_box_and_teammate():
    """A T-like box ahead and a teammate to the left: the resampled centre rays match an
    exact centre-ray cast, and each hit is labelled by what it hit."""
    pose = (-0.3, 0.2, 0.4)
    tee = (0.5, 0.6, 0.3, 0.5, 0.15)
    mate = (-0.6, 0.9, -0.2, *LIMO_HALF)
    max_range, ws = 1.5, 2.0
    scan = _scan(pose, [tee, mate])
    ranges, pts = _resample(scan, pose, 32, max_range)
    r, cls = label_rays(ranges, pose, max_range, ws, [mate[:2]], [mate[2]], LIMO_HALF)

    phi = pose[2] + ray_angles(32)
    exp_r, exp_idx = _cast(pose[:2], phi, [tee, mate])
    exp_cls = np.select([exp_idx == 0, exp_idx == 1], [HIT_MOVABLE, HIT_AGENT], HIT_NONE)
    exp_cls = np.where(exp_r < max_range, exp_cls, HIT_NONE)
    exp_r = np.minimum(exp_r, max_range)
    np.testing.assert_array_equal(cls, exp_cls)
    assert {HIT_MOVABLE, HIT_AGENT, HIT_NONE} <= set(cls.tolist())
    # Chords between 1-degree samples sit inside a convex face; the error is a few mm.
    np.testing.assert_allclose(r, exp_r, atol=5e-3)
    assert pts.shape == (int(np.isfinite(scan["ranges"]).sum()), 2)


@pytest.mark.parametrize("side", [1, -1], ids=["left", "rear"])
def test_scan_to_rays_keeps_a_touching_t(side):
    """A face 5 mm from the robot's side (or rear), seen through 1 cm range noise, stays a
    hit on the rays that point at it: noisy returns just inside the footprint are kept."""
    pose = (0.0, 0.0, 0.0)
    if side == 1:  # wall along x, 5 mm left of the side face
        wall, ray = (0.0, LIMO_HALF[1] + 0.005 + 0.1, 0.0, 0.5, 0.1), 8
    else:  # wall along y, 5 mm behind the rear face
        wall, ray = (-LIMO_HALF[0] - 0.005 - 0.1, 0.0, 0.0, 0.1, 0.5), 16
    gap = (LIMO_HALF[1] if side == 1 else LIMO_HALF[0]) + 0.005
    for seed in range(5):
        scan = _scan(pose, [wall], noise=0.01, seed=seed)
        ranges, _ = _resample(scan, pose, 32, 1.5)
        assert abs(ranges[ray] - gap) < 0.03, (seed, ranges[ray])


def test_scan_points_outside_arena_are_misses():
    """A box beyond the arena square is invisible (swarp's walls are), one inside is not."""
    pose = (1.5, 0.0, 0.0)
    ws = 2.0
    outside = (2.6, 0.0, 0.0, 0.2, 0.6)  # straight ahead, past x = world_size
    inside = (1.5, -0.8, 0.0, 0.3, 0.1)  # to the right, inside the arena
    scan = _scan(pose, [outside, inside])
    ranges, _ = _resample(scan, pose, 32, 1.5)
    assert ranges[0] < 1.5, "the outside box is within range: it must be dropped, not missed"
    r, cls = label_rays(ranges, pose, 1.5, ws, np.zeros((0, 2)), np.zeros(0), LIMO_HALF)
    assert cls[0] == HIT_NONE and r[0] == 1.5
    assert set(cls.tolist()) <= {HIT_NONE, HIT_MOVABLE}
    right = 24  # -90 degrees
    assert cls[right] == HIT_MOVABLE and r[right] == pytest.approx(0.7, abs=5e-3)
