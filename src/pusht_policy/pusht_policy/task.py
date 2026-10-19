"""The Push-T task a checkpoint was trained on, and swarp's episode reset, in numpy.

:class:`Task` reads the geometry, tolerances and episode length off the checkpoint by
building the very ``PushTLidarScenario`` the run trained (no environment, no GPU), so the
T radius, the agent bounding radius and every default match swarp by construction.

:func:`sample_episode` reproduces ``pusht_reset_kernel`` + ``pusht_heading_reset_kernel``
at full difficulty (``goal_spawn_radius = goal_spawn_angle = None``), with one Gazebo-only
addition: swarp lets spawns interpenetrate (its contacts push them apart), Gazebo's rigid
contacts would launch them, so a draw whose footprints overlap is rejected and redrawn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pusht_policy.obs import ObsConfig, _parse


@dataclass(frozen=True)
class Task:
    scenario: str
    n_agents: int
    dt: float
    max_steps: int
    world_size: float
    agent_half: tuple[float, float] | None  # box robots
    agent_radius: float  # swarp's bounding radius (circumscribed for a box)
    tee_half: tuple[tuple[float, float], ...]  # per box, T frame
    tee_off: tuple[tuple[float, float], ...]
    tee_radius: float
    bar_len: float
    bar_width: float
    stem_len: float
    stem_width: float
    goal_tolerance: float
    angle_tolerance: float

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> Task:
        from swarp.scenarios.pusht_lidar import PushTLidarScenario

        kw = {k: _parse(v) for k, v in dict(meta.get("scen_kwargs", {})).items()}
        scen = PushTLidarScenario(n_agents=int(meta["n_agents"]), **kw)
        shape = scen.shape
        cfg = ObsConfig.from_checkpoint(meta)
        half = cfg.half_extents if cfg.agent_shape == "box" else None
        bar, stem = shape.half[0], shape.half[1]
        return cls(
            scenario=str(meta.get("scenario")),
            n_agents=int(meta["n_agents"]),
            dt=float(meta["dt"]),
            max_steps=int(meta["max_steps"]),
            world_size=float(scen.world_size),
            agent_half=half,
            agent_radius=float(scen.agent_radius),
            tee_half=tuple(tuple(map(float, h)) for h in shape.half),
            tee_off=tuple(tuple(map(float, o)) for o in shape.off),
            tee_radius=float(scen.tee_radius),
            bar_len=2 * bar[0], bar_width=2 * bar[1], stem_len=2 * stem[1], stem_width=2 * stem[0],
            goal_tolerance=float(scen.goal_tolerance),
            angle_tolerance=float(scen.angle_tolerance),
        )

    @classmethod
    def from_checkpoint(cls, path: str) -> Task:
        import torch

        return cls.from_meta(torch.load(path, map_location="cpu", weights_only=True))


# ------------------------------------------------------------------------ sampling


def _corners(c, yaw, half) -> np.ndarray:
    cs, sn = math.cos(yaw), math.sin(yaw)
    loc = np.array([[sx * half[0], sy * half[1]] for sx in (-1, 1) for sy in (-1, 1)])
    return loc @ np.array([[cs, sn], [-sn, cs]]) + np.asarray(c)


def boxes_overlap(c1, yaw1, h1, c2, yaw2, h2, margin: float = 0.0) -> bool:
    """Separating-axis test of two oriented rectangles, each grown by ``margin``."""
    h1 = (h1[0] + margin, h1[1] + margin)
    h2 = (h2[0] + margin, h2[1] + margin)
    p1, p2 = _corners(c1, yaw1, h1), _corners(c2, yaw2, h2)
    for yaw in (yaw1, yaw2):
        for ax in ((math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))):
            a, b = p1 @ ax, p2 @ ax
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def tee_boxes(task: Task, x: float, y: float, yaw: float):
    """The T's boxes as ``(centre, yaw, half)`` in the world."""
    cs, sn = math.cos(yaw), math.sin(yaw)
    return [
        ((x + cs * ox - sn * oy, y + sn * ox + cs * oy), yaw, h)
        for (ox, oy), h in zip(task.tee_off, task.tee_half, strict=True)
    ]


def _swarp_draw(task: Task, rng: np.random.Generator) -> dict[str, Any]:
    """One draw of ``pusht_reset_kernel`` + ``pusht_heading_reset_kernel`` (full difficulty)."""
    lim = task.world_size - 2.0 * task.agent_radius
    tlim = task.world_size - task.tee_radius
    clear = task.tee_radius + 2.0 * task.agent_radius
    u = lambda: rng.uniform(-1.0, 1.0)  # noqa: E731
    tx, ty = u() * tlim, u() * tlim
    tth = u() * math.pi
    robots = []
    for _ in range(task.n_agents):
        px, py = u() * lim, u() * lim
        dx, dy = px - tx, py - ty
        dn = math.hypot(dx, dy)
        if dn < clear:
            if dn > 1e-9:
                px, py = tx + dx / dn * clear, ty + dy / dn * clear
            else:
                px, py = tx, ty + clear
        robots.append([min(max(px, -lim), lim), min(max(py, -lim), lim), 0.0])
    gx, gy = u() * tlim, u() * tlim
    gth = u() * math.pi
    for r in robots:
        r[2] = u() * math.pi
    return {"tee": (tx, ty, tth), "goal": (gx, gy, gth), "robots": [tuple(r) for r in robots]}


def sample_episode(
    task: Task, rng: np.random.Generator, margin: float = 0.02, max_tries: int = 1000
) -> tuple[dict[str, Any], int]:
    """A swarp initial state with no footprint overlap; returns ``(state, rejected draws)``."""
    half = task.agent_half or (task.agent_radius, task.agent_radius)
    for tries in range(max_tries):
        s = _swarp_draw(task, rng)
        tb = tee_boxes(task, *s["tee"])
        rs = s["robots"]
        bad = any(
            boxes_overlap((r[0], r[1]), r[2], half, c, yaw, h, margin) for r in rs for c, yaw, h in tb
        ) or any(
            boxes_overlap((a[0], a[1]), a[2], half, (b[0], b[1]), b[2], half, margin)
            for i, a in enumerate(rs) for b in rs[i + 1:]
        )
        if not bad:
            return s, tries
    raise RuntimeError(f"no overlap-free spawn in {max_tries} draws")


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))
