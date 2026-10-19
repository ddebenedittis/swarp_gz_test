"""The scripted sysid experiments, identical for swarp and Gazebo (numpy only).

Every experiment fixes the initial state (two robots and the T, at rest) and an open-loop
``(v, w)`` command per robot per control step of ``DT``. Robots not taking part are parked
far away with zero commands. ``lidar`` experiments have no commands: they compare one scan.

Geometry is the Limo recipe's: robot box half extents ``HX, HY``, the T of
``swarp.scenarios.shapes.tee(1.0, 0.3, 0.7, 0.3)`` (crossbar on top along +y, stem
hanging along -y, origin at the area centroid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DT = 0.1
HX, HY = 0.161, 0.110
BAR_TOP = 0.3559  # T frame: crossbar top face y
STEM_BOTTOM = -0.6441  # stem bottom face y
PARK_2 = (-1.5, 1.5, 0.0)  # robot_2 when idle
PARK_TEE = (1.2, 1.2, 0.0)  # the T when idle


@dataclass
class Experiment:
    name: str
    group: str  # "robot" | "push" | "lidar"
    robots: list[tuple[float, float, float]]
    tee: tuple[float, float, float]
    cmds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 2)))  # [steps, robot, (v, w)]
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.cmds) * DT


def _seq(*phases: tuple[float, float, float], robots: int = 1) -> np.ndarray:
    """Phases ``(seconds, v, w)`` for robot 1 (robot 2 still, unless ``robots=2``)."""
    rows = []
    for sec, v, w in phases:
        n = int(round(sec / DT))
        row = np.zeros((2, 2))
        row[0] = (v, w)
        if robots == 2:
            row[1] = (v, w)
        rows += [row] * n
    return np.array(rows)


def robot_experiments() -> list[Experiment]:
    start = [(-1.0, -1.0, 0.0), PARK_2]
    out = []
    for v in (0.1, 0.2, 0.4):
        out.append(Experiment(f"v{v}", "robot", start, PARK_TEE, _seq((3.0, v, 0.0), (1.5, 0, 0)), {"v": v, "w": 0.0}))
    for w in (0.5, 1.0, 1.5):
        out.append(Experiment(f"w{w}", "robot", start, PARK_TEE, _seq((3.0, 0.0, w), (1.5, 0, 0)), {"v": 0.0, "w": w}))
    out.append(Experiment("arc_v0.3_w0.75", "robot", start, PARK_TEE, _seq((4.0, 0.3, 0.75), (1.5, 0, 0)), {"v": 0.3, "w": 0.75}))
    out.append(Experiment("arc_v0.2_w-1.0", "robot", start, PARK_TEE, _seq((4.0, 0.2, -1.0), (1.5, 0, 0)), {"v": 0.2, "w": -1.0}))
    out.append(Experiment("reverse_0.3", "robot", start, PARK_TEE, _seq((2.0, 0.3, 0), (2.0, -0.3, 0), (1.5, 0, 0)), {"v": 0.3, "w": 0.0}))
    return out


def _behind(contact: tuple[float, float], heading: float, gap: float) -> tuple[float, float, float]:
    """Robot pose whose front face centre is ``gap`` short of ``contact``, facing ``heading``."""
    d = HX + gap
    return (contact[0] - d * math.cos(heading), contact[1] - d * math.sin(heading), heading)


def push_experiments() -> list[Experiment]:
    out = []
    gap = 0.15
    tee = (0.0, -0.6, 0.0)
    r = _behind((0.0, tee[1] + STEM_BOTTOM), math.pi / 2, gap)
    out.append(Experiment("stem_v0.2", "push", [r, PARK_2], tee, _seq((6.0, 0.2, 0), (2.0, 0, 0)), {"v": 0.2}))
    out.append(Experiment("stem_v0.4", "push", [r, PARK_2], tee, _seq((3.5, 0.4, 0), (2.0, 0, 0)), {"v": 0.4}))
    tee = (0.0, 0.7, 0.0)
    top = tee[1] + BAR_TOP
    for off in (0.0, 0.2, 0.4):
        r = _behind((off, top), -math.pi / 2, gap)
        out.append(Experiment(f"bar_off{off}", "push", [r, PARK_2], tee, _seq((5.0, 0.3, 0), (2.0, 0, 0)), {"v": 0.3, "offset": off}))
    h = -math.pi / 2 + math.radians(30)
    r = _behind((0.0, top), h, gap)
    out.append(Experiment("bar_glance30", "push", [r, PARK_2], tee, _seq((5.0, 0.3, 0), (2.0, 0, 0)), {"v": 0.3}))
    r1 = _behind((-0.2, top), -math.pi / 2, gap)
    r2 = _behind((0.2, top), -math.pi / 2, gap)
    out.append(Experiment("bar_pair", "push", [r1, r2], tee, _seq((5.0, 0.3, 0), (2.0, 0, 0), robots=2), {"v": 0.3}))
    return out


def lidar_experiments() -> list[Experiment]:
    """Robot 1 against the crossbar's top face, the face at body bearing ``beta``."""
    out = []
    tee = (0.0, 0.0, 0.0)
    for beta_deg in (0, 45, 90, 135, 180):
        b = math.radians(beta_deg)
        for gap in (0.005, 0.05, 0.15):
            d = HX * abs(math.cos(b)) + HY * abs(math.sin(b)) + gap
            th = -math.pi / 2 - b  # body bearing beta points along world -y, at the face
            r = (0.0, BAR_TOP + d, math.atan2(math.sin(th), math.cos(th)))
            out.append(Experiment(f"lidar_b{beta_deg}_g{gap}", "lidar", [r, PARK_2], tee, meta={"beta": beta_deg, "gap": gap}))
    return out


def all_experiments() -> list[Experiment]:
    return robot_experiments() + push_experiments() + lidar_experiments()


def by_name() -> dict[str, Experiment]:
    return {e.name: e for e in all_experiments()}
