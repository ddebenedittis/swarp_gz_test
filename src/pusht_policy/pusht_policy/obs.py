"""Rebuild swarp's ``pusht_lidar`` observation from world-frame poses and a LaserScan.

Pure numpy: no ROS, no torch, no swarp import, so the same code runs in a ROS node, in a
unit test, and on a robot. Two halves:

* :class:`ObsBuilder` reproduces, for one robot, the observation row
  ``PushTLidarScenario`` computes (``swarp/scenarios/pusht_lidar.py``, ``obs_layout`` and
  the torch oracle ``_refresh_lidar``; fused kernel ``pusht_lidar_obs_kernel``), given
  every robot's pose, velocity and role, the goal pose and the robot's own ``R``-ray scan
  as swarp would see it.
* :func:`scan_to_rays` and :func:`label_rays` turn a real 2D LaserScan into that ``R``-ray
  scan: swarp's rays are infinitely thin, start at the robot centre (not at the lidar),
  ray ``r`` points at ``r * 2 pi / R`` from the heading (counter-clockwise, ray 0 forward;
  over a ``lidar_span`` below ``2 pi``, from ``-span / 2`` to ``+span / 2`` in ``R - 1``
  steps), and the arena walls are invisible to them.

Observation at ``obs_level=1``, ``A`` robots, ``R`` rays, ``F`` stacked frames (all in the
world frame unless noted)::

    pos (2) | vel (2) | role sign (1: +1 observer, -1 pusher) | goal - pos (2)
    | cos, sin goal heading (2) | cos, sin own heading (2, only when heading matters)
    | arena half-extents W/2, H/2 (2, only for a checkpoint trained on ``arena_size``)
    | per teammate, ascending index, self skipped: pj - p (2), role sign (1)
    | F frames, oldest first, each [range / max_range (R) | T-hit mask (R) | robot-hit mask (R)
      | outside mask (R, only for a checkpoint trained with ``lidar_arena_flag``)]

The outside mask marks the T-labelled rays whose hit, placed from the robot's reported pose,
lies past the arena on either axis (the arena half-extents with ``arena_size``, else
``world_size``): a wall or crate just outside the arena still reads as the T, but flagged.

The frame ring shifts by one frame per control step and is zeroed on reset, so the first
observation of an episode carries the current scan and ``F - 1`` zero frames.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: swarp's per-ray hit classes (``swarp.sensors.lidar_kernels``).
HIT_NONE = 0
HIT_AGENT = 1
HIT_MOVABLE = 3

#: Roles as ``PushTLidarScenario.agent_roles()`` and ``PolicyNet.greedy(obs, roles)`` use them.
ROLE_PUSHER = 0
ROLE_OBSERVER = 1

#: ``PushTScenario`` defaults the obs depends on, for checkpoints that leave them implicit.
_DEFAULT_AGENT_RADIUS = 0.025

#: A per-env-arena checkpoint's arena half side [m] where a tool needs one and none is
#: given: the 4 x 4 m every earlier Gazebo evaluation ran in.
DEFAULT_ARENA_HALF = 2.0


def _parse(raw: Any) -> Any:
    """A checkpoint's ``scen_kwargs`` value (stored as ``str(v)``) back to a Python value."""
    if not isinstance(raw, str):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        # Enums print as "DynamicsModel.DIFF_DRIVE"; plain names ("diff_drive") stay.
        return raw.rsplit(".", 1)[-1].lower()


@dataclass(frozen=True)
class ObsConfig:
    """Everything the observation depends on, mirroring ``PushTLidarScenario``'s kwargs."""

    n_agents: int
    n_pushers: int = 2
    n_rays: int = 32
    lidar_max_range: float | None = None  # None: 0.8 * world_size, as swarp
    frame_stack: int = 1
    world_size: float = 1.0
    obs_level: int = 1
    model: str = "holonomic"
    agent_shape: str = "disc"
    agent_radius: float = _DEFAULT_AGENT_RADIUS
    agent_half_extents: tuple[float, float] | None = None  # None: (r, r / 2) for a box
    lidar_body_frame: bool | None = None  # None: whenever the heading matters
    lidar_min_range: float = 0.0  # from the lidar origin; returns closer than this are misses
    lidar_fov_at_mount: bool = False  # the field of view is cut at the mount, not per centre ray
    lidar_span: float = 2.0 * math.pi  # the angle the rays span, centred on the heading
    open_arena: bool = False  # no walls: hits are labelled up to label_bound
    oob_margin: float | None = None  # the T fails its episode past world_size + this
    # The trained arena sizes (swarp's ``arena_size``: a value, (lo, hi), or per axis
    # (w_lo, w_hi, h_lo, h_hi), full extents in metres); None: the fixed square world_size.
    arena_size: float | tuple[float, ...] | None = None
    # The deployed arena's half-extents (x, y) [m]: observed, and the label bound per axis.
    # Required with ``arena_size``, and within its trained range.
    arena_half: tuple[float, float] | None = None
    # Each frame adds the outside-the-arena mask (swarp's ``lidar_arena_flag``).
    lidar_arena_flag: bool = False

    def __post_init__(self) -> None:
        if self.obs_level != 1:
            raise NotImplementedError(
                f"only obs_level=1 is supported (got {self.obs_level}): level 0 observes the "
                "T pose, level 2 the teammates' pooled scans"
            )
        if self.n_rays < 1 or self.frame_stack < 1 or self.n_agents < 1:
            raise ValueError("n_agents, n_rays and frame_stack must all be >= 1")
        if self.model not in ("holonomic", "diff_drive"):
            raise ValueError(f"unknown model {self.model!r}")
        if self.agent_shape not in ("disc", "box"):
            raise ValueError(f"unknown agent_shape {self.agent_shape!r}")
        if self.per_env_arena:
            if self.arena_half is None:
                raise ValueError(
                    "this checkpoint observes the arena's size (it trained on arena_size "
                    f"{self.arena_size}): pass the deployed arena's half-extents (arena_half)"
                )
            lo, hi = self.arena_half_range
            for ax, h in enumerate(self.arena_half):
                if not lo[ax] - 1e-6 <= h <= hi[ax] + 1e-6:
                    raise ValueError(
                        f"arena_half {tuple(self.arena_half)} is outside the trained half-extents "
                        f"{lo[ax]}..{hi[ax]} m on axis {'xy'[ax]}: the policy never saw that arena"
                    )

    # --- derived, with swarp's defaults

    @property
    def max_range(self) -> float:
        return 0.8 * self.world_size if self.lidar_max_range is None else self.lidar_max_range

    @property
    def heading_obs(self) -> bool:
        return self.model != "holonomic" or self.agent_shape == "box"

    @property
    def body_frame(self) -> bool:
        return self.heading_obs if self.lidar_body_frame is None else self.lidar_body_frame

    @property
    def half_extents(self) -> tuple[float, float]:
        """A box robot's half-extents (along, across the heading)."""
        if self.agent_half_extents is not None:
            return tuple(float(x) for x in self.agent_half_extents)
        return (self.agent_radius, 0.5 * self.agent_radius)

    @property
    def per_env_arena(self) -> bool:
        """Whether the checkpoint trained on varying arenas and observes the one it is in."""
        return self.arena_size is not None

    @property
    def arena_half_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The trained half-extents, ``((x_lo, y_lo), (x_hi, y_hi))`` (swarp's
        ``arena_ranges`` halved)."""
        v = self.arena_size
        if isinstance(v, (int, float)):
            r = (float(v),) * 4
        elif len(v) == 2:
            r = (float(v[0]), float(v[1])) * 2
        else:
            r = tuple(float(x) for x in v)
        return (0.5 * r[0], 0.5 * r[2]), (0.5 * r[1], 0.5 * r[3])

    @property
    def label_bound(self) -> float | tuple[float, float]:
        """How far out a hit still counts (swarp's ``label_bound``): the out-of-bounds limit
        when there is one, else the arena (none in an open arena). Per axis, ``(x, y)``, for a
        per-env arena: the deployed arena's half-extents plus ``oob_margin``."""
        if self.per_env_arena:
            m = 0.0 if self.oob_margin is None else float(self.oob_margin)
            return (self.arena_half[0] + m, self.arena_half[1] + m)
        if self.oob_margin is not None:
            return self.world_size + float(self.oob_margin)
        return math.inf if self.open_arena else self.world_size

    @property
    def pusher_count(self) -> int:
        return max(0, min(self.n_pushers, self.n_agents))

    @property
    def self_dim(self) -> int:
        return 9 + (2 if self.heading_obs else 0) + (2 if self.per_env_arena else 0)

    @property
    def ray_channels(self) -> int:
        """Values per ray in a lidar frame: range, T mask, robot mask (, outside mask)."""
        return 4 if self.lidar_arena_flag else 3

    @property
    def arena_extent(self) -> tuple[float, float]:
        """The arena's half-extents the outside mask tests against."""
        if self.per_env_arena:
            return self.arena_half
        return (self.world_size, self.world_size)

    @property
    def obs_dim(self) -> int:
        frames = self.ray_channels * self.n_rays * self.frame_stack
        return self.self_dim + 3 * (self.n_agents - 1) + frames

    def layout(self) -> dict[str, slice]:
        """Name -> slice of the row, as ``PushTLidarScenario.obs_layout()`` at level 1."""
        out = {
            "pos": slice(0, 2),
            "vel": slice(2, 4),
            "role": slice(4, 5),
            "goal_rel": slice(5, 7),
            "goal_angle": slice(7, 9),
        }
        if self.heading_obs:
            out["heading"] = slice(9, 11)
        if self.per_env_arena:
            k = 11 if self.heading_obs else 9
            out["arena"] = slice(k, k + 2)
        k = self.self_dim
        out["teammates"] = slice(k, k + 3 * (self.n_agents - 1))
        k += 3 * (self.n_agents - 1)
        w = self.ray_channels * self.n_rays
        out["lidar"] = slice(k, k + w * self.frame_stack)
        out["lidar_newest"] = slice(k + w * (self.frame_stack - 1), k + w * self.frame_stack)
        return out

    @classmethod
    def from_checkpoint(
        cls, meta: Mapping[str, Any], arena_half: Sequence[float] | float | None = None
    ) -> ObsConfig:
        """From the metadata dict ``swarp.rl.ppo.load_checkpoint`` returns (or the raw
        ``torch.load`` of the file). Checks the result against the stored ``obs_dim``.

        ``meta["scenario"]`` is the recipe name the run was trained with: ``pusht_lidar`` or
        a variant of it such as ``pusht_lidar_limo``. ``arena_half`` (a value for a square,
        or ``(x, y)``) is the deployed arena, needed by a checkpoint trained on
        ``arena_size`` and ignored by the others."""
        name = str(meta.get("scenario", "pusht_lidar"))
        if name != "pusht_lidar" and not name.startswith("pusht_lidar_"):
            raise ValueError(f"not a pusht_lidar checkpoint: {name!r}")
        kw = {k: _parse(v) for k, v in dict(meta.get("scen_kwargs", {})).items()}
        names = cls.__dataclass_fields__.keys() - {"n_agents", "arena_half"}
        args = {k: kw[k] for k in names if k in kw}
        if args.get("arena_size") is not None and arena_half is not None:
            h = (arena_half,) * 2 if isinstance(arena_half, (int, float)) else tuple(arena_half)
            args["arena_half"] = (float(h[0]), float(h[1]))
        if "n_pushers" not in args and "role_counts" in meta:
            args["n_pushers"] = int(meta["role_counts"][0])
        if args.get("agent_half_extents") is not None:
            args["agent_half_extents"] = tuple(float(x) for x in args["agent_half_extents"])
        if "lidar_fov_at_mount" not in kw and "lidar_fov_at_mount" in pruned_kwargs():
            # Trained on a swarp that always cuts the field of view at the mount.
            args["lidar_fov_at_mount"] = True
        cfg = cls(n_agents=int(meta["n_agents"]), **args)
        if "obs_dim" in meta and int(meta["obs_dim"]) != cfg.obs_dim:
            raise ValueError(
                f"checkpoint obs_dim {meta['obs_dim']} != {cfg.obs_dim} rebuilt from {kw}"
            )
        return cfg


class ObsBuilder:
    """One robot's observation row, with its own frame-stack ring.

    Call :meth:`reset` at the start of every episode and :meth:`build` exactly once per
    control step (each call pushes one scan frame into the ring, as one swarp step does).
    """

    def __init__(self, cfg: ObsConfig, agent: int) -> None:
        if not 0 <= agent < cfg.n_agents:
            raise ValueError(f"agent {agent} out of range for {cfg.n_agents} agents")
        self.cfg = cfg
        self.agent = agent
        self._others = [j for j in range(cfg.n_agents) if j != agent]
        self._width = cfg.ray_channels * cfg.n_rays
        self.ring = np.zeros(cfg.frame_stack * self._width, dtype=np.float32)

    def reset(self) -> None:
        """Zero every stacked frame (swarp zeroes the ring on reset)."""
        self.ring[:] = 0.0

    def build(
        self,
        pos: np.ndarray,  # [A, 2] world
        vel: np.ndarray,  # [A, 2] world
        theta: np.ndarray,  # [A]
        roles: np.ndarray,  # [A] int: 0 pusher, 1 observer
        goal_pos: np.ndarray,  # [2] world
        goal_theta: float,
        ranges: np.ndarray,  # [R] from this robot's centre, max_range on a miss
        hit_class: np.ndarray,  # [R] HIT_NONE / HIT_AGENT / HIT_MOVABLE
    ) -> np.ndarray:
        """Push this step's scan into the ring and return the ``[obs_dim]`` float32 row."""
        cfg, a = self.cfg, self.agent
        pos = np.asarray(pos, dtype=np.float64).reshape(cfg.n_agents, 2)
        vel = np.asarray(vel, dtype=np.float64).reshape(cfg.n_agents, 2)
        theta = np.asarray(theta, dtype=np.float64).reshape(cfg.n_agents)
        sgn = np.where(np.asarray(roles).reshape(cfg.n_agents) == ROLE_OBSERVER, 1.0, -1.0)
        goal = np.asarray(goal_pos, dtype=np.float64).reshape(2)
        ranges = np.asarray(ranges, dtype=np.float64).reshape(cfg.n_rays)
        cls = np.asarray(hit_class).reshape(cfg.n_rays)

        p = pos[a]
        feats = [p, vel[a], [sgn[a]], goal - p, [math.cos(goal_theta), math.sin(goal_theta)]]
        if cfg.heading_obs:
            feats.append([math.cos(theta[a]), math.sin(theta[a])])
        if cfg.per_env_arena:
            feats.append(cfg.arena_half)
        for j in self._others:
            feats.append([pos[j, 0] - p[0], pos[j, 1] - p[1], sgn[j]])

        w = self._width
        masks = [cls == HIT_MOVABLE, cls == HIT_AGENT]
        if cfg.lidar_arena_flag:
            phi = ray_angles(cfg.n_rays, cfg.lidar_span) + (theta[a] if cfg.body_frame else 0.0)
            hx = p[0] + ranges * np.cos(phi)
            hy = p[1] + ranges * np.sin(phi)
            ex, ey = cfg.arena_extent
            masks.append((cls == HIT_MOVABLE) & ((np.abs(hx) > ex) | (np.abs(hy) > ey)))
        frame = np.concatenate([ranges / cfg.max_range, *masks]).astype(np.float32)
        self.ring[:-w] = self.ring[w:]
        self.ring[-w:] = frame
        head = np.concatenate([np.asarray(f, dtype=np.float64) for f in feats])
        return np.concatenate([head.astype(np.float32), self.ring])


# --------------------------------------------------------------------- scan conversion


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def ray_angles(n_rays: int, span: float = 2.0 * math.pi) -> np.ndarray:
    """swarp's ray angles relative to the ray frame, counter-clockwise: over the full
    circle ray 0 forward, ``2 pi / n_rays`` apart; over a smaller ``span`` from ``-span / 2``
    to ``+span / 2`` (``PushTLidarScenario(lidar_span=...)``)."""
    if span < 2.0 * math.pi:
        step = span / (n_rays - 1) if n_rays > 1 else 0.0
        return -0.5 * span + np.arange(n_rays) * step
    return np.arange(n_rays) * (2.0 * math.pi / n_rays)


def scan_points(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    mount_xy: Sequence[float] = (0.0, 0.0),
    mount_yaw: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """LaserScan -> ``([N, 2] points in the robot-centre body frame, [N] valid mask)``.

    A return is valid when finite and in ``[range_min, range_max)`` (ROS's convention;
    a reading at ``range_max`` is a no-return). Invalid points are left at the mount.
    """
    r = np.asarray(ranges, dtype=np.float64)
    valid = np.isfinite(r) & (r >= range_min) & (r < range_max)
    r = np.where(valid, r, 0.0)
    ang = angle_min + np.arange(r.size) * angle_increment + mount_yaw
    pts = np.asarray(mount_xy, dtype=np.float64) + r[:, None] * np.stack(
        [np.cos(ang), np.sin(ang)], axis=-1
    )
    return pts, valid


def scan_to_rays(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    mount_xy: Sequence[float],
    mount_yaw: float,
    n_rays: int,
    max_range: float,
    *,
    pose: Sequence[float] = (0.0, 0.0, 0.0),
    own_half: Sequence[float] | None = None,
    own_margin: float = -0.03,
    max_gap: float = 0.15,
    body_frame: bool = True,
    span: float = 2.0 * math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a LaserScan to swarp's ``n_rays`` thin rays from the robot centre, over
    ``span`` (:func:`ray_angles`).

    Returns ``(ray ranges [n_rays], valid scan points [N, 2] in the world frame)``; a ray
    that hits nothing within ``max_range`` reads ``max_range``. ``pose = (x, y, theta)``
    places the robot in the world (only the returned points and, with
    ``body_frame=False``, the ray directions depend on it).

    The scan is turned back into the surface it sampled, then each centre ray is cast
    against it, rather than taking the minimum range of the points in a bearing bin:

    * consecutive returns closer than ``max_gap`` are joined by a segment (the lidar's own
      angular order is the surface order seen from the mount, and the mount sits a few
      centimetres from the centre, so the polyline is the visible outline from the
      centre too); larger gaps are depth jumps and stay open. The default is below the
      narrowest object (a 0.22 m Limo, a 0.3 m T bar), so a gap between two objects is
      not bridged, and above the spacing of a face seen at a grazing angle;
    * each chain of segments is extended at both ends by half the sample spacing, where
      the object's edge lies on average, and a return with no joined neighbour is a disc
      of that radius;
    * each ray's range is its nearest intersection with those segments and discs.

    A bin minimum instead reports the nearest point anywhere in a ``2 pi / n_rays``
    sector (11 degrees at 32 rays), which at a T edge or a robot corner is often a point
    off the thin ray entirely: it widens every object by up to half a bin on each side
    and biases ranges short on slanted faces. On Limo-scale swarp states (2 robots, 32
    rays, a 360-sample scan from the 0.103 m mount with 0.01 m noise) this classifies
    0.4% of rays differently from swarp's own lidar against 4.1% for the bin minimum,
    with a 7 mm mean range error against 56 mm; most of the remaining mismatches are
    surfaces the centre sees and the mount does not.

    Points inside the robot's own footprint grown by ``own_margin`` are dropped first, since
    swarp's robot never sees itself. The default -0.03 drops only points more than 3 cm
    inside: a T touching the robot's side or rear returns points scattered by the range
    noise on both sides of the footprint edge, and swarp's centre rays see that T.
    """
    pts, valid = scan_points(
        ranges, angle_min, angle_increment, range_min, range_max, mount_xy, mount_yaw
    )
    n = pts.shape[0]
    if own_half is not None:
        inside = (np.abs(pts[:, 0]) <= own_half[0] + own_margin) & (
            np.abs(pts[:, 1]) <= own_half[1] + own_margin
        )
        valid &= ~inside
    x, y, th = (float(v) for v in pose)
    phi = ray_angles(n_rays, span) if body_frame else ray_angles(n_rays, span) - th
    d = np.stack([np.cos(phi), np.sin(phi)], axis=-1)  # [R, 2] body frame
    best = np.full(n_rays, float(max_range))

    # Segments between consecutive valid returns, wrapping round a full-circle scan.
    idx = np.arange(n)
    nxt = (idx + 1) % n
    full = n * abs(angle_increment) >= 2.0 * math.pi - 0.5 * abs(angle_increment)
    if not full:
        nxt, idx = nxt[:-1], idx[:-1]
    seg = valid[idx] & valid[nxt] & (np.linalg.norm(pts[nxt] - pts[idx], axis=-1) < max_gap)
    # Half the local sample spacing: how far an object's true edge lies, on average, past
    # its last return.
    spacing = 0.5 * np.linalg.norm(pts - np.asarray(mount_xy), axis=-1) * abs(angle_increment)
    a, b = pts[idx[seg]], pts[nxt[seg]]
    if a.shape[0]:
        # Extend each chain's end segments by that much, so a thin ray that passes just
        # beyond the last return of an edge still hits it.
        starts = np.ones(n, dtype=bool)
        starts[nxt[seg]] = False  # a point some segment ends at does not start a chain
        ends = np.ones(n, dtype=bool)
        ends[idx[seg]] = False
        u = (b - a) / np.maximum(np.linalg.norm(b - a, axis=-1, keepdims=True), 1e-12)
        a = a - (starts[idx[seg]] * spacing[idx[seg]])[:, None] * u
        b = b + (ends[nxt[seg]] * spacing[nxt[seg]])[:, None] * u
        e = b - a  # [S, 2]
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]  # [R, S]
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        t = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
        s = (a[None, :, 0] * d[:, None, 1] - a[None, :, 1] * d[:, None, 0]) / den
        hit = ok & (s >= 0.0) & (s <= 1.0) & (t > 0.0)
        best = np.minimum(best, np.where(hit, t, np.inf).min(axis=1))

    # Isolated returns as discs of half the local sample spacing.
    joined = np.zeros(n, dtype=bool)
    joined[idx[seg]] = True
    joined[nxt[seg]] = True
    lone = valid & ~joined
    if lone.any():
        c = pts[lone]  # [L, 2]
        rad = np.maximum(spacing[lone], 1e-3)
        proj = d @ c.T  # [R, L]
        perp2 = (c * c).sum(-1)[None, :] - proj * proj
        thc = np.sqrt(np.maximum(rad[None, :] ** 2 - perp2, 0.0))
        t = proj - thc
        hit = (perp2 <= rad[None, :] ** 2) & (proj > 0.0) & (t > 0.0)
        best = np.minimum(best, np.where(hit, t, np.inf).min(axis=1))

    best = np.minimum(best, max_range)
    world = pts[valid] @ _rot(th).T + np.array([x, y])
    return best, world


def label_rays(
    ray_ranges: np.ndarray,
    pose: Sequence[float],
    max_range: float,
    world_size: float | Sequence[float],
    mates_pos: np.ndarray,
    mates_theta: np.ndarray,
    mate_half: Sequence[float] | None = None,
    mate_radius: float | None = None,
    *,
    margin: float = 0.03,
    body_frame: bool = True,
    span: float = 2.0 * math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify each ray's hit point: ``(ranges, hit_class)``, both ``[n_rays]``.

    * no hit (``range >= max_range``), or a hit outside the square
      ``|x| > world_size or |y| > world_size`` (or the rectangle, for a per-axis
      ``(x, y)``) -> :data:`HIT_NONE` and ``max_range``: swarp's walls are a position
      clamp, invisible to its lidar (pass :attr:`ObsConfig.label_bound` as ``world_size``:
      in an open arena it reaches past the walls, to where the T fails its episode);
    * a hit inside a teammate's footprint grown by ``margin`` (an oriented box of
      ``mate_half``, or a disc of ``mate_radius``) -> :data:`HIT_AGENT`; this wins over the
      T, as swarp's ties go to agents;
    * anything else -> :data:`HIT_MOVABLE` (the T).

    ``mates_pos [M, 2]`` / ``mates_theta [M]`` are the other robots' world poses; the rays
    span ``span`` (:func:`ray_angles`). swarp's ``lidar_label_from_pose`` trains on this
    rule applied to the poses the robots report.
    """
    r = np.asarray(ray_ranges, dtype=np.float64).copy()
    x, y, th = (float(v) for v in pose)
    phi = ray_angles(r.size, span) + (th if body_frame else 0.0)
    hx = x + r * np.cos(phi)
    hy = y + r * np.sin(phi)
    cls = np.full(r.size, HIT_MOVABLE, dtype=np.uint8)
    mp = np.asarray(mates_pos, dtype=np.float64).reshape(-1, 2)
    mt = np.asarray(mates_theta, dtype=np.float64).reshape(-1)
    for (mx, my), mth in zip(mp, mt, strict=True):
        dx, dy = hx - mx, hy - my
        if mate_half is not None:
            c, s = math.cos(mth), math.sin(mth)
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            inside = (np.abs(lx) <= mate_half[0] + margin) & (np.abs(ly) <= mate_half[1] + margin)
        else:
            inside = np.hypot(dx, dy) <= mate_radius + margin
        cls[inside] = HIT_AGENT
    bx, by = (world_size, world_size) if np.ndim(world_size) == 0 else world_size
    miss = (r >= max_range) | (np.abs(hx) > bx) | (np.abs(hy) > by)
    cls[miss] = HIT_NONE
    r[miss] = max_range
    return r, cls


#: swarp's slack on the field-of-view edge (``pusht_lidar_kernels.FOV_EPS``).
FOV_EPS = 1e-4


def privileged_cols(
    tee: Sequence[float], own_xy: Sequence[float], goal: Sequence[float], scale: float,
    goal_tolerance: float, angle_tolerance: float, flag: bool,
) -> np.ndarray:
    """swarp's ``aux_targets`` row for one robot, which an actor trained with
    ``--privileged-actor`` (``priv_dim`` > 0) reads after its observation: the T's position
    ``tee[:2]`` relative to the robot (world axes) times ``scale`` (``1 / lidar_max_range``),
    the cos/sin of the T's heading and, with ``flag`` (``aux_goal``), 1 while the T is within
    both goal tolerances of ``goal`` = ``(x, y, heading)`` and 0 otherwise."""
    tx, ty, th = tee
    cols = [(tx - own_xy[0]) * scale, (ty - own_xy[1]) * scale, math.cos(th), math.sin(th)]
    if flag:
        err = math.atan2(math.sin(th - goal[2]), math.cos(th - goal[2]))
        on = math.hypot(tx - goal[0], ty - goal[1]) < goal_tolerance and abs(err) < angle_tolerance
        cols.append(float(on))
    return np.asarray(cols, dtype=np.float32)


def pruned_kwargs() -> set[str]:
    """The flags the installed swarp no longer takes, because it implements only their True
    value (``lidar_fov_at_mount``, ``scenery_collide``); empty without swarp. Checkpoints
    from before the pruning still store them, and :class:`~pusht_policy.task.Task` drops
    them before building the scenario."""
    try:
        import inspect

        from swarp.scenarios.pusht_lidar import PushTLidarScenario
    except ImportError:
        return set()
    accepted = inspect.signature(PushTLidarScenario.__init__).parameters
    return {k for k in ("lidar_fov_at_mount", "scenery_collide") if k not in accepted}


def crop_fov_at_mount(
    ranges: Sequence[float], angle_min: float, angle_increment: float, fov: float, mount_yaw: float = 0.0
) -> np.ndarray:
    """swarp's ``lidar_fov_at_mount`` field of view on a raw scan: every return whose bearing
    from the mount, off the heading, exceeds ``fov / 2 + FOV_EPS`` becomes a no-return (inf).
    A sensor cropped (or occluded) at ``fov`` already looks like this; checkpoints trained with
    the cut at the mount need nothing else, those trained per centre ray use
    :func:`blank_outside_fov` instead."""
    r = np.asarray(ranges, dtype=np.float64).copy()
    if fov >= 2.0 * math.pi:
        return r
    ang = angle_min + np.arange(r.size) * angle_increment + mount_yaw
    r[np.abs(np.arctan2(np.sin(ang), np.cos(ang))) > 0.5 * fov + FOV_EPS] = np.inf
    return r


def blank_outside_fov(
    ranges: np.ndarray, hit_class: np.ndarray, fov: float, max_range: float,
    span: float = 2.0 * math.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """swarp's field-of-view blanking (``lidar_fov``): ray ``r``, at ``r * 2 pi / R`` off the
    heading (wrapped to ``(-pi, pi]``), reads as a miss (``max_range``, :data:`HIT_NONE`)
    when that offset exceeds ``fov / 2 + FOV_EPS``. A deployment parameter: ``fov`` is the
    robot's measured unobstructed field of view, whatever the checkpoint was trained on;
    ``fov >= 2 pi`` blanks nothing."""
    r = np.asarray(ranges, dtype=np.float64).copy()
    cls = np.asarray(hit_class).copy()
    if fov >= 2.0 * math.pi:
        return r, cls
    off = ray_angles(r.size, span)
    rel = np.arctan2(np.sin(off), np.cos(off))
    blind = np.abs(rel) > 0.5 * fov + FOV_EPS
    r[blind] = max_range
    cls[blind] = HIT_NONE
    return r, cls


__all__ = [
    "DEFAULT_ARENA_HALF",
    "HIT_AGENT",
    "HIT_MOVABLE",
    "HIT_NONE",
    "ROLE_OBSERVER",
    "ROLE_PUSHER",
    "ObsBuilder",
    "ObsConfig",
    "label_rays",
    "blank_outside_fov",
    "crop_fov_at_mount",
    "privileged_cols",
    "ray_angles",
    "scan_points",
    "scan_to_rays",
]
