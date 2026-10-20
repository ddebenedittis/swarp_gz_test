"""Static walls and landmarks around the arena, for lidar localization (AMCL), and the
occupancy map rasterized from the same geometry.

The arena is the ±2 m square the policies were trained in; everything here lies outside it,
so ``pusht_policy.obs.label_rays`` turns every hit on it into a miss and the policy's
observation does not change. A room of 7.0 × 6.4 m, off-centre, plus four boxes of
different sizes, so the layout has no symmetry AMCL could alias.

    ros2 run pusht_gz make_map <dir>      # writes <dir>/landmarks.pgm and landmarks.yaml

The map's frame is the arena frame (origin at the arena centre, x along the arena's +x).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

#: (name, x, y, yaw, size_x, size_y) of each box; all 0.5 m tall.
BOXES: list[tuple[str, float, float, float, float, float]] = [
    ("wall_east", 3.55, 0.15, 0.0, 0.1, 6.5),
    ("wall_west", -3.45, 0.15, 0.0, 0.1, 6.5),
    ("wall_north", 0.05, 3.35, 0.0, 7.1, 0.1),
    ("wall_south", 0.05, -3.05, 0.0, 7.1, 0.1),
    ("crate", 2.75, 1.9, 0.0, 0.4, 0.6),
    ("cabinet", -2.85, -1.2, 0.4, 0.5, 0.3),
    ("post", 0.8, -2.55, 0.0, 0.25, 0.25),
    ("bench", -1.4, 2.75, 0.0, 0.9, 0.2),
]
HEIGHT = 0.5


def sdf(name: str = "landmarks") -> str:
    """A static SDF model holding every box (lidar-visible, default visibility flags)."""
    links = []
    for n, x, y, yaw, sx, sy in BOXES:
        geom = f"<geometry><box><size>{sx} {sy} {HEIGHT}</size></box></geometry>"
        links.append(
            f'<link name="{n}"><pose>{x} {y} {HEIGHT / 2} 0 0 {yaw}</pose>'
            f'<collision name="c">{geom}</collision>'
            f'<visual name="v">{geom}<material><ambient>0.55 0.45 0.35 1</ambient>'
            f"<diffuse>0.55 0.45 0.35 1</diffuse></material></visual></link>"
        )
    return f'<?xml version="1.0"?><sdf version="1.9"><model name="{name}"><static>true</static>{"".join(links)}</model></sdf>'


def occupancy(resolution: float = 0.05, pad: float = 0.5) -> tuple[np.ndarray, tuple[float, float]]:
    """``(grid [H, W] bool, row 0 = lowest y; (origin_x, origin_y))`` of the boxes."""
    corners = []
    for _, x, y, yaw, sx, sy in BOXES:
        c, s = math.cos(yaw), math.sin(yaw)
        for dx, dy in ((sx, sy), (-sx, sy), (sx, -sy), (-sx, -sy)):
            corners.append((x + c * dx / 2 - s * dy / 2, y + s * dx / 2 + c * dy / 2))
    pts = np.array(corners)
    lo = np.floor((pts.min(0) - pad) / resolution) * resolution
    hi = np.ceil((pts.max(0) + pad) / resolution) * resolution
    w, h = (int(round(v)) for v in (hi - lo) / resolution)
    cx = lo[0] + (np.arange(w) + 0.5) * resolution
    cy = lo[1] + (np.arange(h) + 0.5) * resolution
    gx, gy = np.meshgrid(cx, cy)
    grid = np.zeros((h, w), dtype=bool)
    for _, x, y, yaw, sx, sy in BOXES:
        c, s = math.cos(yaw), math.sin(yaw)
        lx = c * (gx - x) + s * (gy - y)
        ly = -s * (gx - x) + c * (gy - y)
        # A cell is occupied when the box covers any of it (half a cell of slack).
        grid |= (np.abs(lx) <= sx / 2 + resolution / 2) & (np.abs(ly) <= sy / 2 + resolution / 2)
    return grid, (float(lo[0]), float(lo[1]))


def write_map(out_dir: str, name: str = "landmarks", resolution: float = 0.05) -> str:
    """Write ``<name>.pgm`` and ``<name>.yaml`` (nav2 map_server, trinary) and return the yaml path."""
    grid, (ox, oy) = occupancy(resolution)
    os.makedirs(out_dir, exist_ok=True)
    img = np.where(grid, 0, 254).astype(np.uint8)[::-1]  # PGM row 0 is the top (highest y)
    with open(os.path.join(out_dir, f"{name}.pgm"), "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    path = os.path.join(out_dir, f"{name}.yaml")
    with open(path, "w") as f:
        f.write(
            f"image: {name}.pgm\nmode: trinary\nresolution: {resolution}\n"
            f"origin: [{ox:.3f}, {oy:.3f}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n"
        )
    return path


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    print(write_map(out))
