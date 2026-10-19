"""Quasi-static reference for pushing the T with a Limo's flat front face (host, needs cvxpy).

An independent check of the sysid push experiments: no simulator, only quasi-static
mechanics.

* Floor: uniform pressure over the T footprint (a grid of ``--cell`` m cells), Coulomb
  friction at every cell, so the floor wrench is the exact limit surface of that pressure
  distribution: the gradient of the dissipated power ``D(xi) = sum_i p_i |v_i(xi)|``.
* Pusher: the robot box (0.322 x 0.220), velocity controlled along its heading, never
  deflected (an ideal velocity-servoed robot). Contacts are its corners against the T's
  edges and the T's vertices against its edges, unilateral, with Coulomb friction ``mu_c``.
* Each step of ``--ds`` metres of pusher travel solves for the T twist ``xi`` that
  minimises ``D(xi) + sum_j mu_c n_j |slip_j(xi)|`` subject to non-penetration (Peshkin's
  minimum-power principle; the contact friction term uses the normal forces ``n_j`` of the
  previous iterate, a fixed point repeated until ``n`` settles). The constraint duals are
  the contact normal forces.

Inertia is ignored (quasi-static), so the result does not depend on the push speed, the
T's mass or the floor's mu; only on geometry and ``mu_c / mu_floor``.

    python3 src/pusht_policy/scripts/pushing_reference.py --out out/sysid/reference.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cvxpy as cp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pusht_policy.sysid.experiments import BAR_TOP, HX, HY, push_experiments  # noqa: E402

# The Limo-recipe T (swarp.scenarios.shapes.tee(1.0, 0.3, 0.7, 0.3)): boxes (cx, cy, hx, hy)
TEE_BOXES = [(0.0, 0.20588235294117646, 0.5, 0.15), (0.0, -0.29411764705882354, 0.15, 0.35)]


def rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def box_corners(cx, cy, hx, hy):
    return np.array([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]])


def edges(poly):
    """(start, end, outward normal) of a CCW polygon."""
    out = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        t = (b - a) / np.linalg.norm(b - a)
        out.append((a, b, np.array([t[1], -t[0]])))
    return out


def inside_box(p, box, eps=1e-9):
    cx, cy, hx, hy = box
    return abs(p[0] - cx) < hx - eps and abs(p[1] - cy) < hy - eps


def tee_grid(cell):
    pts = []
    for cx, cy, hx, hy in TEE_BOXES:
        xs = np.arange(cx - hx + cell / 2, cx + hx, cell)
        ys = np.arange(cy - hy + cell / 2, cy + hy, cell)
        X, Y = np.meshgrid(xs, ys)
        pts.append(np.column_stack([X.ravel(), Y.ravel()]))
    return np.concatenate(pts)


def tee_outline_vertices():
    """Exterior vertices of the T (the two concave junction points included)."""
    v = []
    for b in TEE_BOXES:
        for c in box_corners(*b):
            if not any(inside_box(c, o) for o in TEE_BOXES):
                v.append(c)
    return np.array(v)


def tee_exterior_edges():
    """Edge pieces of the two boxes that are not inside the other box (T frame)."""
    out = []
    for i, b in enumerate(TEE_BOXES):
        other = TEE_BOXES[1 - i]
        for a, e, n in edges(box_corners(*b)):
            # sample the edge, keep the runs outside the other box
            ts = np.linspace(0, 1, 201)
            pts = a + ts[:, None] * (e - a)
            ok = np.array([not inside_box(p + 1e-6 * n, other) for p in pts])
            if ok.all():
                out.append((a, e, n))
            elif ok.any():
                idx = np.flatnonzero(ok)
                runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
                for r in runs:
                    out.append((pts[r[0]], pts[r[-1]], n))
    return out


def seg_point(p, a, b):
    """(distance along the outward side, projection parameter) of p to segment a-b."""
    ab = b - a
    t = np.dot(p - a, ab) / np.dot(ab, ab)
    return t


def contacts(tee_pose, robot_pose, tol, min_gap=-0.02):
    """Candidate contacts: (point, unit normal pushing the T, gap)."""
    tx, ty, tth = tee_pose
    Rt, Tt = rot(tth), np.array([tx, ty])
    rx, ry, rth = robot_pose
    Rr, Tr = rot(rth), np.array([rx, ry])
    robot_poly = box_corners(0, 0, HX, HY) @ Rr.T + Tr
    out = []
    # robot corners against T edges
    for a, b, n in TEE_EXT:
        aw, bw, nw = a @ Rt.T + Tt, b @ Rt.T + Tt, n @ Rt.T
        for c in robot_poly:
            t = seg_point(c, aw, bw)
            if -1e-9 <= t <= 1 + 1e-9:
                gap = float(np.dot(c - aw, nw))  # >0 outside the T
                if min_gap < gap < tol:
                    out.append((c - gap * nw, -nw, gap))
    # T vertices against robot edges
    for a, b, n in edges(robot_poly):
        for v in TEE_VERTS:
            vw = v @ Rt.T + Tt
            t = seg_point(vw, a, b)
            if 1e-6 < t < 1 - 1e-6:
                gap = float(np.dot(vw - a, n))
                if min_gap < gap < tol:
                    out.append((vw, n, gap))
    return out


def step(tee_pose, robot_pose, heading, grid_b, mu_c, ds, tol, iters=12, floor="coulomb"):
    """One quasi-static step: returns (twist per unit pusher travel, normal forces)."""
    tx, ty, tth = tee_pose
    off = grid_b @ rot(tth).T  # cell offsets from the T origin, world axes
    p = 1.0 / len(off)
    vp = np.array([math.cos(heading), math.sin(heading)])
    cts = contacts(tee_pose, robot_pose, tol)
    xi = cp.Variable(3)
    vx = xi[0] - xi[2] * off[:, 1]
    vy = xi[1] + xi[2] * off[:, 0]
    if floor == "coulomb":
        D = p * cp.sum(cp.norm(cp.vstack([vx, vy]), 2, axis=0))
    else:  # viscous (swarp's model): Rayleigh dissipation, the force matches Coulomb's at unit speed
        D = 0.5 * p * (cp.sum_squares(vx) + cp.sum_squares(vy))
    if not cts:
        return np.zeros(3), []
    cons, slips = [], []
    for c, n, gap in cts:
        r = c - np.array([tx, ty])
        vT = cp.hstack([xi[0] - xi[2] * r[1], xi[1] + xi[2] * r[0]])
        rel = vT - vp
        cons.append(rel @ n >= -gap / ds)
        t = np.array([-n[1], n[0]])
        slips.append(rel @ t)
    nf = np.zeros(len(cts))
    for _ in range(iters):
        obj = D + sum(mu_c * nf[j] * cp.abs(slips[j]) for j in range(len(cts))) if mu_c > 0 else D
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.CLARABEL)
        new = np.array([max(float(k.dual_value), 0.0) for k in cons])
        if mu_c == 0 or np.abs(new - nf).max() < 1e-4:
            nf = new
            break
        nf = 0.5 * nf + 0.5 * new
    return np.array(xi.value), nf


def simulate(exp, mu_c, ds, cell, push_len, tol=0.003, floor="coulomb"):
    """Pusher starts touching the T on the experiment's approach line, travels push_len."""
    r0 = np.array(exp.robots[0], float)
    heading = r0[2]
    d = np.array([math.cos(heading), math.sin(heading)])
    tee = np.array(exp.tee, float)
    grid_b = tee_grid(cell)
    # slide the robot forward until it first touches the T (the experiments start 0.15 m back)
    lo, hi = 0.0, 0.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        pose = (*(r0[:2] + mid * d), heading)
        if any(g < 1e-4 for _, _, g in contacts(tee, pose, 0.05, min_gap=-1.0)):
            hi = mid
        else:
            lo = mid
    robot = np.array([*(r0[:2] + hi * d), heading])
    path, yaw0 = 0.0, tee[2]
    traj = [tee.copy()]
    for _ in range(int(round(push_len / ds))):
        # contacts at the current poses; the twist then carries both over ds of pusher travel
        xi, _ = step(tee, robot, heading, grid_b, mu_c, ds, tol, floor=floor)
        robot[:2] += ds * d
        dpos = xi[:2] * ds
        tee = tee + np.array([dpos[0], dpos[1], xi[2] * ds])
        path += float(np.hypot(*dpos))
        traj.append(tee.copy())
    traj = np.array(traj)
    dyaw = math.degrees(math.atan2(math.sin(tee[2] - yaw0), math.cos(tee[2] - yaw0)))
    return {"deg_per_m": dyaw / max(path, 1e-9), "path": path, "dyaw": dyaw, "traj": traj,
            "T/pusher": path / push_len}


TEE_EXT = tee_exterior_edges()
TEE_VERTS = tee_outline_vertices()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--ds", type=float, default=0.01, help="pusher travel per step [m]")
    ap.add_argument("--push", type=float, default=1.35, help="pusher travel in contact [m]")
    ap.add_argument("--mu-c", default="0,0.1,0.3")
    ap.add_argument("--out", default="out/sysid/reference.md")
    ap.add_argument("--viscous", action="store_true", help="also the viscous floor swarp uses, at mu_c = 0")
    args = ap.parse_args()
    exps = [e for e in push_experiments() if e.name.startswith("bar_") and e.name != "bar_pair"]
    mus = [float(m) for m in args.mu_c.split(",")]
    rows = []
    for e in exps:
        row = [e.name]
        for mu in mus:
            r = simulate(e, mu, args.ds, args.cell, args.push)
            row.append(f"{r['deg_per_m']:+.1f} ({r['dyaw']:+.0f}° over {r['path']:.2f} m)")
            np.save(os.path.join(os.path.dirname(args.out), f"reference_{e.name}_mu{mu}.npy"), r["traj"])
            print(e.name, mu, row[-1], flush=True)
        rows.append(row)
    if args.viscous:
        for row, e in zip(rows, exps, strict=True):
            r = simulate(e, 0.0, args.ds, args.cell, args.push, floor="viscous")
            row.append(f"{r['deg_per_m']:+.1f} ({r['dyaw']:+.0f}° over {r['path']:.2f} m)")
            print(e.name, "viscous", row[-1], flush=True)
    head = ["exp"] + [f"μ_c = {m:g}: T rot [deg/m] (total)" for m in mus]
    if args.viscous:
        head.append("viscous floor, μ_c = 0 (swarp's model)")
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    text = (
        f"# Quasi-static pushing reference\n\nUniform-pressure Coulomb floor ({args.cell} m grid), "
        f"velocity-controlled flat pusher, {args.push} m of pusher travel from first contact, step {args.ds} m.\n\n"
        + "\n".join(md) + "\n"
    )
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
