"""Quasi-static reference for a skid-steer Limo turning while pressed against the T (host, cvxpy).

What a real robot does when told ``(v, w)`` flush against the T, with no simulator:

* each robot is a rigid box on four wheels at (+-0.1, +-0.086) m; a wheel's contact patch
  moves at its commanded surface speed ``v -+ w * track / 2`` along the heading, and any
  slip between the patch and the floor costs anisotropic Coulomb friction
  ``N * |(mu_long * s_long, mu_lat * s_lat)|`` with ``N = m g / 4``;
* the T slides on a uniform-pressure Coulomb floor (as ``pushing_reference.py``);
* robot-T contacts are unilateral and frictionless (corners against edges, vertices against
  edges, from ``pushing_reference.contacts``).

Each time step solves for every body's twist by the minimum-power principle: the motion
that minimises the total friction power subject to non-penetration. The robots' motors are
ideal velocity sources at the wheels, so a robot that cannot turn shows up as wheels
slipping, not as a stalled motor (a real motor's torque limit would only lock it sooner).

    python3 src/pusht_policy/scripts/pressed_turn_reference.py --out out/sysid/contact/reference.md
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import cvxpy as cp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import pushing_reference as pr  # noqa: E402
from pusht_policy.sysid.experiments import contact_experiments  # noqa: E402

G = 9.81
WHEELS = [(0.1, 0.086, +1), (-0.1, 0.086, +1), (0.1, -0.086, -1), (-0.1, -0.086, -1)]  # x, y, left=+1
TRACK = 0.172


def run(exp, dt, duration, cell, robot_mass, tee_mass, mu_long, mu_lat, mu_tee, tol=0.003):
    robots = [np.array(r, float) for r in exp.robots]
    active = [i for i in range(len(robots)) if np.abs(exp.cmds[:, i]).sum() > 0]
    tee = np.array(exp.tee, float)
    grid_b = pr.tee_grid(cell)
    p_tee = mu_tee * tee_mass * G / len(grid_b)
    n_wheel = robot_mass * G / 4.0
    v_cmd, w_cmd = exp.cmds[0, active[0]]
    hist = []
    for _ in range(int(round(duration / dt))):
        x = {i: cp.Variable(3) for i in active}  # (vx, vy, w) world, about the robot centre
        xt = cp.Variable(3)  # the T, about its origin
        power = []
        for i in active:
            th = robots[i][2]
            c, s = math.cos(th), math.sin(th)
            h, hp = np.array([c, s]), np.array([-s, c])
            for wx, wy, side in WHEELS:
                r = np.array([c * wx - s * wy, s * wx + c * wy])
                vp = cp.hstack([x[i][0] - x[i][2] * r[1], x[i][1] + x[i][2] * r[0]])
                u = v_cmd - side * w_cmd * TRACK / 2  # left wheel slower for +w
                slip = cp.hstack([mu_long * (vp @ h - u), mu_lat * (vp @ hp)])
                power.append(n_wheel * cp.norm(slip, 2))
        off = grid_b @ pr.rot(tee[2]).T
        vx = xt[0] - xt[2] * off[:, 1]
        vy = xt[1] + xt[2] * off[:, 0]
        power.append(p_tee * cp.sum(cp.norm(cp.vstack([vx, vy]), 2, axis=0)))
        cons = []
        for i in active:
            for cpt, n, gap in pr.contacts(tee, robots[i], tol):
                rt = cpt - tee[:2]
                rr = cpt - robots[i][:2]
                vt = cp.hstack([xt[0] - xt[2] * rt[1], xt[1] + xt[2] * rt[0]])
                vr = cp.hstack([x[i][0] - x[i][2] * rr[1], x[i][1] + x[i][2] * rr[0]])
                cons.append((vt - vr) @ n >= -gap / dt)
        prob = cp.Problem(cp.Minimize(sum(power)), cons)
        prob.solve(solver=cp.CLARABEL)
        force = sum(float(k.dual_value) for k in cons) if cons else 0.0
        for i in active:
            robots[i] = robots[i] + dt * np.asarray(x[i].value)
        tee = tee + dt * np.asarray(xt.value)
        hist.append((robots[active[0]].copy(), tee.copy(), force, [float(x[i].value[2]) for i in active]))
    w = np.mean([np.mean(h[3]) for h in hist[len(hist) // 2:]])  # steady half
    r0 = np.array(exp.robots[active[0]], float)
    rf, tf = hist[-1][0], hist[-1][1]
    return {
        "w_ss": w, "ratio": w / w_cmd,
        "robot_disp": float(np.hypot(*(rf[:2] - r0[:2]))),
        "tee_disp": float(np.hypot(*(tf[:2] - np.array(exp.tee[:2])))),
        "tee_dyaw": math.degrees(tf[2] - exp.tee[2]),
        "force": float(np.mean([h[2] for h in hist[len(hist) // 2:]])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--cell", type=float, default=0.04)
    ap.add_argument("--robot-mass", type=float, default=4.2)
    ap.add_argument("--tee-mass", type=float, default=0.5)
    ap.add_argument("--mu-long", type=float, default=1.0, help="wheel friction along the heading (gz: 1.0)")
    ap.add_argument("--mu-lat", type=float, default=0.2, help="wheel friction across it (gz: 0.2)")
    ap.add_argument("--mu-tee", type=float, default=0.3)
    ap.add_argument("--out", default="out/sysid/contact/reference.md")
    args = ap.parse_args()
    rows = []
    for e in contact_experiments():
        r = run(e, args.dt, args.duration, args.cell, args.robot_mass, args.tee_mass,
                args.mu_long, args.mu_lat, args.mu_tee)
        row = [e.name, f"{r['w_ss']:+.2f} ({r['ratio']:.2f}×)", f"{r['robot_disp']:.2f}",
               f"{r['tee_disp']:.2f} m {r['tee_dyaw']:+.0f}°", f"{r['force']:.1f}"]
        print(" | ".join(row), flush=True)
        rows.append(row)
    head = ["exp", "steady yaw rate [rad/s]", "robot 1 displacement [m]", "T", "contact force, steady [N]"]
    md = "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])
    text = (
        "# Quasi-static reference: turning flush against the T\n\n"
        f"Robots {args.robot_mass} kg on 4 wheels (mu {args.mu_long} along, {args.mu_lat} across), "
        f"T {args.tee_mass} kg on a uniform Coulomb floor (mu {args.mu_tee}), frictionless robot-T contact, "
        f"{args.duration} s of the command, steady values over the second half.\n\n" + md + "\n"
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
