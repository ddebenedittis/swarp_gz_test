"""Tables of swarp vs Gazebo from ``out/sysid/{swarp,gz}_*.npz`` (numpy only).

    python3 -m pusht_policy.sysid.compare out/sysid      # writes out/sysid/report.md
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np

from pusht_policy.sysid.experiments import DT, Experiment, by_name

HIT_AGENT, HIT_MOVABLE = 1, 3


def load(d: str, sim: str, name: str) -> dict | None:
    p = os.path.join(d, f"{sim}_{name}.npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def body_vel(r: dict, i: int):
    """Forward and lateral body-frame speed, yaw rate of robot ``i`` on the record grid."""
    yaw = r["robots"][:, i, 2]
    v = r["robots_vel"][:, i]
    fwd = v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw)
    lat = -v[:, 0] * np.sin(yaw) + v[:, 1] * np.cos(yaw)
    return fwd, lat, r["robots_w"][:, i]


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def ratio(a, b):
    return a / b if (a is not None and b not in (None, 0) and math.isfinite(a) and math.isfinite(b)) else float("nan")


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------- robot alone


def on_steps(e: Experiment, i: int = 0) -> int:
    nz = np.flatnonzero(np.abs(e.cmds[:, i]).sum(-1) > 0)
    return int(nz[-1]) + 1 if len(nz) else 0


def robot_metrics(e: Experiment, r: dict, raw: np.ndarray | None) -> dict:
    n_on = on_steps(e)
    fwd, lat, w = body_vel(r, 0)
    ss = slice(max(1, n_on - 9), n_on + 1)  # the last second of the command
    v_cmd, w_cmd = e.cmds[0, 0]
    out = {"v_ss": float(fwd[ss].mean()), "w_ss": float(w[ss].mean())}
    out["v_ratio"] = ratio(out["v_ss"], v_cmd)
    out["w_ratio"] = ratio(out["w_ss"], w_cmd)
    out["radius"] = ratio(out["v_ss"], out["w_ss"]) if w_cmd else float("nan")
    # rise: time from the first command to 90% of the steady value (50 Hz in Gazebo)
    if raw is not None:
        t = raw[:, 0]
        yaw = raw[:, 3]
        sig = raw[:, 4] * np.cos(yaw) + raw[:, 5] * np.sin(yaw) if v_cmd else raw[:, 6]
    else:
        t = r["t"]
        sig = fwd if v_cmd else w
    target = out["v_ss"] if v_cmd else out["w_ss"]
    hit = np.flatnonzero((t >= 0) & (np.sign(target) * sig >= 0.9 * abs(target)))
    out["t90"] = float(t[hit[0]]) if len(hit) else float("nan")
    p = r["robots"][:, 0, :2]
    out["stop_dist"] = float(np.hypot(*(p[-1] - p[n_on]))) if n_on < len(p) - 1 else float("nan")
    if e.name.startswith("reverse"):
        n_sw = int(round(2.0 / DT))
        if raw is not None:
            k = np.flatnonzero((t >= n_sw * DT) & (sig <= -0.9 * 0.3))
            out["t_reverse"] = float(t[k[0]] - n_sw * DT) if len(k) else float("nan")
        else:
            k = np.flatnonzero((r["t"] >= n_sw * DT) & (fwd <= -0.27))
            out["t_reverse"] = float(r["t"][k[0]] - n_sw * DT) if len(k) else float("nan")
    return out


# --------------------------------------------------------------------- push


def push_metrics(e: Experiment, r: dict) -> dict:
    n_on = on_steps(e)
    tee = r["tee"]
    tspd = np.hypot(*r["tee_vel"].T)
    pushers = [i for i in range(2) if np.abs(e.cmds[:, i]).sum() > 0]
    contact = np.zeros(len(tee), bool)
    contact[1 : n_on + 1] = tspd[1 : n_on + 1] > 0.02
    out = {"onset": float(r["t"][np.argmax(contact)]) if contact.any() else float("nan")}
    if not contact.any():
        return out
    k0, k1 = np.flatnonzero(contact)[[0, -1]]
    rspd = np.mean([np.hypot(*r["robots_vel"][contact, i].T).mean() for i in pushers])
    out["T_speed"] = float(tspd[contact].mean())
    out["T/robot"] = float(tspd[contact].mean() / max(rspd, 1e-9))
    out["T/cmd"] = float(tspd[contact].mean() / e.meta["v"])
    path = float(np.hypot(*np.diff(tee[k0 : k1 + 1, :2], axis=0).T).sum())
    dyaw = float(wrap(tee[k1, 2] - tee[k0, 2]))
    out["T_path"] = path
    out["T_rot_deg_per_m"] = float(np.degrees(dyaw) / max(path, 1e-9))
    out["robot_dheading_deg"] = float(np.mean([np.degrees(wrap(r["robots"][k1, i, 2] - r["robots"][k0, i, 2])) for i in pushers]))
    out["robot_lat_slip"] = float(np.mean([np.abs(body_vel(r, i)[1][contact]).mean() for i in pushers]))
    out["T_stop_dist"] = float(np.hypot(*(tee[-1, :2] - tee[n_on, :2])))
    out["T_final"] = (float(tee[-1, 0] - tee[0, 0]), float(tee[-1, 1] - tee[0, 1]), float(np.degrees(wrap(tee[-1, 2] - tee[0, 2]))))
    return out


# --------------------------------------------------------------------- lidar


def lidar_metrics(s: dict, g: dict) -> dict:
    cs, cg = s["cls"].astype(int), g["cls"].astype(int)
    rs, rg = s["rays"], g["rays"]
    both_t = (cs == HIT_MOVABLE) & (cg == HIT_MOVABLE)
    out = {
        "agree": float((cs == cg).mean()),
        "T_rays": f"{int((cs == HIT_MOVABLE).sum())}/{int((cg == HIT_MOVABLE).sum())}",
        "near_T_swarp": float(rs[cs == HIT_MOVABLE].min()) if (cs == HIT_MOVABLE).any() else float("nan"),
        "near_T_gz": float(rg[cg == HIT_MOVABLE].min()) if (cg == HIT_MOVABLE).any() else float("nan"),
        "range_mae": float(np.abs(rs[both_t] - rg[both_t]).mean()) if both_t.any() else float("nan"),
        "missed_rays": " ".join(str(k) for k in np.flatnonzero((cs == HIT_MOVABLE) & (cg != HIT_MOVABLE))),
    }
    return out


# --------------------------------------------------------------------- paired episodes


def paired(d: str) -> str:
    p = os.path.join(d, "swarp_paired.npz")
    if not os.path.exists(p):
        return ""
    s = np.load(p)
    dt = float(s["dt"])
    rows, errs = [], {5: [], 10: [], 20: []}
    for i in range(len(s["seed"])):
        rec = os.path.join(str(s["run_dir"][i]), "rec", f"ep{int(s['episode'][i]):04d}_robot_1.npz")
        if not os.path.exists(rec):
            continue
        g = np.load(rec)
        gt = g["tee"]
        n_g = len(gt)  # gz steps (stops at success or the horizon)
        row = [f"{int(s['seed'][i])}/{int(s['episode'][i])}"]
        st = int(s["solved_at"][i])
        row += [f"{st * dt:.1f}" if st > 0 else "fail", f"{n_g * dt:.1f}" if n_g < 900 else "fail"]
        for sec in (5, 10, 20):
            k = int(round(sec / dt))
            if k < n_g:
                # gz record k = the obs at step k = the state after k steps
                e_pos = float(np.hypot(*(s["tee"][i, k, :2] - gt[k, :2])))
                e_yaw = float(np.degrees(abs(wrap(s["tee"][i, k, 2] - gt[k, 2]))))
                errs[sec].append(e_pos)
                row.append(f"{e_pos:.2f} m / {e_yaw:.0f}°")
            else:
                row.append("ended")
        rows.append(row)
    summ = ", ".join(f"{sec}s median {np.median(v):.2f} m (n={len(v)})" for sec, v in errs.items() if v)
    return (
        "## 4. Paired episodes (swarp greedy from the Gazebo episodes' initial states)\n\n"
        + table(["seed/ep", "swarp solve [s]", "gz solve [s]", "T err @5s", "T err @10s", "T err @20s"], rows)
        + f"\n\nT position divergence: {summ}.\n"
    )


# --------------------------------------------------------------------- report


def main(argv=None) -> None:
    d = (argv or sys.argv[1:] or ["out/sysid"])[0]
    exps = by_name()
    md = ["# swarp vs Gazebo sysid\n"]

    rows = []
    for e in [x for x in exps.values() if x.group == "robot"]:
        s, g = load(d, "swarp", e.name), load(d, "gz", e.name)
        if s is None or g is None:
            continue
        ms, mg = robot_metrics(e, s, None), robot_metrics(e, g, g.get("raw_robot_1"))
        v, w = e.cmds[0, 0]
        rows.append([
            e.name, f"{v:g}/{w:g}",
            f"{fmt(ms['v_ss'])} / {fmt(mg['v_ss'])}", f"{fmt(ms['w_ss'])} / {fmt(mg['w_ss'])}",
            fmt(ratio(mg["v_ss"], ms["v_ss"]) if v else ratio(mg["w_ss"], ms["w_ss"])),
            f"{fmt(ms['t90'], 2)} / {fmt(mg['t90'], 2)}",
            f"{fmt(ms['radius'], 2)} / {fmt(mg['radius'], 2)}" if w else "",
            f"{fmt(ms['stop_dist'])} / {fmt(mg['stop_dist'])}",
            f"{fmt(ms.get('t_reverse'), 2)} / {fmt(mg.get('t_reverse'), 2)}" if "t_reverse" in ms else "",
        ])
    md.append("## 1. Robot alone (swarp / gz)\n\n" + table(
        ["exp", "cmd v/w", "v_ss", "w_ss", "gz/swarp", "t90 [s]", "radius [m]", "stop dist [m]", "t reverse [s]"], rows
    ) + "\n\nswarp samples at DT = 0.1 s, so its t90 = 0.10 means within one step (it is instantaneous); Gazebo's is from the 50 Hz ground truth.\n")

    rows = []
    for e in [x for x in exps.values() if x.group == "push"]:
        s, g = load(d, "swarp", e.name), load(d, "gz", e.name)
        if s is None or g is None:
            continue
        ms, mg = push_metrics(e, s), push_metrics(e, g)
        def pair(k, nd=3):
            return f"{fmt(ms.get(k), nd)} / {fmt(mg.get(k), nd)}"
        fin = lambda m: "({:+.2f}, {:+.2f}, {:+.0f}°)".format(*m["T_final"]) if "T_final" in m else "n/a"  # noqa: E731
        ref = os.path.join(d, f"reference_{e.name}_mu0.1.npy")
        if os.path.exists(ref):
            tr = np.load(ref)
            path = float(np.hypot(*np.diff(tr[:, :2], axis=0).T).sum())
            ref_rot = fmt(float(np.degrees(wrap(tr[-1, 2] - tr[0, 2]))) / max(path, 1e-9), 1)
        else:
            ref_rot = "n/a"
        rows.append([
            e.name, pair("onset", 1), pair("T/cmd", 2), pair("T/robot", 2),
            f"{pair('T_rot_deg_per_m', 1)} / {ref_rot}",
            pair("robot_dheading_deg", 1), pair("robot_lat_slip", 3), pair("T_stop_dist", 3),
            f"{fin(ms)} / {fin(mg)}",
        ])
    md.append("## 2. Pushing (swarp / gz), contact = T speed > 0.02 m/s while commanded\n\n" + table(
        ["exp", "contact at [s]", "T speed / cmd", "T speed / robot speed", "T rot [deg/m] (swarp / gz / quasi-static ref)", "robot Δheading [deg]",
         "robot lateral slip [m/s]", "T stop dist [m]", "T final (dx, dy, dyaw)"], rows
    ) + "\n\nThe quasi-static reference (`scripts/pushing_reference.py`): uniform-pressure Coulomb floor, ideal velocity-controlled flat pusher, robot-T friction 0.1 (gz's min(0.1, 0.3)), 1.35 m of pusher travel.\n")

    rows = []
    for e in [x for x in exps.values() if x.group == "lidar"]:
        s, g = load(d, "swarp", e.name), load(d, "gz", e.name)
        if s is None or g is None:
            continue
        m = lidar_metrics(s, g)
        rows.append([
            f"{e.meta['beta']}°", e.meta["gap"], fmt(m["agree"], 2), m["T_rays"],
            f"{fmt(m['near_T_swarp'])} / {fmt(m['near_T_gz'])}", fmt(m["range_mae"]), m["missed_rays"] or "-",
        ])
    md.append("## 3. Lidar against the T (32 rays, robot 1 at bearing β from the bar face)\n\n" + table(
        ["bearing", "gap [m]", "class agreement", "T rays swarp/gz", "nearest T swarp / gz [m]",
         "range MAE (both T) [m]", "rays T in swarp, not in gz"], rows
    ) + "\n\nRay k points at k·11.25° CCW from the heading; bearing β puts the face at ray β/11.25.\n")

    rows = []
    for e in [x for x in exps.values() if x.group == "contact"]:
        s, g = load(d, "swarp", e.name), load(d, "gz", e.name)
        if s is None or g is None:
            continue
        n_on = on_steps(e)
        def cm(r):
            out = []
            for i in [i for i in range(2) if np.abs(e.cmds[:, i]).sum() > 0]:
                dyaw = float(np.degrees(np.unwrap(r["robots"][: n_on + 1, i, 2])[-1] - r["robots"][0, i, 2]))
                disp = float(np.hypot(*(r["robots"][n_on, i, :2] - r["robots"][0, i, :2])))
                out.append(f"{dyaw:+.0f}° / {disp:.2f} m")
            tee = r["tee"]
            out.append(f"{np.hypot(*(tee[n_on, :2] - tee[0, :2])):.2f} m, {np.degrees(wrap(tee[n_on, 2] - tee[0, 2])):+.0f}°")
            return out
        ideal = np.degrees(e.meta["w"] * n_on * DT)
        rows.append([e.name, f"{ideal:+.0f}°", " ; ".join(cm(s)), " ; ".join(cm(g))])
    if rows:
        md.append("## 5. Turning while pressed against the T (per robot: Δyaw / displacement ; T: displacement, Δyaw)\n\n" + table(
            ["exp", "commanded Δyaw", "swarp", "gz"], rows) + "\n")

    md.append(paired(d))
    text = "\n".join(md)
    with open(os.path.join(d, "report.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
