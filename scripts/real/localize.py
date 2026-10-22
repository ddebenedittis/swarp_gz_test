"""Initial localization of the two Limos on a saved map, the T's pose, and the arena frame.

Runs in the Humble laptop container (scripts/real/laptop.bash py scripts/real/localize.py CMD ...):

    scan [--n 20]          median of N scans per robot -> STATE/scan_<robot>.npz
    match                  global scan-to-map match per robot (likelihood field) -> STATE/poses.json, STATE/overlay.png
    tee [--poses amcl]     fit the checkpoint's T to the returns that are on neither the map nor a robot,
                           propose the arena (centre on the corridor axis through the T, x along the walls)
                           -> STATE/tee.json, STATE/arena.json; prints the arena poses and wall clearances
    args                   the strings for laptop.bash amcl: "x,y,yaw;x,y,yaw" (rad) and map_to_arena
    seed [--poses amcl]    publish /robot_i/initialpose (poses.json, or the current AMCL estimate), then check
                           each AMCL answered with a fresh pose (alive) -> exit 1 if not
    spin ROBOT [--w 0.5]   360 deg in place (closed loop on odometry), 0.12 m forward and back, stop;
                           through robot_bridge (start the robot with robots.bash start slow|live)
    check                  AMCL pose (map, arena) and 1-sigma per robot
    verify                 fresh scans, local match around each AMCL estimate: the disagreement in cm / deg

Poses are base_footprint in the map frame, yaw in degrees in the JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
ROBOTS = ("robot_1", "robot_2")
MOUNT_X = 0.103          # lidar ahead of base_footprint (limo_bridge laser_xyz)
SELF_R = 0.15            # returns closer than this are the robot itself
ROBOT_HALF = (0.161, 0.110)


# ------------------------------------------------------------------ map

def load_map(path: str):
    import yaml
    from PIL import Image
    from scipy import ndimage

    meta = yaml.safe_load(open(path))
    img = np.array(Image.open(os.path.join(os.path.dirname(path), meta["image"])).convert("L"), dtype=float)
    p = img / 255.0 if meta.get("negate", 0) else (255.0 - img) / 255.0
    occ = p > meta.get("occupied_thresh", 0.65)
    free = p < meta.get("free_thresh", 0.25)
    res = float(meta["resolution"])
    ox, oy = float(meta["origin"][0]), float(meta["origin"][1])
    dist = ndimage.distance_transform_edt(~occ) * res
    return dict(occ=occ, free=free, res=res, ox=ox, oy=oy, H=img.shape[0], W=img.shape[1], dist=dist,
                lik=np.exp(-dist ** 2 / (2 * 0.08 ** 2)))


def cells(m, x, y):
    col = np.floor((x - m["ox"]) / m["res"]).astype(int)
    row = (m["H"] - 1 - np.floor((y - m["oy"]) / m["res"])).astype(int)
    ok = (col >= 0) & (col < m["W"]) & (row >= 0) & (row < m["H"])
    return row.clip(0, m["H"] - 1), col.clip(0, m["W"] - 1), ok


def scan_points(state: Path, robot: str, max_r: float = 11.0):
    """Robot-frame (base_footprint) points of the median scan, self-returns dropped."""
    d = np.load(state / f"scan_{robot}.npz")
    r, a = d["ranges"], d["angle_min"] + np.arange(d["ranges"].size) * d["angle_increment"]
    ok = np.isfinite(r) & (r > SELF_R) & (r < max_r)
    return MOUNT_X + r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])


def to_world(pose, lx, ly):
    x, y, t = pose
    return x + math.cos(t) * lx - math.sin(t) * ly, y + math.sin(t) * lx + math.cos(t) * ly


def score(m, pose, lx, ly):
    wx, wy = to_world(pose, lx, ly)
    r, c, ok = cells(m, wx, wy)
    return float(np.where(ok, m["lik"][r, c], 0.0).mean())


def refine(m, pose, lx, ly, dxy=0.15, dth=3.0):
    best = (score(m, pose, lx, ly), *pose)
    for step_xy, step_th in ((dxy / 6, math.radians(dth / 6)), (dxy / 24, math.radians(dth / 24))):
        x0, y0, t0 = best[1:]
        for t in t0 + step_th * np.arange(-6, 7):
            for x in x0 + step_xy * np.arange(-6, 7):
                for y in y0 + step_xy * np.arange(-6, 7):
                    s = score(m, (x, y, t), lx, ly)
                    if s > best[0]:
                        best = (s, x, y, t)
    return best


def global_match(m, lx, ly):
    fy, fx = np.nonzero(m["free"][::2, ::2])
    fy, fx = fy * 2, fx * 2
    # candidate cells at least 0.15 m from any wall (a robot stands there)
    keep = m["dist"][fy, fx] > 0.15
    cx = m["ox"] + (fx[keep] + 0.5) * m["res"]
    cy = m["oy"] + (m["H"] - fy[keep] - 0.5) * m["res"]
    sub = np.linspace(0, lx.size - 1, min(lx.size, 120)).astype(int)
    sx, sy = lx[sub], ly[sub]
    cands = []
    for t in np.radians(np.arange(0, 360, 2.0)):
        c, s = math.cos(t), math.sin(t)
        wx = cx[:, None] + c * sx[None] - s * sy[None]
        wy = cy[:, None] + s * sx[None] + c * sy[None]
        r, cc, ok = cells(m, wx, wy)
        v = np.where(ok, m["lik"][r, cc], 0.0).mean(1)
        for i in np.argsort(v)[-3:]:
            cands.append((float(v[i]), float(cx[i]), float(cy[i]), float(t)))
    cands.sort(reverse=True)
    refined = sorted((refine(m, c[1:], lx, ly) for c in cands[:8]), reverse=True)
    best = refined[0]
    # a second, distinct hypothesis that scores nearly as well means the match is ambiguous
    rival = next((r for r in refined[1:] if math.hypot(r[1] - best[1], r[2] - best[2]) > 0.3
                  or abs(math.remainder(r[3] - best[3], 2 * math.pi)) > math.radians(10)), None)
    return best, rival


# ------------------------------------------------------------------ ROS helpers

def ros_node(name: str):
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    return rclpy.create_node(name)


def spin_for(node, cond, timeout):
    import rclpy

    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
    return cond()


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def amcl_poses(timeout=4.0):
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    n = ros_node("localize_amcl")
    got = {}
    for r in ROBOTS:
        for q in (QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE), 10):
            n.create_subscription(PoseWithCovarianceStamped, f"/{r}/amcl_pose", lambda m, r=r: got.__setitem__(r, m), q)
    spin_for(n, lambda: len(got) == len(ROBOTS), timeout)
    out = {}
    for r, m in got.items():
        p, c = m.pose.pose, m.pose.covariance
        st = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        out[r] = dict(x=p.position.x, y=p.position.y, yaw=math.degrees(yaw_of(p.orientation)),
                      sx=math.sqrt(max(c[0], 0)), sy=math.sqrt(max(c[7], 0)), syaw=math.degrees(math.sqrt(max(c[35], 0))),
                      age=time.time() - st)
    n.destroy_node()
    return out


# ------------------------------------------------------------------ commands

def cmd_scan(a):
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan

    n = ros_node("localize_scan")
    buf = {r: [] for r in ROBOTS}
    meta = {}

    def on(r, m):
        if len(buf[r]) < a.n:
            buf[r].append(np.array(m.ranges, dtype=float))
            meta[r] = (m.angle_min, m.angle_increment)

    for r in ROBOTS:
        n.create_subscription(LaserScan, f"/{r}/scan", lambda m, r=r: on(r, m), qos_profile_sensor_data)
    spin_for(n, lambda: all(len(v) >= a.n for v in buf.values()), 30.0)
    for r in ROBOTS:
        if not buf[r]:
            sys.exit(f"{r}: no /{r}/scan (robot stack up? DDS?)")
        R = np.vstack([b for b in buf[r] if b.size == buf[r][0].size])
        R[(R <= 0) | ~np.isfinite(R)] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(R, axis=0)
        np.savez(a.state / f"scan_{r}.npz", ranges=med, angle_min=meta[r][0], angle_increment=meta[r][1])
        print(f"{r}: {len(buf[r])} scans, {np.isfinite(med).sum()} beams with returns")


def cmd_match(a):
    m = load_map(a.map)
    poses = {}
    for r in ROBOTS:
        lx, ly = scan_points(a.state, r)
        (s, x, y, t), rival = global_match(m, lx, ly)
        poses[r] = dict(x=x, y=y, yaw=math.degrees(t), score=s)
        warn = ""
        if s < 0.6:
            warn += "  LOW SCORE: drive/rotate the robot or check the map"
        if rival and rival[0] > s - 0.03:
            warn += f"  AMBIGUOUS: also ({rival[1]:.2f}, {rival[2]:.2f}, {math.degrees(rival[3]):.0f} deg) scores {rival[0]:.3f}"
        print(f"{r}: map ({x:.3f}, {y:.3f}, {math.degrees(t):.1f} deg)  score {s:.3f} ({lx.size} beams){warn}")
    json.dump(poses, open(a.state / "poses.json", "w"), indent=1)
    overlay(a, m, poses)


def current_poses(a):
    if a.poses == "amcl":
        am = amcl_poses()
        return {r: dict(x=v["x"], y=v["y"], yaw=v["yaw"]) for r, v in am.items()}
    return json.load(open(a.state / "poses.json"))


def offmap_points(a, m, poses):
    pts = []
    for r in ROBOTS:
        p = poses[r]
        lx, ly = scan_points(a.state, r, max_r=4.0)
        wx, wy = to_world((p["x"], p["y"], math.radians(p["yaw"])), lx, ly)
        rr, cc, ok = cells(m, wx, wy)
        off = ok & (m["dist"][rr, cc] > 0.25)
        for q in ROBOTS:   # drop the other robots' bodies
            off &= np.hypot(wx - poses[q]["x"], wy - poses[q]["y"]) > 0.3
        pts.append(np.c_[wx[off], wy[off]])
    return np.vstack(pts)


def wall_direction(m):
    """Dominant wall orientation of the map (mod 90 deg), from the occupied cells' gradients."""
    from scipy import ndimage

    o = ndimage.gaussian_filter(m["occ"].astype(float), 1.0)
    gx, gy = ndimage.sobel(o, 1), -ndimage.sobel(o, 0)
    mag = np.hypot(gx, gy)
    th = np.arctan2(gy, gx)[mag > 0.1 * mag.max()]
    w = mag[mag > 0.1 * mag.max()]
    # 4*theta folds the four wall normals together
    z = np.sum(w * np.exp(4j * th))
    return math.remainder(np.angle(z) / 4, math.pi / 2)


def cmd_tee(a):
    from pusht_policy.task import Task

    m = load_map(a.map)
    poses = current_poses(a)
    P = offmap_points(a, m, poses)
    if len(P) < 10:
        sys.exit(f"only {len(P)} returns off the map: is the T within ~4 m and seen by a robot?")
    task = Task.from_checkpoint(a.checkpoint)
    boxes = list(zip(task.tee_off, task.tee_half))
    # the biggest cluster (the T), seeded at the densest point
    d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    seed = P[np.argmax((d < 0.4).sum(1))]
    P = P[np.hypot(P[:, 0] - seed[0], P[:, 1] - seed[1]) < 0.5]
    c0 = P.mean(0)

    def outline_dist(px, py):
        ds = []
        for (cx, cy), (hx, hy) in boxes:
            qx, qy = np.abs(px - cx) - hx, np.abs(py - cy) - hy
            ds.append(np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0))
        return np.abs(np.minimum.reduce(ds))

    best = None
    for th in np.radians(np.arange(0, 360, 1.0)):
        cs, sn = math.cos(th), math.sin(th)
        for x in c0[0] + np.arange(-0.35, 0.351, 0.02):
            for y in c0[1] + np.arange(-0.35, 0.351, 0.02):
                dx, dy = P[:, 0] - x, P[:, 1] - y
                e = np.minimum(outline_dist(cs * dx + sn * dy, -sn * dx + cs * dy), 0.1).mean()
                if best is None or e < best[0]:
                    best = (e, x, y, th)
    e, tx, ty, tth = best
    tee = dict(x=tx, y=ty, yaw=math.degrees(tth), residual_cm=e * 100, points=len(P))
    print(f"T: map ({tx:.3f}, {ty:.3f}, {math.degrees(tth):.1f} deg), {len(P)} points, mean residual {e * 100:.1f} cm"
          + ("  POOR FIT: check the overlay" if e > 0.03 else ""))
    json.dump(tee, open(a.state / "tee.json", "w"), indent=1)

    # Arena: x along the walls, centred between the walls on the normal through the T.
    ax = wall_direction(m)
    if a.arena_yaw is not None:
        ax = math.radians(a.arena_yaw)
    nx, ny = -math.sin(ax), math.cos(ax)

    def wall_along(sign):
        for k in range(1, 200):
            r, c, ok = cells(m, np.array([tx + sign * nx * k * m["res"]]), np.array([ty + sign * ny * k * m["res"]]))
            if not ok[0] or m["occ"][r[0], c[0]]:
                return k * m["res"]
        return 10.0

    up, down = wall_along(1), wall_along(-1)
    off = (up - down) / 2
    cx, cy = tx + nx * off, ty + ny * off
    if a.arena_centre:
        cx, cy = (float(v) for v in a.arena_centre.split(","))
    half_w = (up + down) / 2
    co, so = math.cos(ax), math.sin(ax)
    m2a = (-co * cx - so * cy, so * cx - co * cy, -ax)
    arena = dict(centre=[cx, cy, math.degrees(ax)], map_to_arena=list(m2a), wall_half_width=half_w)
    json.dump(arena, open(a.state / "arena.json", "w"), indent=1)

    def in_arena(x, y, yaw):
        dx, dy = x - cx, y - cy
        return co * dx + so * dy, -so * dx + co * dy, math.degrees(math.remainder(math.radians(yaw) - ax, 2 * math.pi))

    print(f"arena: centre map ({cx:.3f}, {cy:.3f}), x along the walls at {math.degrees(ax):.1f} deg; "
          f"walls at +-{half_w:.2f} m across  -> map_to_arena {m2a[0]:.3f},{m2a[1]:.3f},{m2a[2]:.4f}")
    print(f"  T       arena ({'%.2f, %.2f, %.0f deg' % in_arena(tx, ty, math.degrees(tth))})")
    for r in ROBOTS:
        p = poses[r]
        print(f"  {r} arena ({'%.2f, %.2f, %.0f deg' % in_arena(p['x'], p['y'], p['yaw'])})")
    if half_w < 2.5:
        print(f"  NOTE: the walls are {half_w:.2f} m from the axis, inside the policy's 2.5 m label bound: a robot within "
              f"1.5 m of a wall sees it as the T. Keep the T and the goal near the axis (|y| small).")
    overlay(a, m, poses, tee=(tx, ty, tth), boxes=boxes, arena=(cx, cy, ax))


def cmd_args(a):
    p = json.load(open(a.state / "poses.json"))
    try:
        am = amcl_poses(2.0) if a.poses == "amcl" else {}
    except Exception:
        am = {}
    src = am if len(am) == len(ROBOTS) else p
    print(";".join(f"{src[r]['x']:.3f},{src[r]['y']:.3f},{math.radians(src[r]['yaw']):.4f}" for r in ROBOTS))
    ar = json.load(open(a.state / "arena.json")) if (a.state / "arena.json").exists() else None
    print(",".join(f"{v:.4f}" for v in ar["map_to_arena"]) if ar else "0,0,0")


def cmd_seed(a):
    from geometry_msgs.msg import PoseWithCovarianceStamped

    poses = current_poses(a)
    n = ros_node("localize_seed")
    t_sent = time.time()
    for r in ROBOTS:
        p = poses[r]
        pub = n.create_publisher(PoseWithCovarianceStamped, f"/{r}/initialpose", 10)
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = n.get_clock().now().to_msg()
        msg.pose.pose.position.x, msg.pose.pose.position.y = p["x"], p["y"]
        y = math.radians(p["yaw"])
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(y / 2), math.cos(y / 2)
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.05 ** 2
        msg.pose.covariance[35] = math.radians(3) ** 2
        spin_for(n, lambda: pub.get_subscription_count() > 0, 10.0)
        for _ in range(3):
            pub.publish(msg)
            spin_for(n, lambda: False, 0.2)
        print(f"{r}: initialpose ({p['x']:.3f}, {p['y']:.3f}, {p['yaw']:.1f} deg)")
    n.destroy_node()
    time.sleep(2.5)
    am = amcl_poses()
    dead = [r for r in ROBOTS if r not in am or am[r]["age"] > time.time() - t_sent + 0.5]
    for r in ROBOTS:
        if r in am:
            print(f"{r}: AMCL answered ({am[r]['x']:.3f}, {am[r]['y']:.3f}, {am[r]['yaw']:.1f} deg), {am[r]['age']:.1f} s old")
    if dead:
        print(f"AMCL NOT ALIVE for {dead}: it stalls when the robot's TF starts after it. Restart it "
              f"(laptop.bash amcl ...) with the robots' stacks already up, then seed again.")
        sys.exit(1)


def cmd_spin(a):
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry

    n = ros_node("localize_spin")
    pub = n.create_publisher(Twist, f"/{a.robot}/cmd_vel", 10)
    st = dict(yaw=None, acc=0.0, x=0.0, y=0.0)

    def on_odom(m):
        yaw = yaw_of(m.pose.pose.orientation)
        if st["yaw"] is not None:
            st["acc"] += math.remainder(yaw - st["yaw"], 2 * math.pi)
        st.update(yaw=yaw, x=m.pose.pose.position.x, y=m.pose.pose.position.y)

    n.create_subscription(Odometry, f"/{a.robot}/odom", on_odom, 20)

    def send(v, w):
        t = Twist()
        t.linear.x, t.angular.z = v, w
        pub.publish(t)

    try:
        if not spin_for(n, lambda: st["yaw"] is not None, 20.0):
            sys.exit(f"no /{a.robot}/odom")
        t0 = time.time()
        while abs(st["acc"]) < 2 * math.pi and time.time() - t0 < 40:
            send(0.0, a.w)
            spin_for(n, lambda: False, 0.1)
        print(f"{a.robot}: turned {math.degrees(st['acc']):.0f} deg in {time.time() - t0:.1f} s")
        for v in (0.1, -0.1):
            x0, y0, t1 = st["x"], st["y"], time.time()
            while math.hypot(st["x"] - x0, st["y"] - y0) < 0.12 and time.time() - t1 < 3:
                send(v, 0.0)
                spin_for(n, lambda: False, 0.1)
        print(f"{a.robot}: forward and back 0.12 m")
    finally:
        for _ in range(5):
            send(0.0, 0.0)
            time.sleep(0.05)


def cmd_check(a):
    am = amcl_poses()
    ar = json.load(open(a.state / "arena.json")) if (a.state / "arena.json").exists() else None
    for r in ROBOTS:
        if r not in am:
            print(f"{r}: no amcl_pose")
            continue
        v = am[r]
        s = (f"{r}: map ({v['x']:.3f}, {v['y']:.3f}, {v['yaw']:.1f} deg)  1-sigma {v['sx'] * 100:.1f} / {v['sy'] * 100:.1f} cm, "
             f"{v['syaw']:.1f} deg, {v['age']:.1f} s old")
        if ar:
            cx, cy, ad = ar["centre"]
            c, s_ = math.cos(math.radians(ad)), math.sin(math.radians(ad))
            dx, dy = v["x"] - cx, v["y"] - cy
            s += f"  arena ({c * dx + s_ * dy:.2f}, {-s_ * dx + c * dy:.2f}, {v['yaw'] - ad:.0f} deg)"
        print(s)


def cmd_verify(a):
    am = amcl_poses()
    a.n = 15
    cmd_scan(a)
    m = load_map(a.map)
    for r in ROBOTS:
        if r not in am:
            print(f"{r}: no amcl_pose")
            continue
        v = am[r]
        lx, ly = scan_points(a.state, r)
        s, x, y, t = refine(m, (v["x"], v["y"], math.radians(v["yaw"])), lx, ly, dxy=0.4, dth=10.0)
        dp, dy = math.hypot(x - v["x"], y - v["y"]), math.degrees(math.remainder(t - math.radians(v["yaw"]), 2 * math.pi))
        ok = dp < 0.12 and abs(dy) < 4 and s > 0.6
        print(f"{r}: AMCL vs scan match {dp * 100:.1f} cm, {dy:+.1f} deg (score {s:.3f}) {'OK' if ok else 'CHECK: re-seed from a fresh match'}")


def overlay(a, m, poses, tee=None, boxes=None, arena=None):
    from PIL import Image, ImageDraw

    xs = [p["x"] for p in poses.values()] + ([tee[0]] if tee else [])
    ys = [p["y"] for p in poses.values()] + ([tee[1]] if tee else [])
    x0, x1, y0, y1 = min(xs) - 3, max(xs) + 3, min(ys) - 3, max(ys) + 3
    k = 8
    c0, c1 = int((x0 - m["ox"]) / m["res"]), int((x1 - m["ox"]) / m["res"])
    r0, r1 = int(m["H"] - (y1 - m["oy"]) / m["res"]), int(m["H"] - (y0 - m["oy"]) / m["res"])
    c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, m["W"]), min(r1, m["H"])
    base = np.where(m["occ"], 0, np.where(m["free"], 255, 200)).astype(np.uint8)[r0:r1, c0:c1]
    im = Image.fromarray(base).convert("RGB").resize(((c1 - c0) * k, (r1 - r0) * k), Image.NEAREST)
    d = ImageDraw.Draw(im)

    def P(x, y):
        return ((x - (m["ox"] + c0 * m["res"])) / m["res"] * k, ((m["oy"] + (m["H"] - r0) * m["res"]) - y) / m["res"] * k)

    for g in range(int(x0) - 1, int(x1) + 2):
        d.line([P(g, y0), P(g, y1)], fill=(215, 215, 255))
    for g in range(int(y0) - 1, int(y1) + 2):
        d.line([P(x0, g), P(x1, g)], fill=(215, 215, 255))
    cols = {"robot_1": (220, 30, 30), "robot_2": (30, 80, 230)}
    for r, p in poses.items():
        pose = (p["x"], p["y"], math.radians(p["yaw"]))
        lx, ly = scan_points(a.state, r)
        wx, wy = to_world(pose, lx, ly)
        rr, cc, ok = cells(m, wx, wy)
        on = ok & (m["dist"][rr, cc] < 0.15)
        for i in range(wx.size):
            q = P(wx[i], wy[i])
            d.ellipse([q[0] - 3, q[1] - 3, q[0] + 3, q[1] + 3], fill=cols[r] if on[i] else (0, 190, 0))
        c, s = math.cos(pose[2]), math.sin(pose[2])
        hx, hy = ROBOT_HALF
        d.polygon([P(p["x"] + c * u - s * v, p["y"] + s * u + c * v) for u, v in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))],
                  outline=cols[r], width=3)
        d.line([P(p["x"], p["y"]), P(p["x"] + 0.4 * c, p["y"] + 0.4 * s)], fill=cols[r], width=3)
        d.text(P(p["x"] + 0.2, p["y"] + 0.3), f"{r} ({p['x']:.2f},{p['y']:.2f},{p['yaw']:.0f})", fill=cols[r])
    if tee:
        tx, ty, tt = tee
        c, s = math.cos(tt), math.sin(tt)
        for (ox, oy), (hx, hy) in boxes:
            bx, by = tx + c * ox - s * oy, ty + s * ox + c * oy
            d.polygon([P(bx + c * u - s * v, by + s * u + c * v) for u, v in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))],
                      outline=(255, 140, 0), width=3)
    if arena:
        cx, cy, ax = arena
        c, s = math.cos(ax), math.sin(ax)
        for h, col in ((2.0, (120, 120, 120)), (1.75, (255, 0, 255))):
            d.polygon([P(cx + c * u - s * v, cy + s * u + c * v) for u, v in ((h, h), (h, -h), (-h, -h), (-h, h))], outline=col, width=2)
        d.line([P(cx, cy), P(cx + 0.5 * c, cy + 0.5 * s)], fill=(120, 120, 120), width=3)
    out = a.state / "overlay.png"
    im.save(out)
    print(f"overlay: {out} (robot points on the map in their colour, green: off the map; orange: T fit; "
          f"grey: arena, magenta: arena stop)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "match", "tee", "args", "seed", "spin", "check", "verify"])
    ap.add_argument("robot", nargs="?")
    ap.add_argument("--map", default=os.environ.get("MAP", str(WS / "out/real/maps/map_corridor.yaml")))
    ap.add_argument("--state", type=Path, default=WS / "out/real/state")
    ap.add_argument("--checkpoint", default="/opt/swarp/runs/remote-davide-cf/L8b-s2-slim240-s0/pusht_lidar_limo_slim240_s2_best.pt")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--poses", choices=["state", "amcl"], default="state")
    ap.add_argument("--arena-yaw", type=float, help="override the arena x axis [deg in the map]")
    ap.add_argument("--arena-centre", help="override the arena centre 'x,y' in the map")
    a = ap.parse_args()
    a.state.mkdir(parents=True, exist_ok=True)
    if a.cmd == "spin" and a.robot not in ROBOTS:
        sys.exit("spin needs robot_1 or robot_2")
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
