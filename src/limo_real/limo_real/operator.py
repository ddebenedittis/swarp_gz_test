"""Operator console for real Push-T episodes: goal, start/stop, e-stop, scoring.

    ros2 run limo_real operator --ros-args -p checkpoint:=/opt/swarp/runs/<run>/<ckpt>.pt -p out:=out/real/<name>

Type commands at the ``push-t>`` prompt:

    goal X Y YAW_DEG     set the goal (arena frame), latched on /goal
    goal mocap           take the goal from the mocap body goal_marker (/goal_marker/pose)
    start                start an episode: /episode = next id (the policies drive)
    stop [ok|fail]       end it (/episode = -1); ok/fail scores it by hand (no T tracking)
    estop | release      latch / release /estop (every robot_bridge zeroes its chassis)
    status               T error to the goal, pose ages, every robot_bridge's gate state
    quit                 stop the episode and exit

While an episode runs, the T pose (``/tee/pose``, from mocap) is scored every ``dt`` as
swarp and the Gazebo episode manager do: success when ``|p - g| < goal_tolerance`` and
``|Δθ| < angle_tolerance``, timeout after ``max_steps * dt``; either ends the episode.
Tolerances, ``dt`` and ``max_steps`` come from ``checkpoint`` when given. Each episode is
a row of ``<out>/episodes.csv`` with the initial poses, the result and the gate events.
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int32, String

from limo_bridge import spin
from limo_real.frames import pose_msg, wrap, yaw_of

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def _from_checkpoint(path: str) -> dict:
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False)
    kw = meta.get("scen_kwargs", {})
    return {
        "dt": float(meta["dt"]),
        "max_steps": int(meta["max_steps"]),
        "goal_tolerance": float(kw.get("goal_tolerance", 0.15)),
        "angle_tolerance": float(kw.get("angle_tolerance", 0.25)),
        "n_robots": int(meta["n_agents"]),
    }


class Operator(Node):
    def __init__(self):
        super().__init__("operator")
        p = self.declare_parameter
        ckpt = p("checkpoint", "").value
        task = _from_checkpoint(ckpt) if ckpt else {}
        self.dt = float(task.get("dt", p("dt", 0.2).value))
        self.max_steps = int(task.get("max_steps", p("max_steps", 450).value))
        self.goal_tol = float(task.get("goal_tolerance", p("goal_tolerance", 0.15).value))
        self.angle_tol = float(task.get("angle_tolerance", p("angle_tolerance", 0.25).value))
        n = int(task.get("n_robots", p("n_robots", 2).value))
        self.robots = [f"robot_{i}" for i in range(1, n + 1)]
        self.out = p("out", "out/real").value
        self.frame_id = p("frame_id", "arena").value

        self.poses: dict[str, tuple[float, float, float, float]] = {}  # body -> (wall t, x, y, yaw)
        self.gates: dict[str, str] = {}
        self.events: list[str] = []
        for b in self.robots + ["tee", "goal_marker"]:
            self.create_subscription(PoseStamped, f"/{b}/pose", lambda m, b=b: self._on_pose(b, m), 10)
        for r in self.robots:
            self.create_subscription(String, f"/{r}/bridge_state", lambda m, r=r: self._on_gate(r, m), LATCHED)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal", LATCHED)
        self.episode_pub = self.create_publisher(Int32, "/episode", LATCHED)
        self.estop_pub = self.create_publisher(Bool, "/estop", LATCHED)

        self.goal: tuple[float, float, float] | None = None
        self.episode = -1
        self.next_id = 0
        self.t0 = 0.0
        self.k = 0
        self.min_err = math.inf
        self.start_state: dict = {}
        self.lock = threading.Lock()
        self.episode_pub.publish(Int32(data=-1))
        self.create_timer(self.dt, self._score)
        os.makedirs(self.out, exist_ok=True)
        self.csv_path = os.path.join(self.out, "episodes.csv")

    # ------------------------------------------------------------------ inputs

    def _on_pose(self, b: str, m: PoseStamped) -> None:
        self.poses[b] = (time.monotonic(), m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))

    def _on_gate(self, r: str, m: String) -> None:
        if self.gates.get(r) != m.data:
            self.gates[r] = m.data
            if self.episode >= 0 and not m.data.endswith("ok"):
                self.events.append(f"{r}:{m.data}@{time.monotonic() - self.t0:.1f}")

    def fresh(self, b: str, age: float = 0.5):
        p = self.poses.get(b)
        return p if p is not None and time.monotonic() - p[0] < age else None

    # ------------------------------------------------------------------ commands

    def set_goal(self, x: float, y: float, yaw: float) -> None:
        self.goal = (x, y, wrap(yaw))
        self.goal_pub.publish(pose_msg(self.get_clock().now().to_msg(), self.frame_id, x, y, yaw))

    def tee_error(self):
        tee = self.fresh("tee")
        if tee is None or self.goal is None:
            return None
        return math.hypot(tee[1] - self.goal[0], tee[2] - self.goal[1]), abs(wrap(tee[3] - self.goal[2]))

    def start(self) -> str:
        with self.lock:
            if self.goal is None:
                return "set a goal first"
            if self.episode >= 0:
                return f"episode {self.episode} is running"
            missing = [r for r in self.robots if self.fresh(r) is None]
            if missing:
                return f"no fresh pose for {missing}"
            self.episode, self.next_id = self.next_id, self.next_id + 1
            self.t0, self.k, self.min_err, self.events = time.monotonic(), 0, math.inf, []
            self.start_state = {b: self.fresh(b)[1:] for b in self.robots + ["tee"] if self.fresh(b)}
            self.episode_pub.publish(Int32(data=self.episode))
            return f"episode {self.episode} started"

    def stop(self, result: str = "aborted") -> str:
        with self.lock:
            if self.episode < 0:
                self.episode_pub.publish(Int32(data=-1))
                return "no episode running"
            ep, self.episode = self.episode, -1
            self.episode_pub.publish(Int32(data=-1))
            err = self.tee_error() or (math.nan, math.nan)
            row = {"episode": ep, "result": result, "time": round(time.monotonic() - self.t0, 2),
                   "final_pos_err": round(err[0], 4), "final_ang_err": round(err[1], 4),
                   "min_pos_err": round(self.min_err, 4), "goal": self.goal, "events": " ".join(self.events)}
            for b in self.robots + ["tee"]:
                row[f"{b}_start"] = tuple(round(v, 3) for v in self.start_state.get(b, ()))
            new = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            return f"episode {ep}: {result.upper()} after {row['time']} s, final err {err[0]:.3f} m / {err[1]:.3f} rad"

    def _score(self) -> None:
        if self.episode < 0:
            return
        self.k += 1
        err = self.tee_error()
        if err is not None:
            self.min_err = min(self.min_err, err[0])
            if err[0] < self.goal_tol and err[1] < self.angle_tol:
                print("\n" + self.stop("solved"), flush=True)
                return
        if self.k >= self.max_steps:
            print("\n" + self.stop("timeout"), flush=True)

    def status(self) -> str:
        now = time.monotonic()
        lines = [f"episode {self.episode}, goal {self.goal}"]
        err = self.tee_error()
        lines.append("T: " + ("no fresh /tee/pose" if err is None else f"{err[0]:.3f} m / {err[1]:.3f} rad from goal"))
        for b in self.robots + ["tee", "goal_marker"]:
            p = self.poses.get(b)
            lines.append(f"  {b}: " + ("never" if p is None else f"({p[1]:.2f}, {p[2]:.2f}, {math.degrees(p[3]):.0f} deg), {now - p[0]:.2f} s old"))
        lines += [f"  {r} gate: {self.gates.get(r, 'no bridge')}" for r in self.robots]
        return "\n".join(lines)

    def command(self, line: str) -> str | None:
        w = line.split()
        if not w:
            return ""
        c = w[0].lower()
        if c == "goal" and len(w) == 4:
            self.set_goal(float(w[1]), float(w[2]), math.radians(float(w[3])))
            return f"goal {self.goal}"
        if c == "goal" and len(w) == 2 and w[1] == "mocap":
            g = self.fresh("goal_marker")
            if g is None:
                return "no fresh /goal_marker/pose"
            self.set_goal(*g[1:])
            return f"goal {self.goal}"
        if c == "start":
            return self.start()
        if c == "stop":
            return self.stop({"ok": "solved_manual", "fail": "failed_manual"}.get(w[1] if len(w) > 1 else "", "aborted"))
        if c == "estop":
            self.estop_pub.publish(Bool(data=True))
            self.stop("estop")
            return "E-STOP engaged"
        if c == "release":
            self.estop_pub.publish(Bool(data=False))
            return "e-stop released"
        if c == "status":
            return self.status()
        if c in ("quit", "exit"):
            return None
        return "commands: goal X Y YAW_DEG | goal mocap | start | stop [ok|fail] | estop | release | status | quit"


def main(args=None) -> None:
    spin.init(args)  # Ctrl-C raises KeyboardInterrupt here, with the context still alive
    node = Operator()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    print(f"robots {node.robots}, dt {node.dt}, {node.max_steps} steps, tolerances {node.goal_tol} m / "
          f"{node.angle_tol} rad -> {node.csv_path}")
    try:
        while True:
            try:
                line = input("push-t> ")
            except EOFError:
                break
            try:
                out = node.command(line)
            except ValueError as e:
                out = f"error: {e}"
            if out is None:
                break
            if out:
                print(out, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print(node.stop(), flush=True)
        time.sleep(0.3)
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
