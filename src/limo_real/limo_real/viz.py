"""RViz markers for the real setup (``config/real.rviz``), on ``/viz/markers``:

* each robot's footprint (the checkpoint's box) and name, locked to its TF frame
  ``<robot>/base_footprint``, so it sits where the localizer puts it;
* the goal T (``/goal``) with the goal tolerance circle, and the T (``/tee/pose``) when
  something tracks it;
* the arena square ``[-world_size, world_size]^2``.

The goal is also an interactive marker (server ``/goal_drag``): drag the T in the plane, turn
it by its ring, and on release the pose goes on ``/goal_request`` (in ``arena``, through TF
from RViz's fixed frame). The ``operator`` console applies it, also mid-episode.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from interactive_markers import InteractiveMarkerServer
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import (InteractiveMarker, InteractiveMarkerControl, InteractiveMarkerFeedback, Marker,
                                    MarkerArray)

from limo_real.frames import compose, yaw_of

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
ROBOT_COLORS = [(0.9, 0.2, 0.2), (0.2, 0.4, 0.9), (0.2, 0.7, 0.3), (0.8, 0.5, 0.1)]


def _marker(ns: str, mid: int, kind: int, frame: str, rgba: tuple[float, float, float, float]) -> Marker:
    m = Marker()
    m.header.frame_id = frame
    m.ns, m.id, m.type, m.action = ns, mid, kind, Marker.ADD
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    m.pose.orientation.w = 1.0
    m.frame_locked = True
    return m


def _set_pose(m: Marker | PoseStamped, x: float, y: float, z: float, yaw: float) -> None:
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
    m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)


class Viz(Node):
    def __init__(self):
        super().__init__("viz")
        p = self.declare_parameter
        ckpt = p("checkpoint", "").value
        if ckpt:
            from pusht_policy.task import Task

            task = Task.from_checkpoint(ckpt)
            n, self.world_size, self.goal_tol = task.n_agents, task.world_size, task.goal_tolerance
            self.tee_boxes = list(zip(task.tee_off, task.tee_half, strict=True))
            half = task.agent_half or (task.agent_radius, task.agent_radius)
        else:
            n, self.world_size, self.goal_tol = int(p("n_robots", 2).value), 2.0, 0.15
            self.tee_boxes = []
            half = (0.161, 0.110)
        self.robots = [f"robot_{i}" for i in range(1, n + 1)]
        self.half = half
        self.arena_frame = p("arena_frame", "arena").value
        self.goal: PoseStamped | None = None
        self.tee: PoseStamped | None = None
        self.create_subscription(PoseStamped, "/goal", self._on_goal, LATCHED)
        self.create_subscription(PoseStamped, "/tee/pose", self._on_tee, 10)
        self.pub = self.create_publisher(MarkerArray, "/viz/markers", 10)
        self.create_timer(0.2, self._publish)
        if not self.tee_boxes:
            self.get_logger().warn("no checkpoint: the goal and the T are drawn as arrows only")
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.request_pub = self.create_publisher(PoseStamped, "/goal_request", 10)
        self.server = InteractiveMarkerServer(self, "goal_drag")
        self.server.insert(self._goal_handle(), feedback_callback=self._on_drag)
        self.server.applyChanges()

    def _goal_handle(self) -> InteractiveMarker:
        """The draggable goal: the T (or an arrow) moved in the plane, a ring turning it about z."""
        im = InteractiveMarker()
        im.header.frame_id = self.arena_frame
        im.name, im.description, im.scale = "goal", "goal", 0.6
        im.pose.orientation.w = 1.0
        rgba = (0.1, 0.9, 0.3, 0.25)
        visuals = []
        for (ox, oy), (hx, hy) in self.tee_boxes:
            m = _marker("", 0, Marker.CUBE, "", rgba)
            _set_pose(m, ox, oy, 0.03, 0.0)
            m.scale.x, m.scale.y, m.scale.z = 2 * hx, 2 * hy, 0.06
            visuals.append(m)
        if not visuals:
            m = _marker("", 0, Marker.ARROW, "", rgba)
            m.scale.x, m.scale.y, m.scale.z = 0.3, 0.06, 0.06
            visuals.append(m)
        for mode, markers in ((InteractiveMarkerControl.MOVE_PLANE, visuals), (InteractiveMarkerControl.ROTATE_AXIS, [])):
            c = InteractiveMarkerControl()
            c.orientation.w = c.orientation.y = math.sqrt(0.5)  # control axis = arena z
            c.interaction_mode, c.always_visible, c.markers = mode, bool(markers), markers
            im.controls.append(c)
        return im

    def _on_drag(self, fb: InteractiveMarkerFeedback) -> None:
        if fb.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return
        # RViz reports the pose in its fixed frame, not the marker's
        pose = (fb.pose.position.x, fb.pose.position.y, yaw_of(fb.pose.orientation))
        if fb.header.frame_id not in ("", self.arena_frame):
            try:
                t = self.tf.lookup_transform(self.arena_frame, fb.header.frame_id, Time()).transform
            except TransformException as e:
                self.get_logger().warn(f"goal drag dropped: {e}")
                return
            pose = compose((t.translation.x, t.translation.y, yaw_of(t.rotation)), pose)
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.arena_frame
        _set_pose(m, *pose[:2], 0.0, pose[2])
        self.request_pub.publish(m)

    def _on_goal(self, m: PoseStamped) -> None:
        self.goal = m
        self.server.setPose("goal", m.pose, m.header)
        self.server.applyChanges()

    def _on_tee(self, m: PoseStamped) -> None:
        self.tee = m

    def _tee(self, ns: str, pose: PoseStamped, rgba) -> list[Marker]:
        frame = pose.header.frame_id or self.arena_frame
        x, y, yaw = pose.pose.position.x, pose.pose.position.y, yaw_of(pose.pose.orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        out = []
        for k, ((ox, oy), (hx, hy)) in enumerate(self.tee_boxes):
            m = _marker(ns, k, Marker.CUBE, frame, rgba)
            _set_pose(m, x + c * ox - s * oy, y + s * ox + c * oy, 0.02, yaw)
            m.scale.x, m.scale.y, m.scale.z = 2 * hx, 2 * hy, 0.04
            out.append(m)
        arrow = _marker(ns, len(self.tee_boxes), Marker.ARROW, frame, rgba[:3] + (1.0,))
        _set_pose(arrow, x, y, 0.05, yaw)
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.25, 0.03, 0.03
        out.append(arrow)
        return out

    def _circle(self, ns: str, frame: str, x: float, y: float, r: float, rgba) -> Marker:
        m = _marker(ns, 0, Marker.LINE_STRIP, frame, rgba)
        m.scale.x = 0.01
        m.points = [Point(x=x + r * math.cos(a), y=y + r * math.sin(a), z=0.01)
                    for a in (2 * math.pi * k / 48 for k in range(49))]
        return m

    def _publish(self) -> None:
        ms = []
        for i, r in enumerate(self.robots):
            rgb = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            body = _marker(f"{r}/footprint", 0, Marker.CUBE, f"{r}/base_footprint", rgb + (0.6,))
            body.pose.position.z = 0.08
            body.scale.x, body.scale.y, body.scale.z = 2 * self.half[0], 2 * self.half[1], 0.16
            nose = _marker(f"{r}/footprint", 1, Marker.ARROW, f"{r}/base_footprint", rgb + (1.0,))
            nose.pose.position.z = 0.17
            nose.scale.x, nose.scale.y, nose.scale.z = 0.25, 0.03, 0.03
            label = _marker(f"{r}/footprint", 2, Marker.TEXT_VIEW_FACING, f"{r}/base_footprint", (1.0, 1.0, 1.0, 1.0))
            label.pose.position.z = 0.35
            label.scale.z = 0.12
            label.text = r
            ms += [body, nose, label]
        arena = _marker("arena", 0, Marker.LINE_STRIP, self.arena_frame, (0.6, 0.6, 0.6, 1.0))
        arena.scale.x = 0.02
        h = self.world_size
        arena.points = [Point(x=x, y=y, z=0.0) for x, y in ((-h, -h), (h, -h), (h, h), (-h, h), (-h, -h))]
        ms.append(arena)
        if self.goal is not None:
            ms += self._tee("goal", self.goal, (0.1, 0.9, 0.3, 0.45))
            g = self.goal.pose.position
            ms.append(self._circle("goal_tolerance", self.goal.header.frame_id or self.arena_frame,
                                   g.x, g.y, self.goal_tol, (0.1, 0.9, 0.3, 1.0)))
        if self.tee is not None:
            ms += self._tee("tee", self.tee, (1.0, 0.6, 0.1, 0.8))
        self.pub.publish(MarkerArray(markers=ms))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Viz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
