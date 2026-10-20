"""Onboard safety bridge between the laptop's policy and one Limo's driver.

Runs on the robot (ROS 2 Foxy / Python 3.8 on the Limo Pro, Humble / 3.10 on the Limo ROS2),
with rclpy and the standard library only. AgileX's ``limo_base`` subscribes to an absolute
``/cmd_vel`` and forwards every Twist to the chassis unclamped, with no timeout, so a laptop
crash, a WiFi drop or a stopped launch would leave the robot driving. This node:

* subscribes ``/<robot>/cmd_vel`` (the policy's command) and publishes, on ``driver_cmd_vel``,
  the command passed through :func:`limo_bridge.safety.gate`: clamped to the trained
  ``|v| <= max_v``, ``|w| <= max_w``; zero when no command arrived for ``cmd_timeout``
  (the watchdog), while ``/estop`` (latched Bool) is true, or, with the arena stop on, when
  the robot's ``/<robot>/pose`` is stale; outward motion zeroed past the arena edge. It
  republishes at ``rate`` Hz, so the chassis gets zeros continuously while the laptop is
  silent, and sends zeros on shutdown;
* with ``dry_run`` (shadow mode) never publishes to the driver; the gated command still goes
  to ``/<robot>/cmd_vel_gated`` (always published) for logging;
* relays the driver's ``driver_scan`` -> ``/<robot>/scan`` and ``driver_odom`` ->
  ``/<robot>/odom`` with frames prefixed ``<robot>/`` (and the scan rotated by
  ``scan_yaw_offset``; the odometry's turns scaled by ``odom_yaw_scale``, the measured
  true / odometry turn ratio: a skid steer's wheel odometry overestimates turns), and publishes TF ``<robot>/odom`` -> ``<robot>/base_footprint`` and
  the static ``<robot>/base_footprint`` -> ``<robot>/laser_link``, so several robots share a
  ROS domain;
* reports its gate state on ``/<robot>/bridge_state`` (String, on change and at 1 Hz).

In Gazebo the same node sits between ``/<robot>/cmd_vel`` and the DiffDrive's
``/<robot>/limo/cmd_vel``, and relays the DiffDrive's ``/<robot>/limo/odom``; the scan relay
is off (the sim's scan is already namespaced).
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage

from limo_bridge import spin
from limo_bridge.safety import OK, Limits, gate, yaw_of

try:
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
except ImportError:  # Foxy
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
    from rclpy.qos import QoSReliabilityPolicy as ReliabilityPolicy
from rclpy.qos import QoSProfile, qos_profile_sensor_data

LATCHED = QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
)


class RobotBridge(Node):
    def __init__(self):
        super().__init__("robot_bridge")
        p = self.declare_parameter
        self.robot = p("robot_name", "robot_1").value
        r = "/" + self.robot
        self.lim = Limits(
            max_v=float(p("max_v", 0.4).value),
            max_w=float(p("max_w", 1.5).value),
            cmd_timeout=float(p("cmd_timeout", 0.3).value),
            arena_half=float(p("arena_half", 2.0).value),
            arena_margin=float(p("arena_margin", 0.25).value),
            pose_timeout=float(p("pose_timeout", 0.5).value),
        )
        self.dry_run = bool(p("dry_run", False).value)
        rate = float(p("rate", 20.0).value)
        driver_cmd = p("driver_cmd_vel", "/cmd_vel").value
        driver_scan = p("driver_scan", "/scan").value
        driver_odom = p("driver_odom", "/odom").value
        relay_scan = bool(p("relay_scan", True).value)
        relay_odom = bool(p("relay_odom", True).value)
        self.publish_tf = bool(p("publish_tf", True).value)
        publish_laser_tf = bool(p("publish_laser_tf", True).value)
        self.scan_yaw_offset = float(p("scan_yaw_offset", 0.0).value)
        self.odom_yaw_scale = float(p("odom_yaw_scale", 1.0).value)
        self._odom_in = None  # last driver odometry pose (x, y, yaw)
        self._odom_out = (0.0, 0.0, 0.0)  # re-integrated pose with the turns scaled
        prefix = self.robot + "/"
        self.odom_frame = prefix + p("odom_frame", "odom").value
        self.base_frame = prefix + p("base_frame", "base_footprint").value
        self.laser_frame = prefix + p("laser_frame", "laser_link").value
        laser_xyz = [float(v) for v in p("laser_xyz", [0.103, 0.0, 0.116]).value]

        self.cmd = None  # (v, w)
        self.cmd_t = -math.inf
        self.pose = None  # (x, y, yaw)
        self.pose_t = -math.inf
        self.estop = False
        self.state = ""

        self.create_subscription(Twist, r + "/cmd_vel", self._on_cmd, 10)
        self.create_subscription(PoseStamped, p("pose_topic", r + "/pose").value, self._on_pose, 10)
        self.create_subscription(Bool, p("estop_topic", "/estop").value, self._on_estop, LATCHED)
        self.driver_pub = None if self.dry_run else self.create_publisher(Twist, driver_cmd, 10)
        self.gated_pub = self.create_publisher(Twist, r + "/cmd_vel_gated", 10)
        self.state_pub = self.create_publisher(String, r + "/bridge_state", LATCHED)
        if relay_scan:
            self.scan_pub = self.create_publisher(LaserScan, r + "/scan", qos_profile_sensor_data)
            self.create_subscription(LaserScan, driver_scan, self._on_scan, qos_profile_sensor_data)
        if relay_odom or self.publish_tf:
            self.odom_pub = self.create_publisher(Odometry, r + "/odom", 10) if relay_odom else None
            self.tf_pub = self.create_publisher(TFMessage, "/tf", 10) if self.publish_tf else None
            self.create_subscription(Odometry, driver_odom, self._on_odom, 10)
        if publish_laser_tf:
            self.static_pub = self.create_publisher(TFMessage, "/tf_static", LATCHED)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id, t.child_frame_id = self.base_frame, self.laser_frame
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = laser_xyz
            t.transform.rotation.w = 1.0
            self.static_pub.publish(TFMessage(transforms=[t]))

        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(1.0, lambda: self._report(force=True))
        self.get_logger().info(
            "%s: %s -> %s%s, |v| <= %.2f, |w| <= %.2f, timeout %.2f s, arena %s"
            % (self.robot, r + "/cmd_vel", driver_cmd, " (DRY RUN: not forwarded)" if self.dry_run else "",
               self.lim.max_v, self.lim.max_w, self.lim.cmd_timeout,
               "off" if self.lim.arena_half <= 0 else "+-%.2f m" % (self.lim.arena_half + self.lim.arena_margin))
        )

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------------ inputs

    def _on_cmd(self, m: Twist) -> None:
        self.cmd = (m.linear.x, m.angular.z)
        self.cmd_t = self.now()
        self._tick()

    def _on_pose(self, m: PoseStamped) -> None:
        self.pose = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))
        self.pose_t = self.now()

    def _on_estop(self, m: Bool) -> None:
        if bool(m.data) != self.estop:
            self.get_logger().warn("E-STOP %s" % ("ENGAGED" if m.data else "released"))
        self.estop = bool(m.data)
        self._tick()

    def _on_scan(self, m: LaserScan) -> None:
        m.header.frame_id = self.laser_frame
        m.angle_min += self.scan_yaw_offset
        m.angle_max += self.scan_yaw_offset
        self.scan_pub.publish(m)

    def _rescale(self, m: Odometry) -> None:
        """Re-integrate the driver's odometry with every turn increment scaled by
        ``odom_yaw_scale`` (translations kept, in the corrected heading)."""
        q = m.pose.pose.orientation
        cur = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(q))
        prev, self._odom_in = self._odom_in, cur
        x, y, th = self._odom_out
        if prev is not None:
            c, s = math.cos(prev[2]), math.sin(prev[2])
            ddx, ddy = cur[0] - prev[0], cur[1] - prev[1]
            dx, dy = c * ddx + s * ddy, -s * ddx + c * ddy
            dth = math.atan2(math.sin(cur[2] - prev[2]), math.cos(cur[2] - prev[2]))
            x += math.cos(th) * dx - math.sin(th) * dy
            y += math.sin(th) * dx + math.cos(th) * dy
            th = math.atan2(math.sin(th + self.odom_yaw_scale * dth), math.cos(th + self.odom_yaw_scale * dth))
        else:
            x, y, th = cur
        self._odom_out = (x, y, th)
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(th / 2.0), math.cos(th / 2.0)
        m.twist.twist.angular.z *= self.odom_yaw_scale

    def _on_odom(self, m: Odometry) -> None:
        m.header.frame_id = self.odom_frame
        m.child_frame_id = self.base_frame
        if self.odom_yaw_scale != 1.0:
            self._rescale(m)
        if self.odom_pub is not None:
            self.odom_pub.publish(m)
        if self.tf_pub is not None:
            t = TransformStamped()
            t.header = m.header
            t.child_frame_id = self.base_frame
            t.transform.translation.x = m.pose.pose.position.x
            t.transform.translation.y = m.pose.pose.position.y
            t.transform.rotation = m.pose.pose.orientation
            self.tf_pub.publish(TFMessage(transforms=[t]))

    # ------------------------------------------------------------------ output

    def _tick(self) -> None:
        now = self.now()
        v, w, reason = gate(self.cmd, now - self.cmd_t, self.estop, self.pose, now - self.pose_t, self.lim)
        out = Twist()
        out.linear.x, out.angular.z = float(v), float(w)
        if self.driver_pub is not None:
            self.driver_pub.publish(out)
        self.gated_pub.publish(out)
        self.state = ("dry_run:" if self.dry_run else "") + reason
        self._report()

    def _report(self, force: bool = False) -> None:
        if force or self.state != getattr(self, "_reported", None):
            if self.state != getattr(self, "_reported", None) and not self.state.endswith(OK):
                self.get_logger().info("gate: %s" % self.state)
            self._reported = self.state
            self.state_pub.publish(String(data=self.state))

    def stop(self) -> None:
        """Zero the chassis on the way out (SIGINT/SIGTERM)."""
        if self.driver_pub is None:
            return
        for _ in range(3):
            self.driver_pub.publish(Twist())
        spin.flush(self.driver_pub)


def main(args=None) -> None:
    spin.init(args)
    node = RobotBridge()
    spin.spin_until_signal(node, node.stop)
