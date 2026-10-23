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
* takes ``dry_run``, ``max_v``, ``max_w`` and ``arena_half`` changes while running (set them
  atomically, e.g. ``scripts/real/robots.bash caps``): a restart would reset the odometry the
  localizer tracks. Going dry sends one zero to the driver first;
* relays the driver's ``driver_scan`` -> ``/<robot>/scan`` and ``driver_odom`` ->
  ``/<robot>/odom`` with frames prefixed ``<robot>/`` (and the scan rotated by
  ``scan_yaw_offset``), and publishes TF ``<robot>/odom`` -> ``<robot>/base_footprint`` and
  the static ``<robot>/base_footprint`` -> ``<robot>/laser_link``, so several robots share a
  ROS domain. With ``imu_topic`` set, the relayed odometry is re-integrated as ``limo_base``
  integrates it on the real Limo: heading from the IMU's yaw rate, translation from the
  odometry's forward speed along it (a skid steer's wheel odometry overestimates turns);
  the real driver already does this, so ``imu_topic`` is for Gazebo;
* with ``loc_half > 0`` publishes ``/<robot>/scan_loc`` for the localizer: the scan with
  every return that lands inside the square ``|x|, |y| <= loc_half`` (at the robot's latest
  ``/<robot>/pose``; unmasked before the first pose) blanked. The arena holds only the T and the robots, which are not on
  the map: AMCL then matches only the walls and landmarks outside it;
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
from rcl_interfaces.msg import SetParametersResult
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
        imu_topic = p("imu_topic", "").value
        self.loc_half = float(p("loc_half", 0.0).value)
        self.laser_x = float(p("laser_xyz", [0.103, 0.0, 0.116]).value[0])
        self.imu_yaw = None  # integrated IMU heading
        self.imu_t = None
        self.odom_t = None
        self.odom_pose = (0.0, 0.0, 0.0)  # re-integrated (x, y, yaw) with imu_topic
        prefix = self.robot + "/"
        self.odom_frame = prefix + p("odom_frame", "odom").value
        self.base_frame = prefix + p("base_frame", "base_footprint").value
        self.laser_frame = prefix + p("laser_frame", "laser_link").value
        laser_xyz = [float(v) for v in self.get_parameter("laser_xyz").value]

        self.cmd = None  # (v, w)
        self.cmd_t = -math.inf
        self.pose = None  # (x, y, yaw)
        self.pose_t = -math.inf
        self.estop = False
        self.state = ""

        self.create_subscription(Twist, r + "/cmd_vel", self._on_cmd, 10)
        self.create_subscription(PoseStamped, p("pose_topic", r + "/pose").value, self._on_pose, 10)
        self.create_subscription(Bool, p("estop_topic", "/estop").value, self._on_estop, LATCHED)
        self.driver_pub = self.create_publisher(Twist, driver_cmd, 10)
        self.gated_pub = self.create_publisher(Twist, r + "/cmd_vel_gated", 10)
        self.state_pub = self.create_publisher(String, r + "/bridge_state", LATCHED)
        self.scan_pub = self.create_publisher(LaserScan, r + "/scan", qos_profile_sensor_data) if relay_scan else None
        self.loc_pub = (self.create_publisher(LaserScan, r + "/scan_loc", qos_profile_sensor_data)
                        if self.loc_half > 0 else None)
        if relay_scan or self.loc_pub is not None:
            self.create_subscription(LaserScan, driver_scan if relay_scan else r + "/scan", self._on_scan,
                                     qos_profile_sensor_data)
        if imu_topic:
            from sensor_msgs.msg import Imu

            self.create_subscription(Imu, imu_topic, self._on_imu, qos_profile_sensor_data)
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
        self.add_on_set_parameters_callback(self._on_params)
        self.get_logger().info("%s: %s -> %s, timeout %.2f s" % (self.robot, r + "/cmd_vel", driver_cmd,
                                                                self.lim.cmd_timeout))
        self._log_caps()

    def _log_caps(self) -> None:
        self.get_logger().info(
            "%s|v| <= %.2f, |w| <= %.2f, arena %s"
            % ("DRY RUN (not forwarded), " if self.dry_run else "", self.lim.max_v, self.lim.max_w,
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

    def _on_params(self, params) -> SetParametersResult:
        lim, dry = self.lim, self.dry_run
        for q in params:
            if q.name == "dry_run":
                dry = bool(q.value)
            elif q.name in ("max_v", "max_w", "arena_half"):
                v = float(q.value)
                if not math.isfinite(v) or v < 0.0:
                    return SetParametersResult(successful=False, reason="%s must be finite and >= 0" % q.name)
                lim = lim._replace(**{q.name: v})
            else:
                return SetParametersResult(successful=False, reason="%s is fixed at launch" % q.name)
        if dry and not self.dry_run:
            self.driver_pub.publish(Twist())
        self.lim, self.dry_run = lim, dry
        self._log_caps()
        return SetParametersResult(successful=True)

    def _on_estop(self, m: Bool) -> None:
        if bool(m.data) != self.estop:
            self.get_logger().warn("E-STOP %s" % ("ENGAGED" if m.data else "released"))
        self.estop = bool(m.data)
        self._tick()

    def _on_scan(self, m: LaserScan) -> None:
        m.header.frame_id = self.laser_frame
        m.angle_min += self.scan_yaw_offset
        m.angle_max += self.scan_yaw_offset
        if self.scan_pub is not None:
            self.scan_pub.publish(m)
        if self.loc_pub is not None and self.pose is None:
            self.loc_pub.publish(m)  # no pose yet (the localizer starting): nothing to mask with
        elif self.loc_pub is not None:
            x, y, th = self.pose
            lx, ly = x + self.laser_x * math.cos(th), y + self.laser_x * math.sin(th)
            h = self.loc_half
            ranges = list(m.ranges)
            for k, rng in enumerate(ranges):
                if math.isfinite(rng):
                    a = th + m.angle_min + k * m.angle_increment
                    if abs(lx + rng * math.cos(a)) <= h and abs(ly + rng * math.sin(a)) <= h:
                        ranges[k] = math.inf
            m.ranges = ranges
            self.loc_pub.publish(m)

    @staticmethod
    def _stamp(msg) -> float:
        return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec

    def _on_imu(self, m) -> None:
        t = self._stamp(m)
        if self.imu_t is not None and 0.0 < t - self.imu_t < 0.5:
            self.imu_yaw += m.angular_velocity.z * (t - self.imu_t)
        elif self.imu_yaw is None:
            self.imu_yaw = 0.0
        self.imu_t = t

    def _imu_odometry(self, m: Odometry) -> None:
        """limo_base's integration: heading from the IMU, the forward speed along it."""
        t = self._stamp(m)
        x, y, _ = self.odom_pose
        th = self.imu_yaw
        if self.odom_t is not None and 0.0 < t - self.odom_t < 0.5:
            dt = t - self.odom_t
            x += math.cos(th) * m.twist.twist.linear.x * dt
            y += math.sin(th) * m.twist.twist.linear.x * dt
        self.odom_t = t
        self.odom_pose = (x, y, th)
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        q = m.pose.pose.orientation
        q.x, q.y, q.z, q.w = 0.0, 0.0, math.sin(th / 2.0), math.cos(th / 2.0)

    def _on_odom(self, m: Odometry) -> None:
        m.header.frame_id = self.odom_frame
        m.child_frame_id = self.base_frame
        if self.imu_yaw is not None:
            self._imu_odometry(m)
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
        if not self.dry_run:
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
        for _ in range(3):
            self.driver_pub.publish(Twist())
        spin.flush(self.driver_pub)


def main(args=None) -> None:
    spin.init(args)
    node = RobotBridge()
    spin.spin_until_signal(node, node.stop)
