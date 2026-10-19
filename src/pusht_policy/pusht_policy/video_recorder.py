"""Record the evaluation episodes from the world's video camera to one mp4.

Runs next to ``eval.launch.py record_video:=true``. Frames come straight from gz-transport
(``/video/image``, rendered at a fixed rate of sim time) rather than through a ROS bridge,
and go to an ``ffmpeg`` pipe, so nothing touches the disk but the mp4. Only frames of a
running episode (``/episode`` >= 0) are written; when one ends, its last frame is held for
``hold`` seconds under a SOLVED / TIMEOUT banner. A frame missing from the stream (the
recorder fell behind) is filled by repeating the previous one, so the video plays at sim
time.

This process makes no gz service calls, so the gz Python subscription is safe here (see
``pusht_gz.gz_reset``).
"""

from __future__ import annotations

import math
import subprocess
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node as GzNode
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from rclpy.node import Node
from std_msgs.msg import Int32

from pusht_policy.policy_node import LATCHED, yaw_of

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class VideoRecorder(Node):
    def __init__(self):
        super().__init__("video_recorder")
        self.path = self.declare_parameter("output", "out/video.mp4").value
        self.fps = float(self.declare_parameter("fps", 30.0).value)
        self.hold = float(self.declare_parameter("hold", 1.5).value)
        self.title = self.declare_parameter("title", "").value
        self.goal_tol = float(self.declare_parameter("goal_tolerance", 0.15).value)
        self.angle_tol = float(self.declare_parameter("angle_tolerance", 0.25).value)
        self.font = ImageFont.truetype(FONT, 30)
        self.small = ImageFont.truetype(FONT, 22)
        self.lock = threading.Lock()
        self.episode = -1
        self.t0 = None  # sim time of the episode's first frame
        self.last = None  # (stamp, rgb ndarray) of the last frame written
        self.tee = self.goal = None
        self.ffmpeg = None
        self.size = None
        self.frames = 0
        self.create_subscription(Int32, "/episode", self._on_episode, LATCHED)
        self.create_subscription(PoseStamped, "/goal", self._on_goal, LATCHED)
        self.create_subscription(PoseStamped, "/tee/pose", self._on_tee, 10)
        self.gz = GzNode()
        self.gz.subscribe(Image, "/video/image", self._on_image)

    # ------------------------------------------------------------------ inputs

    def _on_goal(self, m):
        self.goal = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))

    def _on_tee(self, m):
        self.tee = (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))

    def _on_episode(self, m: Int32) -> None:
        with self.lock:
            if m.data == self.episode:
                return
            if self.episode >= 0 and self.last is not None:
                self._hold_end()
            self.episode = m.data
            self.t0 = None
            self.last = None

    def _solved(self) -> bool:
        if self.tee is None or self.goal is None:
            return False
        d = math.hypot(self.tee[0] - self.goal[0], self.tee[1] - self.goal[1])
        a = abs(math.atan2(math.sin(self.tee[2] - self.goal[2]), math.cos(self.tee[2] - self.goal[2])))
        return d < self.goal_tol and a < self.angle_tol

    def _on_image(self, msg) -> None:
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nsec
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        with self.lock:
            if self.episode < 0:
                return
            if self.t0 is None:
                self.t0 = stamp
            if self.last is not None:
                # repeat the previous frame over any gap, so the video keeps sim time
                gap = int(round((stamp - self.last[0]) * self.fps)) - 1
                for _ in range(max(0, min(gap, int(self.fps)))):
                    self._write(self.last[1])
            frame = self._overlay(rgb, stamp - self.t0)
            self._write(frame)
            self.last = (stamp, frame)

    # ------------------------------------------------------------------ output

    def _overlay(self, rgb, t: float, banner: str | None = None, color=(255, 255, 255)) -> np.ndarray:
        img = PILImage.fromarray(rgb)
        d = ImageDraw.Draw(img)
        text = f"episode {self.episode}   t = {t:5.1f} s"
        d.rectangle((0, 0, img.width, 50), fill=(0, 0, 0))
        d.text((16, 9), text, font=self.font, fill=(255, 255, 255))
        if self.title:
            w = d.textlength(self.title, font=self.small)
            d.text((img.width - w - 16, 14), self.title, font=self.small, fill=(200, 200, 200))
        if banner:
            w = d.textlength(banner, font=self.font)
            d.rectangle((0, img.height - 56, img.width, img.height), fill=(0, 0, 0))
            d.text(((img.width - w) / 2, img.height - 48), banner, font=self.font, fill=color)
        return np.asarray(img)

    def _hold_end(self) -> None:
        t = self.last[0] - self.t0
        solved = self._solved()
        banner = f"SOLVED in {t:.1f} s" if solved else f"TIMEOUT at {t:.1f} s"
        color = (80, 230, 80) if solved else (255, 90, 70)
        # redraw on the raw frame's pixels under the overlay (the header is redrawn anyway)
        frame = self._overlay(self.last[1], t, banner, color)
        for _ in range(int(round(self.hold * self.fps))):
            self._write(frame)
        self.get_logger().info(f"episode {self.episode}: {banner}")

    def _write(self, frame: np.ndarray) -> None:
        if self.ffmpeg is None:
            h, w = frame.shape[:2]
            self.size = (w, h)
            self.ffmpeg = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", f"{self.fps:g}", "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE,
            )
            self.get_logger().info(f"recording {w}x{h} at {self.fps:g} fps -> {self.path}")
        self.ffmpeg.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> None:
        with self.lock:
            if self.episode >= 0 and self.last is not None:
                self._hold_end()
                self.episode = -1
            if self.ffmpeg is not None:
                self.ffmpeg.stdin.close()
                self.ffmpeg.wait()
                self.get_logger().info(f"{self.frames} frames ({self.frames / self.fps:.1f} s) -> {self.path}")
                self.ffmpeg = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()
