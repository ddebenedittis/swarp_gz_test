"""Shared bits of the robot measurement scripts."""

from __future__ import annotations

import json
import os
import time

import rclpy
from rclpy.node import Node


def wait(node: Node, seconds: float, pred=None) -> bool:
    """Spin ``node`` for ``seconds`` of wall time, or until ``pred()`` is true."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.005)
        if pred is not None and pred():
            return True
    return False


def save(out: str, name: str, summary: dict) -> None:
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"-> {path}")
