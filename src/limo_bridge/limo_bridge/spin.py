"""Spin a node until SIGINT/SIGTERM, then run a last action while publishing still works.

rclpy's own SIGINT handler (Galactic and later) shuts the context down before control
returns to Python, so a ``finally: pub.publish(zero)`` after ``rclpy.spin`` is silently
skipped (``rclpy.ok()`` is already false) and the robot keeps its last command. Here rclpy
is initialised without its signal handlers, SIGINT and SIGTERM only set a flag, and the node
spins in short slices until the flag is set; ``on_stop`` then runs with the context alive.

Works on Foxy (no ``SignalHandlerOptions``; its handler is replaced by ours) and later.
"""

from __future__ import annotations

import signal
import time

import rclpy


def init(args=None) -> None:
    try:
        from rclpy.signals import SignalHandlerOptions

        rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):  # Foxy
        rclpy.init(args=args)


def spin_until_signal(node, on_stop=None, slice_s: float = 0.05) -> None:
    """Spin ``node`` until SIGINT/SIGTERM (or the context dies), call ``on_stop()``, destroy
    the node and shut rclpy down."""
    stop = []

    def handler(signum, _frame):
        stop.append(signum)

    old = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while not stop and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=slice_s)
    except KeyboardInterrupt:
        pass
    finally:
        if on_stop is not None and rclpy.ok():
            on_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for s, h in old.items():
            signal.signal(s, h)


def flush(publisher, timeout_s: float = 0.3) -> None:
    """Give a just-published message time to leave the process: wait for every reliable
    subscriber's ack where rclpy supports it (Humble and later), else sleep briefly."""
    wait = getattr(publisher, "wait_for_all_acked", None)
    if wait is not None:
        from rclpy.duration import Duration

        try:
            wait(Duration(seconds=timeout_s))
            return
        except Exception:  # noqa: BLE001 - best effort on the way out
            pass
    time.sleep(min(timeout_s, 0.1))
