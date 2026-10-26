"""Delayed messages between the robots' policy nodes, for a checkpoint trained with swarp's
``--comm latent|vae|pose`` (``PolicyNet.delayed``).

No ROS here: :mod:`pusht_policy.policy_node` only wires the topics, and the test replays a
swarp rollout through the same :class:`Messenger`.

Each control step of episode ``e``, on the scan stamped ``t``, a robot sends
:meth:`Messenger.payload` (``std_msgs/Float32MultiArray`` on ``/robot_i/msg``)::

    e | t_hi | t_lo | m (msg_dim)

with ``t = t_hi * 2**16 + t_lo`` [s] (one float32 holds neither an epoch time nor a long sim
time to the millisecond; split, ``t_lo`` keeps ~4 ms) and ``m`` the ``msg_dim`` floats
``actor.messages`` encodes its observation to. On its step at ``t`` a robot hears, from each
teammate, the newest message of episode ``e`` stamped in ``(t - 1.5 dt, t - 0.5 dt]``: the
teammate's previous step, whichever way the two robots' scan-triggered step grids are
offset. A teammate with none (the episode's first step, a lost or late packet, a stale one)
is not heard, as a dropped packet in training. Matching on stamps rather than on step
counts keeps a robot that lost a scan, and so stepped later from then on, from reading a
two-step-old message as the previous one. The stamps are the scans', so the robots' clocks
must agree to well under ``dt / 2`` (sim time, or synchronised clocks on the fleet).

The actor then reads the mean of the heard messages and whether any was heard, the columns
swarp's ``PolicyNet._listen`` computes from the teammates' previous observations
(``swarp/rl/networks.py``), here from the messages themselves.

The message is the encoder's mean (``messages(obs, role)`` with no noise): exact for
``"pose"`` and ``"latent"``, and for ``"vae"`` the greedy message swarp's eval sends, while
its training rollouts send ``mean + std * noise``, so a ``"vae"`` actor run with ``sample``
hears less noisy messages than it trained on.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

import numpy as np
import torch

#: Payload columns before the message: episode id, stamp's high and low parts.
HEADER = 3
_SPLIT = 65536.0


def split_stamp(t: float) -> tuple[float, float]:
    hi = float(np.floor(t / _SPLIT))
    return hi, t - hi * _SPLIT


class Inbox:
    """The last few messages one teammate sent, as ``(episode, stamp, message)``."""

    def __init__(self, msg_dim: int, maxlen: int = 8):
        self.msg_dim = msg_dim
        self.msgs: deque[tuple[int, float, np.ndarray]] = deque(maxlen=maxlen)

    def push(self, data: Sequence[float]) -> None:
        if len(data) != HEADER + self.msg_dim:  # another checkpoint's width: never heard
            return
        stamp = float(data[1]) * _SPLIT + float(data[2])
        self.msgs.append((round(data[0]), stamp, np.asarray(data[HEADER:], dtype=np.float32)))

    def get(self, episode: int, lo: float, hi: float) -> np.ndarray | None:
        """The newest message of ``episode`` stamped in ``(lo, hi]``, or ``None``."""
        best = None
        for e, s, m in self.msgs:
            if e == episode and lo < s <= hi and (best is None or s > best[0]):
                best = (s, m)
        return None if best is None else best[1]


class Messenger:
    """One robot's side of the delayed messages, for ``actor`` (a delayed-comm
    ``PolicyNet``) acting every ``dt`` with ``role`` ``[1]`` among teammates ``mates``."""

    def __init__(self, actor: torch.nn.Module, role: torch.Tensor, mates: Iterable[int], dt: float):
        if not getattr(actor, "delayed", False):
            raise ValueError("the actor sends no delayed messages")
        self.actor, self.role, self.dt = actor, role, float(dt)
        self.msg_dim = int(actor.msg_dim)
        self.inbox = {j: Inbox(self.msg_dim) for j in mates}

    def receive(self, mate: int, data: Sequence[float]) -> None:
        self.inbox[mate].push(data)

    @torch.no_grad()
    def payload(self, episode: int, t: float, obs: torch.Tensor) -> list[float]:
        """What to send after the step on the scan stamped ``t`` [s], on ``obs``
        ``[1, obs_dim (+ priv_dim)]``: a privileged actor's extra columns never enter the
        message, as in swarp."""
        m = self.actor.messages(obs, self.role)[0]
        return [float(episode), *split_stamp(t), *m.tolist()]

    def heard(self, episode: int, t: float) -> torch.Tensor:
        """``[1, msg_dim + 1]``: the mean of the teammates' previous-step messages for the
        step at ``t`` (stamped in ``(t - 1.5 dt, t - 0.5 dt]``), and 1 when any was heard
        (zeros when none)."""
        lo, hi = t - 1.5 * self.dt, t - 0.5 * self.dt
        got = [m for box in self.inbox.values() if (m := box.get(episode, lo, hi)) is not None]
        out = torch.zeros(1, self.msg_dim + 1)
        if got:
            out[0, :-1] = torch.from_numpy(np.mean(got, axis=0, dtype=np.float32))
            out[0, -1] = 1.0
        return out

    @torch.no_grad()
    def step(self, obs: torch.Tensor, heard: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``actor.step``'s ``(loc, scale)`` on ``obs`` with ``heard`` in place of the
        listened columns: ``_listen`` pools the messages it encodes from every agent's
        previous observation, which one robot does not have, so it is shadowed for the call
        and everything around it (role input, privileged columns, trunk, heads) is swarp's."""
        self.actor._listen = lambda *_: heard.to(obs.dtype)
        try:
            loc, scale = self.actor.step(obs, self.role)[:2]
        finally:
            del self.actor._listen
        return loc, scale
