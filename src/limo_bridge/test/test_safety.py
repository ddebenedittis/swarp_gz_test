import math

from limo_bridge.safety import ARENA, ESTOP, NO_POSE, OK, STALE, Limits, gate

LIM = Limits()  # 0.4 m/s, 1.5 rad/s, 0.3 s, arena +-2.25 m
CENTRE = (0.0, 0.0, 0.0)


def test_estop_beats_everything():
    assert gate((0.2, 0.5), 0.0, True, CENTRE, 0.0, LIM) == (0.0, 0.0, ESTOP)


def test_watchdog():
    assert gate((0.2, 0.5), 0.29, False, CENTRE, 0.0, LIM) == (0.2, 0.5, OK)
    assert gate((0.2, 0.5), 0.31, False, CENTRE, 0.0, LIM) == (0.0, 0.0, STALE)
    assert gate(None, 0.0, False, CENTRE, 0.0, LIM) == (0.0, 0.0, STALE)
    assert gate((0.2, 0.5), math.nan, False, CENTRE, 0.0, LIM)[2] == STALE


def test_clamp_and_non_finite():
    assert gate((1.0, -9.0), 0.0, False, CENTRE, 0.0, LIM) == (0.4, -1.5, OK)
    assert gate((math.nan, math.inf), 0.0, False, CENTRE, 0.0, LIM) == (0.0, 0.0, OK)


def test_stale_pose_stops_only_with_the_arena_stop_on():
    assert gate((0.2, 0.5), 0.0, False, CENTRE, 0.6, LIM) == (0.0, 0.0, NO_POSE)
    assert gate((0.2, 0.5), 0.0, False, None, 0.0, LIM) == (0.0, 0.0, NO_POSE)
    off = LIM._replace(arena_half=0.0)
    assert gate((0.2, 0.5), 0.0, False, None, 0.0, off) == (0.2, 0.5, OK)


def test_arena_blocks_outward_motion_only():
    out_east = (2.3, 0.0, 0.0)  # past +x, facing +x
    assert gate((0.2, 0.5), 0.0, False, out_east, 0.0, LIM) == (0.0, 0.5, ARENA)
    assert gate((-0.2, 0.5), 0.0, False, out_east, 0.0, LIM) == (-0.2, 0.5, OK)  # backing in
    facing_in = (2.3, 0.0, math.pi)
    assert gate((0.2, 0.0), 0.0, False, facing_in, 0.0, LIM) == (0.2, 0.0, OK)
    out_south = (0.0, -2.3, -math.pi / 2)  # past -y, facing -y
    assert gate((0.2, 0.0), 0.0, False, out_south, 0.0, LIM)[2] == ARENA
    assert gate((0.2, 0.0), 0.0, False, (2.2, 0.0, 0.0), 0.0, LIM)[2] == OK  # inside the margin
