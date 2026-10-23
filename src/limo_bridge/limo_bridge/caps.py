"""Switch this robot's running robot_bridge (dry_run, max_v, max_w, arena_half) in one atomic
parameter set, then print the gate it reports. Runs ONBOARD, against the local bridge: the
laptop's Fast DDS and Foxy's Cyclone exchange topics but not service replies.

    python3 -m limo_bridge.caps ROBOT NAME:=VALUE ... [--force]     (robots.bash caps MODE wraps it)

Leaving dry_run (going live) is refused while an episode runs (latched ``/episode`` >= 0): stop
it first, or pass --force.
"""

import sys
import time

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParametersAtomically
from std_msgs.msg import Int32, String

from limo_bridge.robot_bridge import LATCHED


def param(name, text):
    v = ParameterValue()
    if text.lower() in ("true", "false"):
        v.type, v.bool_value = ParameterType.PARAMETER_BOOL, text.lower() == "true"
    else:
        v.type, v.double_value = ParameterType.PARAMETER_DOUBLE, float(text)
    return Parameter(name=name, value=v)


def spin_until(node, cond, timeout):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
    return cond()


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = len(args) < len(sys.argv) - 1
    robot, params = args[0], [param(*a.split(":=", 1)) for a in args[1:]]
    rclpy.init(args=[])
    n = rclpy.create_node("caps_" + robot)
    st = {}
    n.create_subscription(Int32, "/episode", lambda m: st.__setitem__("ep", m.data), LATCHED)
    n.create_subscription(String, "/%s/bridge_state" % robot, lambda m: st.__setitem__("gate", m.data), LATCHED)
    if any(p.name == "dry_run" and not p.value.bool_value for p in params) and not force:
        spin_until(n, lambda: "ep" in st, 3.0)  # no answer: no operator running, no episode
        if st.get("ep", -1) >= 0:
            print("%s: episode %d is running: stop it (laptop.bash op stop) before going live, or --force"
                  % (robot, st["ep"]))
            sys.exit(1)
    c = n.create_client(SetParametersAtomically, "/%s/robot_bridge/set_parameters_atomically" % robot)
    if not c.wait_for_service(timeout_sec=10.0):
        print("%s: robot_bridge not running" % robot)
        sys.exit(1)
    f = c.call_async(SetParametersAtomically.Request(parameters=params))
    spin_until(n, f.done, 5.0)
    res = f.result().result if f.done() else None
    st.pop("gate", None)
    spin_until(n, lambda: "gate" in st, 1.5)  # the 1 Hz report shows the new state
    ok = bool(res and res.successful)
    print("%s: %s, gate %s" % (robot, "set " + " ".join(args[1:]) if ok else "FAILED (%s)" % (res.reason if res else "no answer"),
                                st.get("gate", "?")))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
