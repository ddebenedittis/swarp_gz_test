#!/bin/bash
# Emergency stop for the real Limos, from the laptop host (no ROS needed here).
#
#   scripts/real_stop.bash [user@robot ...]      (default: both lab Limos)
#
# In parallel: on every robot, ~/swarp_ws/stop.sh (SIGINT to robot_bridge, which sends zeros
# to the chassis as it exits, then the whole robot.launch.py stack), and, when the laptop
# container swarp_real runs, the latched /estop (every bridge still alive zeroes its chassis)
# and SIGINT to the laptop launch (each policy publishes a zero twist).
ROBOTS=("$@")
[ ${#ROBOTS[@]} -eq 0 ] && ROBOTS=(agilex@10.42.0.152 agilex@lm2935.local)  # lm2591.local: mDNS not answering (2026-10-23)

for r in "${ROBOTS[@]}"; do
    ssh -o ConnectTimeout=3 -o BatchMode=yes "$r" '~/swarp_ws/stop.sh' 2>&1 | sed "s|^|[$r] |" | grep -vE "quantum|store now|upgraded" &
done
if docker ps --format '{{.Names}}' | grep -qx swarp_real; then
    docker exec swarp_real bash -c 'source /opt/ros/humble/setup.bash; ros2 topic pub --once -w 0 --qos-durability transient_local --qos-reliability reliable /estop std_msgs/msg/Bool "{data: true}" > /dev/null' &
    docker exec swarp_real pkill -INT -f 'laptop.launch.py|policy_node|operator|scripts/real/explore.py|scripts/real/localize.py' &
fi
wait
echo "stop sent to: ${ROBOTS[*]}"
