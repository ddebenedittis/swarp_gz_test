#!/bin/bash
# Robot side of the real experiments, run from the laptop host.
#
#   scripts/real/robots.bash install                 copy + build limo_bridge into ~/swarp_ws, write run/stop scripts
#   scripts/real/robots.bash start MODE [i ...]      MODE: dry (shadow) | live (episode caps) | slow (0.15/0.6) | map (slow, arena stop off)
#   scripts/real/robots.bash stop [i ...]            ~/swarp_ws/stop.sh (robot_bridge sends zeros first)
#   scripts/real/robots.bash status                  uptime, clock offset, running nodes, deaths in robot.log
#
# i = 1, 2 (default: both). Restarting a robot resets its odometry: restart the laptop AMCL after it
# (scripts/real/laptop.bash amcl ...).
set -euo pipefail
source "$(dirname "$(realpath "$0")")/config.bash"

quiet() { grep -vE "post-quantum|store now|upgraded" || true; }
pick() { if [ $# -eq 0 ]; then echo 1 2; else echo "$@"; fi; }

install_robot() {
    local k=$1 h=${ROBOT_HOSTS[$1]} distro=${ROBOT_DISTROS[$1]} agx=${ROBOT_AGX_WS[$1]} rmw=""
    ssh "$h" 'mkdir -p ~/swarp_ws/src' 2>&1 | quiet
    rsync -a --delete --exclude __pycache__ "${WS_DIR}/src/limo_bridge/" "$h:swarp_ws/src/limo_bridge/" 2>&1 | quiet
    if [ "${ROBOT_RMW[$k]}" = cyclone ]; then
        rmw='export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$HOME/swarp_ws/cyclonedds.xml'
        ssh "$h" "cat > ~/swarp_ws/cyclonedds.xml" 2>&1 <<EOF | quiet
<?xml version="1.0" encoding="UTF-8"?>
<!-- Unicast discovery with the laptop only: Foxy's Cyclone 0.7 segfaults on the multicast-discovered
     Fast DDS participant at 10.42.0.1 and on shared-memory locators. -->
<CycloneDDS>
  <Domain id="any">
    <General>
      <NetworkInterfaceAddress>wlan0</NetworkInterfaceAddress>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>60</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="${LAPTOP_IP}"/>
        <Peer address="127.0.0.1"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
EOF
    fi
    ssh "$h" "cat > ~/swarp_ws/run_robot.sh" 2>&1 <<EOF | quiet
#!/bin/bash
# Starts limo_bridge robot.launch.py in the background; args go to the launch. Log: ~/swarp_ws/robot.log
source /opt/ros/${distro}/setup.bash
source ${agx}/install/setup.bash
source ~/swarp_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}
${rmw}
cd ~/swarp_ws
setsid nohup ros2 launch limo_bridge robot.launch.py "\$@" > ~/swarp_ws/robot.log 2>&1 < /dev/null &
echo \$! > ~/swarp_ws/robot.pid
echo started pid \$(cat ~/swarp_ws/robot.pid)
EOF
    # The patterns must never appear in the command line that runs stop.sh (pkill -f would kill it).
    ssh "$h" "cat > ~/swarp_ws/stop.sh" 2>&1 <<'EOF' | quiet
#!/bin/bash
# Stop the robot: SIGINT to robot_bridge (it sends zeros to the chassis), then to the whole launch.
pkill -INT -f 'limo_bridge/robot_bridge' ; sleep 0.3
[ -f ~/swarp_ws/robot.pid ] && kill -INT -- -$(cat ~/swarp_ws/robot.pid) 2>/dev/null
pkill -INT -f 'robot.launch.py'; sleep 2
pkill -KILL -f 'robot.launch.py|limo_bridge/robot_bridge|ydlidar_ros2_driver_node|nav2_amcl/lib|nav2_map_server/lib|nav2_lifecycle_manager/lib' 2>/dev/null
echo stopped; pgrep -af 'limo_base|robot_bridge|ydlidar|amcl' | grep -v pgrep || true
EOF
    ssh "$h" "chmod +x ~/swarp_ws/*.sh; source /opt/ros/${distro}/setup.bash; source ${agx}/install/setup.bash; cd ~/swarp_ws && colcon build --packages-select limo_bridge 2>&1 | tail -1" 2>&1 | quiet
}

start_robot() {
    local k=$1 mode=$2 h=${ROBOT_HOSTS[$1]} i=$(( $1 + 1 )) args
    args="robot_name:=robot_${i} port_name:=${ROBOT_PORTS[$k]} loc_half:=${LOC_HALF}"
    case "$mode" in
        dry)  args+=" dry_run:=true arena_half:=${ARENA_HALF}" ;;
        live) args+=" max_v:=${EPISODE_MAX_V} max_w:=${EPISODE_MAX_W} arena_half:=${ARENA_HALF}" ;;
        slow) args+=" max_v:=0.15 max_w:=0.6 arena_half:=${ARENA_HALF}" ;;
        map)  args="robot_name:=robot_${i} port_name:=${ROBOT_PORTS[$k]} max_v:=0.15 max_w:=0.6 arena_half:=0 loc_half:=0" ;;
        *) echo "unknown mode $mode" >&2; exit 2 ;;
    esac
    ssh "$h" "~/swarp_ws/stop.sh > /dev/null; ~/swarp_ws/run_robot.sh ${args}" 2>&1 | quiet
}

cmd=${1:-}; shift || true
case "$cmd" in
    install) for i in $(pick "$@"); do echo "== robot_$i"; install_robot $((i - 1)); done ;;
    start)   mode=$1; shift; for i in $(pick "$@"); do echo "== robot_$i ($mode)"; start_robot $((i - 1)) "$mode" & done; wait ;;
    stop)    for i in $(pick "$@"); do ssh "${ROBOT_HOSTS[$((i - 1))]}" '~/swarp_ws/stop.sh' 2>&1 | quiet | sed "s/^/[robot_$i] /" & done; wait ;;
    status)
        for i in 1 2; do
            h=${ROBOT_HOSTS[$((i - 1))]}
            echo "== robot_$i $h: $(ssh -o ConnectTimeout=5 "$h" 'uptime -p; echo "nodes: $(pgrep -fc "limo_base|robot_bridge|ydlidar_ros2")"; echo "deaths: $(grep -c died ~/swarp_ws/robot.log 2>/dev/null)"' 2>&1 | quiet | tr '\n' ' ')"
        done
        python3 - "${ROBOT_HOSTS[@]}" <<'EOF'
import subprocess, sys, time
for h in sys.argv[1:]:
    p = subprocess.Popen(["ssh", h, "python3 -u -c 'import sys,time\nfor l in sys.stdin: print(repr(time.time()), flush=True)'"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    res = []
    for _ in range(20):
        t0 = time.time(); p.stdin.write("x\n"); p.stdin.flush(); r = float(p.stdout.readline()); t1 = time.time()
        res.append((t1 - t0, r - (t0 + t1) / 2))
    p.stdin.close(); p.wait(); res.sort()
    print(f"{h}: clock offset {res[0][1] * 1e3:+.1f} ms (keep within ~20 ms: policy pairs poses within 50 ms)")
EOF
        ;;
    *) sed -n 2,12p "$0"; exit 2 ;;
esac
