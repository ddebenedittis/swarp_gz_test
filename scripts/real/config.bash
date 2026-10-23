# Lab setup for the real two-Limo experiments; sourced by the scripts in scripts/real/.
# robot_i is the policy's agent i-1; the arrays are indexed 0 = robot_1, 1 = robot_2.
REAL_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
WS_DIR="$(realpath "${REAL_DIR}/../..")"

ROBOT_HOSTS=(agilex@10.42.0.152 agilex@lm2935.local)  # lm2591.local: mDNS not answering (2026-10-23)
ROBOT_DISTROS=(humble foxy)                                   # Limo ROS2 (x86, Humble) | Limo Pro (Orin Nano, Foxy)
ROBOT_AGX_WS=(/home/agilex/agilex_ws /home/agilex/limo_ros2_ws)  # AgileX workspace: limo_base, ydlidar driver
ROBOT_PORTS=(ttylimo ttyTHS0)                                 # chassis serial port (the Orin Nano's is ttyTHS0, not ttyTHS1)
# Extra robot.launch.py args per robot, e.g. when robot_1's udev names are missing (USB on another bus after a
# reboot): ROBOT_EXTRA_1="port_name:=ttyUSB1 lidar_port:=/dev/serial/by-path/pci-0000:00:14.0-usb-0:3:1.0-port0"
ROBOT_EXTRA=("${ROBOT_EXTRA_1:-}" "${ROBOT_EXTRA_2:-}")
# Foxy's Fast DDS aborts against newer peers; its Cyclone 0.7 segfaults on multicast discovery of the
# Fast DDS participant at 10.42.0.1 and on shared-memory locators: unicast Cyclone to the laptop only.
ROBOT_RMW=("" "cyclone")
LAPTOP_IP=10.42.0.44
export ROS_DOMAIN_ID=28

# L9c (recipe pusht_lidar_limo_slim240_arena): 2-5 m arenas, told this one (ARENA_HALF), 40 rays over
# +-110 deg. L8b (fixed 4 x 4 m, 240 deg): CHECKPOINT=/opt/swarp/runs/remote-davide-cf/L8b-s2-slim240-s0/\
# pusht_lidar_limo_slim240_s2_best.pt with LIDAR_FOV=4.18879.
CHECKPOINT="${CHECKPOINT:-/opt/swarp/runs/remote-davide-cf/L9c-arena-s0/pusht_lidar_limo_slim240_arena_best.pt}"
MAPS_DIR="${WS_DIR}/out/real/maps"
MAP="${MAP:-${MAPS_DIR}/map_corridor2.yaml}"
STATE_DIR="${WS_DIR}/out/real/state"      # scans, poses, T fit, arena (localize.py)
RUN_DIR="${RUN_DIR:-${WS_DIR}/out/real/run_$(date +%Y%m%d)}"

CONTAINER=swarp_real                      # Humble laptop container (docker/Dockerfile.humble)
IMAGE=swarp_real_humble

# Episode / manoeuvre limits.
EPISODE_MAX_V=0.3
EPISODE_MAX_W=1.0
ARENA_HALF="${ARENA_HALF:-1.0}"         # robot_bridge arena stop at ARENA_HALF + 0.25; L9c observes it and labels to ARENA_HALF + 0.5
LOC_HALF=1.8            # scan_loc mask (AMCL input): keep it inside the nearest wall
LIDAR_FOV="${LIDAR_FOV:-3.83972}"  # L9c's trained 220 deg (L8b: 4.18879); the drivers deliver -116..+108 deg
