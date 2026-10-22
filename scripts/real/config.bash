# Lab setup for the real two-Limo experiments; sourced by the scripts in scripts/real/.
# robot_i is the policy's agent i-1; the arrays are indexed 0 = robot_1, 1 = robot_2.
REAL_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
WS_DIR="$(realpath "${REAL_DIR}/../..")"

ROBOT_HOSTS=(agilex@lm2591.local agilex@lm2935.local)
ROBOT_DISTROS=(humble foxy)                                   # Limo ROS2 (x86, Humble) | Limo Pro (Orin Nano, Foxy)
ROBOT_AGX_WS=(/home/agilex/agilex_ws /home/agilex/limo_ros2_ws)  # AgileX workspace: limo_base, ydlidar driver
ROBOT_PORTS=(ttylimo ttyTHS0)                                 # chassis serial port (the Orin Nano's is ttyTHS0, not ttyTHS1)
# Foxy's Fast DDS aborts against newer peers; its Cyclone 0.7 segfaults on multicast discovery of the
# Fast DDS participant at 10.42.0.1 and on shared-memory locators: unicast Cyclone to the laptop only.
ROBOT_RMW=("" "cyclone")
LAPTOP_IP=10.42.0.44
export ROS_DOMAIN_ID=28

CHECKPOINT=/opt/swarp/runs/remote-davide-cf/L8b-s2-slim240-s0/pusht_lidar_limo_slim240_s2_best.pt
MAPS_DIR="${WS_DIR}/out/real/maps"
MAP="${MAP:-${MAPS_DIR}/map_corridor.yaml}"
STATE_DIR="${WS_DIR}/out/real/state"      # scans, poses, T fit, arena (localize.py)
RUN_DIR="${RUN_DIR:-${WS_DIR}/out/real/run_$(date +%Y%m%d)}"

CONTAINER=swarp_real                      # Humble laptop container (docker/Dockerfile.humble)
IMAGE=swarp_real_humble

# Episode / manoeuvre limits.
EPISODE_MAX_V=0.3
EPISODE_MAX_W=1.0
ARENA_HALF=1.5          # robot_bridge arena stop at ARENA_HALF + 0.25 from the arena centre
LOC_HALF=1.8            # scan_loc mask (AMCL input): keep it inside the nearest wall
LIDAR_FOV=4.18879       # L8b's trained 240 deg; the drivers deliver -116..+108 deg
