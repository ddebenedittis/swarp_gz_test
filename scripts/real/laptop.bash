#!/bin/bash
# Laptop side of the real experiments, run from the laptop host. Everything ROS runs in the Humble
# container (the robots' 2023 Humble and Foxy crash on Jazzy's discovery data).
#
#   scripts/real/laptop.bash container             (re)create the container (X11) and build the workspace
#   scripts/real/laptop.bash py SCRIPT [ARGS]      run a python script in the container (scripts/real/*.py)
#   scripts/real/laptop.bash amcl "x,y,yaw;x,y,yaw" MAP_TO_ARENA    (re)start both AMCLs + amcl_relay (yaw in rad),
#                                                  wait until active, seed them there and check they answer
#   scripts/real/laptop.bash rviz                  RViz (limo_real/config/real.rviz) + the viz markers
#   scripts/real/laptop.bash operator              operator console reading the pipe out/real/op.fifo
#   scripts/real/laptop.bash op "CMD"              send a console command (goal X Y DEG | start | stop | estop | release | status)
#   scripts/real/laptop.bash policies [remap]      both policy nodes on /robot_i/cmd_vel (the bridges' caps decide:
#                                                  robots.bash caps shadow|live), episodes recorded; return when loaded.
#                                                  remap: output on /robot_i/cmd_vel_shadow instead (bridges untouched)
#   scripts/real/laptop.bash bag NAME              ros2 bag of the episode topics into RUN_DIR/NAME; return once it records robot_2
#   scripts/real/laptop.bash slam ROBOT            slam_toolbox on /ROBOT/scan (autonomous mapping)
#   scripts/real/laptop.bash slam-save NAME        save the SLAM map to out/real/maps/NAME.{pgm,yaml}
#   scripts/real/laptop.bash kill WHAT             stop: amcl | rviz | operator | policies | bag | slam | all
set -euo pipefail
source "$(dirname "$(realpath "$0")")/config.bash"
cd "${WS_DIR}"

SRC="source /opt/ros/humble/setup.bash; source ${WS_DIR}/install_humble/setup.bash; cd ${WS_DIR}"
run_bg() { local log=$1; shift; docker exec -d "${CONTAINER}" bash -c "${SRC}; exec $* > ${log} 2>&1"; }
LOGS="${WS_DIR}/out/real/logs"; mkdir -p "${LOGS}" "${RUN_DIR}" "${STATE_DIR}" "${MAPS_DIR}"
FIFO="${WS_DIR}/out/real/op.fifo"

cmd=${1:-}; shift || true
case "$cmd" in
container)
    XAUTH=/tmp/.docker.xauth
    [ -d "${XAUTH}" ] && rm -rf "${XAUTH}"; : > "${XAUTH}"
    xauth nlist "${DISPLAY}" | sed -e 's/^..../ffff/' | xauth -f "${XAUTH}" nmerge -; chmod 644 "${XAUTH}"
    xhost +local:docker > /dev/null
    docker rm -f "${CONTAINER}" > /dev/null 2>&1 || true
    gpu=(); docker info 2>/dev/null | grep -qE 'Runtimes:.*nvidia' && gpu=(--gpus all --env NVIDIA_DRIVER_CAPABILITIES=all)
    [ -d /dev/dri ] && gpu+=(--device /dev/dri)
    docker run -d --name "${CONTAINER}" --net=host --volume /dev/shm:/dev/shm \
        --env DISPLAY="${DISPLAY}" --env XAUTHORITY="${XAUTH}" --env QT_X11_NO_MITSHM=1 \
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume "${XAUTH}:${XAUTH}:ro" "${gpu[@]}" \
        --env WS_DIR="${WS_DIR}" --env ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
        --volume "${WS_DIR}:${WS_DIR}" --volume "$(realpath "${WS_DIR}/../swarp"):/opt/swarp:ro" --workdir "${WS_DIR}" \
        "${IMAGE}" sleep infinity > /dev/null
    # colcon with the venv off PATH (its setuptools breaks ament_cmake_python); no Gazebo packages here.
    docker exec "${CONTAINER}" bash -c "cd ${WS_DIR}; PATH=/opt/ros/humble/bin:/usr/bin:/bin \
        PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages \
        bash -c 'source /opt/ros/humble/setup.bash; colcon build --symlink-install --build-base build_humble \
        --install-base install_humble --packages-ignore limo_gz_description pusht_gz_plugins pusht_gz' 2>&1 | tail -1"
    ;;
py)
    docker exec -i -e CHECKPOINT="${CHECKPOINT}" -e ARENA_HALF="${ARENA_HALF}" -e MAP="${MAP}" \
        -e ARENA_CENTRE="${ARENA_CENTRE}" -e ARENA_YAW="${ARENA_YAW}" "${CONTAINER}" bash -c "${SRC}; python3 $*" ;;
amcl)
    poses=$1; m2a=$2
    docker exec "${CONTAINER}" pkill -INT -f poses.launch.py || true
    for _ in $(seq 30); do docker exec "${CONTAINER}" pgrep -f nav2_amcl/amcl > /dev/null || break; sleep 1; done
    rm -f "${LOGS}/amcl.log"
    run_bg "${LOGS}/amcl.log" ros2 launch limo_real poses.launch.py pose_source:=amcl amcl_onboard:=false \
        map:="${MAP}" "'initial_poses:=${poses}'" "map_to_arena:=${m2a}"
    for k in $(seq 60); do [ "$(cat "${LOGS}/amcl.log" 2>/dev/null | grep -c 'Managed nodes are active')" -ge 2 ] && break; sleep 1; done
    echo "AMCL up after ${k} s (log ${LOGS}/amcl.log)"
    "$0" py scripts/real/localize.py seed "'--from=${poses}'"
    ;;
rviz)
    docker exec "${CONTAINER}" pkill -f limo_real/viz && sleep 1 || true  # restart: it may run with an old arena_half
    run_bg "${LOGS}/viz.log" ros2 run limo_real viz --ros-args -p checkpoint:="${CHECKPOINT}" -p arena_half:=${ARENA_HALF}
    docker exec "${CONTAINER}" pkill -f rviz2 || true
    run_bg "${LOGS}/rviz.log" rviz2 -d "${WS_DIR}/install_humble/limo_real/share/limo_real/config/real.rviz"
    ;;
operator)
    docker exec "${CONTAINER}" bash -c "pkill -f '[s]leep infinity > ${FIFO}'; pkill -f '[l]imo_real/operator'; true"
    rm -f "${FIFO}" "${LOGS}/operator.log"
    docker exec -d "${CONTAINER}" bash -c "${SRC}; mkfifo ${FIFO}; (sleep infinity > ${FIFO} &); \
        exec ros2 run limo_real operator --ros-args -p checkpoint:=${CHECKPOINT} -p out:=${RUN_DIR} < ${FIFO} > ${LOGS}/operator.log 2>&1"
    for _ in $(seq 60); do grep -q "push-t>" "${LOGS}/operator.log" 2>/dev/null && break; sleep 0.5; done
    echo "operator up: ${LOGS}/operator.log"
    ;;
op)
    echo "$*" > "${FIFO}"; sleep 1; tail -n 12 "${LOGS}/operator.log" ;;
policies)
    mode=${1:-}
    docker exec "${CONTAINER}" pkill -INT -f policy_node || true
    for _ in $(seq 30); do docker exec "${CONTAINER}" pgrep -f policy_node > /dev/null || break; sleep 0.2; done
    rm -f "${LOGS}"/policy_{1,2}.log
    for i in 1 2; do
        extra="-p record_dir:=${RUN_DIR}/rec"; [ "$mode" = remap ] && extra="-r /robot_${i}/cmd_vel:=/robot_${i}/cmd_vel_shadow"
        run_bg "${LOGS}/policy_${i}.log" ros2 run pusht_policy policy_node --ros-args -r __node:=policy_robot_${i} ${extra} \
            -p checkpoint:="${CHECKPOINT}" -p robot_index:=${i} -p n_robots:=2 -p lidar_fov:=${LIDAR_FOV} \
            -p arena_half:=${ARENA_HALF} \
            -p lidar_mount_x:=0.103 -p max_pose_age:=0.05
    done
    for k in $(seq 120); do
        [ "$(cat "${LOGS}"/policy_{1,2}.log 2>/dev/null | grep -c ' of 2, role ')" -ge 2 ] && break; sleep 0.5
    done
    echo "policies ${mode:-on /robot_i/cmd_vel} loaded after $((k / 2)) s: ${LOGS}/policy_{1,2}.log"
    ;;
bag)
    rm -f "${LOGS}/bag.log"
    docker exec -d "${CONTAINER}" bash -c "${SRC}; cd ${RUN_DIR}; exec ros2 bag record -o $1 /tf /tf_static \
        /robot_1/scan /robot_2/scan /robot_1/pose /robot_2/pose /robot_1/amcl_pose /robot_2/amcl_pose \
        /robot_1/cmd_vel /robot_2/cmd_vel /robot_1/cmd_vel_gated /robot_2/cmd_vel_gated \
        /robot_1/bridge_state /robot_2/bridge_state /robot_1/odom /robot_2/odom /goal /episode /estop > ${LOGS}/bag.log 2>&1"
    # a new process sees robot_2 (unicast Cyclone) only at its next announcement (SPDPInterval, robots.bash install)
    for k in $(seq 60); do [ "$(cat "${LOGS}/bag.log" 2>/dev/null | grep -c "Subscribed to topic '/robot_2/")" -ge 4 ] && break; sleep 1; done
    echo "bag recording into ${RUN_DIR}/$1, robot_2 topics after ${k} s"
    ;;
slam)
    r=$1
    docker exec "${CONTAINER}" pkill -INT -f slam_toolbox || true
    run_bg "${LOGS}/slam.log" ros2 run slam_toolbox async_slam_toolbox_node --ros-args \
        -p odom_frame:=${r}/odom -p base_frame:=${r}/base_footprint -p map_frame:=map \
        -p scan_topic:=/${r}/scan -p resolution:=0.05 -p max_laser_range:=10.0 -p minimum_travel_distance:=0.1 \
        -p minimum_travel_heading:=0.1 -p mode:=mapping -p use_sim_time:=false
    echo "slam_toolbox on /${r}/scan: ${LOGS}/slam.log"
    ;;
slam-save)
    docker exec "${CONTAINER}" bash -c "${SRC}; ros2 run nav2_map_server map_saver_cli -f ${MAPS_DIR}/$1 --ros-args -p save_map_timeout:=10.0" 2>&1 | tail -2
    ls -la "${MAPS_DIR}/$1".*
    ;;
kill)
    case "$1" in
        amcl) docker exec "${CONTAINER}" pkill -INT -f poses.launch.py ;;
        rviz) docker exec "${CONTAINER}" pkill -f 'rviz2|limo_real/viz' ;;
        operator) docker exec "${CONTAINER}" bash -c "pkill -f 'sleep infinity > ${FIFO}'; pkill -INT -f limo_real/operator" ;;
        policies) docker exec "${CONTAINER}" pkill -INT -f policy_node ;;
        bag) docker exec "${CONTAINER}" pkill -INT -f 'ros2 bag record' ;;
        slam) docker exec "${CONTAINER}" pkill -INT -f slam_toolbox ;;
        all) for w in policies operator bag slam amcl rviz; do "$0" kill $w || true; done ;;
    esac || true
    ;;
*) sed -n 2,17p "$0"; exit 2 ;;
esac
