# Sourced by every interactive shell (and by `docker exec ... bash`): ROS + the workspace overlay.
source /opt/ros/jazzy/setup.bash
if [ -n "${WS_DIR}" ] && [ -f "${WS_DIR}/install/setup.bash" ]; then
    source "${WS_DIR}/install/setup.bash"
fi
