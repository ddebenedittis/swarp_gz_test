#!/bin/bash
# Source ROS, build the workspace if needed, source it, then run the command.
#
# COLCON_BUILD selects when the workspace at $WS_DIR is (re)built:
#   auto   (default) build only when install/setup.bash is missing
#   always build on every start (a few seconds for these small packages)
#   never  never build
set -e
source /opt/ros/jazzy/setup.bash

if [ -n "${WS_DIR}" ] && [ -d "${WS_DIR}/src" ]; then
    cd "${WS_DIR}"
    mode="${COLCON_BUILD:-auto}"
    if [ "${mode}" = "always" ] || { [ "${mode}" = "auto" ] && [ ! -f install/setup.bash ]; }; then
        echo "[entrypoint] colcon build --symlink-install in ${WS_DIR}" >&2
        colcon build --symlink-install >&2
    fi
    if [ -f install/setup.bash ]; then
        source install/setup.bash
    fi
fi

exec "$@"
