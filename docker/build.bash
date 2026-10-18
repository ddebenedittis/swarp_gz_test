#!/bin/bash
# Build the swarp_gz_test image and write docker/.env for docker compose.
#
# Usage: docker/build.bash [--no-cache]
set -euo pipefail

DOCKER_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
WS_DIR="$(dirname "${DOCKER_DIR}")"
SWARP_DIR="$(realpath -m "${WS_DIR}/../swarp")"

# Use the NVIDIA runtime (EGL rendering for gpu_lidar, GUI) only when the host has it.
if docker info 2>/dev/null | grep -qE 'Runtimes:.*nvidia'; then
    DOCKER_RUNTIME=nvidia
else
    DOCKER_RUNTIME=runc
fi

# docker compose reads this file from the project directory (docker/).
cat > "${DOCKER_DIR}/.env" <<ENV
WS_DIR=${WS_DIR}
SWARP_DIR=${SWARP_DIR}
USERNAME=$(id -un)
HOST_UID=$(id -u)
HOST_GID=$(id -g)
VIDEO_GID=$(getent group video | cut -d: -f3 || echo 44)
RENDER_GID=$(getent group render | cut -d: -f3 || echo 44)
DOCKER_RUNTIME=${DOCKER_RUNTIME}
ENV
echo "Wrote ${DOCKER_DIR}/.env (runtime: ${DOCKER_RUNTIME})"

[ -d "${SWARP_DIR}" ] || echo "WARNING: ${SWARP_DIR} not found, /opt/swarp will be empty" >&2

docker compose -f "${DOCKER_DIR}/docker-compose.yaml" build "$@"
