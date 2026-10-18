#!/bin/bash
# Interactive dev shell with X11 (Gazebo GUI, RViz). The repo is mounted at the same absolute
# path, ../swarp read-only at /opt/swarp.
#
# Usage: docker/run.bash [command ...]   (default: bash)
set -euo pipefail

DOCKER_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
WS_DIR="$(dirname "${DOCKER_DIR}")"
SWARP_DIR="$(realpath -m "${WS_DIR}/../swarp")"
IMAGE_NAME=swarp_gz_test

# X11: a dedicated xauth file with the cookie made host-independent (ffff family), so the
# container can authenticate; only local (non-network) clients are allowed by xhost.
XAUTH=/tmp/.docker.xauth
gpu_args=()
x11_args=()
if [ -n "${DISPLAY:-}" ]; then
    [ -d "${XAUTH}" ] && rm -rf "${XAUTH}"
    : > "${XAUTH}"
    xauth nlist "${DISPLAY}" | sed -e 's/^..../ffff/' | xauth -f "${XAUTH}" nmerge -
    chmod 644 "${XAUTH}"
    xhost +local:docker > /dev/null
    x11_args=(
        --env "DISPLAY=${DISPLAY}"
        --env "XAUTHORITY=${XAUTH}"
        --env QT_X11_NO_MITSHM=1
        --volume /tmp/.X11-unix:/tmp/.X11-unix:rw
        --volume "${XAUTH}:${XAUTH}:ro"
    )
fi

if docker info 2>/dev/null | grep -qE 'Runtimes:.*nvidia'; then
    gpu_args=(--gpus all --env NVIDIA_DRIVER_CAPABILITIES=all)
fi
if [ -d /dev/dri ]; then
    gpu_args+=(--device /dev/dri)
    for g in video render; do
        gid="$(getent group "${g}" | cut -d: -f3 || true)"
        [ -n "${gid}" ] && gpu_args+=(--group-add "${gid}")
    done
fi

[ $# -eq 0 ] && set -- bash
tty_args=(-i)
[ -t 0 ] && tty_args+=(-t)

docker run "${tty_args[@]}" --rm \
    --net=host \
    --volume /dev/shm:/dev/shm \
    "${x11_args[@]}" \
    "${gpu_args[@]}" \
    --env "WS_DIR=${WS_DIR}" \
    --env "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" \
    --volume "${WS_DIR}:${WS_DIR}" \
    --volume "${SWARP_DIR}:/opt/swarp:ro" \
    --workdir "${WS_DIR}" \
    "${IMAGE_NAME}" "$@"
