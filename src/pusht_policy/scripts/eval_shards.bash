#!/bin/bash
# Run a Gazebo evaluation as N parallel headless containers, each with its own ROS domain,
# Gazebo partition and seed, then aggregate the CSVs. Run on the host from anywhere.
#
# Usage: src/pusht_policy/scripts/eval_shards.bash <checkpoint> <shards> <episodes per shard>
#            [run name] [base seed] [extra launch args ...]
#
# The checkpoint path must be visible in the container: inside this repo, or under ../swarp
# (give it as /opt/swarp/...). Shard k writes out/<run name>/shard<k>/episodes.csv.
set -euo pipefail

WS_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../..")"
COMPOSE=(docker compose -f "${WS_DIR}/docker/docker-compose.yaml")
CKPT="$1"; SHARDS="$2"; EPISODES="$3"
RUN="${4:-$(basename "${CKPT%.pt}")_$(date +%Y%m%d-%H%M%S)}"
BASE_SEED="${5:-0}"
shift $(( $# < 5 ? $# : 5 ))
EXTRA=("$@")

# Build the workspace once; the shards then share it without rebuilding.
"${COMPOSE[@]}" run --rm -e COLCON_BUILD=always eval true

names=()
for k in $(seq 0 $((SHARDS - 1))); do
    name="pusht_eval_${RUN//[^A-Za-z0-9_.-]/_}_${k}"
    "${COMPOSE[@]}" run -d --name "${name}" \
        -e COLCON_BUILD=never \
        -e ROS_DOMAIN_ID=$((30 + k)) \
        -e GZ_PARTITION="${name}" \
        eval ros2 launch pusht_policy eval.launch.py \
        checkpoint:="${CKPT}" episodes:="${EPISODES}" seed:=$((BASE_SEED + k)) \
        run_name:="${RUN}/shard${k}" "${EXTRA[@]}" > /dev/null
    names+=("${name}")
    echo "started ${name} (seed $((BASE_SEED + k)), ROS_DOMAIN_ID $((30 + k)))"
done

for name in "${names[@]}"; do
    code="$(docker wait "${name}")"
    echo "${name} exited ${code}"
    docker logs "${name}" > "${WS_DIR}/out/${RUN}/$(echo "${name}" | sed 's/.*_//').log" 2>&1 || true
    docker rm "${name}" > /dev/null
done

python3 -m pusht_policy.aggregate "${WS_DIR}/out/${RUN}" 2>/dev/null \
    || PYTHONPATH="${WS_DIR}/src/pusht_policy" python3 "${WS_DIR}/src/pusht_policy/pusht_policy/aggregate.py" "${WS_DIR}/out/${RUN}"
