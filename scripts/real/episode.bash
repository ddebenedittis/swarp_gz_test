#!/bin/bash
# One live episode in one call (MOTION: ask the user first). The goal must already be set (op "goal ..." or RViz).
#
#   [EXP=exp_name [VIDEO=n]] scripts/real/episode.bash NAME [DUR]
#                                            bag NAME, caps live, op start, monitor until the episode ends
#                                            (90 s timeout, op "stop ok|fail") or DUR s (default 100)
#
# With EXP, the episode is then saved to ~/Documents/swarp_exp/<date>/<EXP>_<id> (save_exp.bash).
#
# A robot PINNED at the arena stop for ~9 s stops the episode by itself (monitor.py --stop-pinned).
# Whatever happens (end, error, Ctrl-C, kill), on exit: caps dry, bag stopped.
set -uo pipefail
source "$(dirname "$(realpath "$0")")/config.bash"
R="${REAL_DIR}"
name=${1:?usage: episode.bash NAME [DUR]}; dur=${2:-100}

cleanup() {
    trap - EXIT INT TERM
    "$R/robots.bash" caps dry
    "$R/laptop.bash" kill bag
    echo "== episode ${name} over: bridges dry, bag stopped (${RUN_DIR}/${name})"
    [ -z "${EXP:-}" ] || "$R/save_exp.bash" "${EXP}" "${name}" "${VIDEO:--}"
}
trap cleanup EXIT INT TERM

"$R/laptop.bash" bag "${name}"
"$R/robots.bash" caps live || exit 1
"$R/laptop.bash" op start | tail -n 2
"$R/laptop.bash" py scripts/real/monitor.py "${dur}" 3 --stop-pinned | tee "${RUN_DIR}/${name}_monitor.log"
"$R/laptop.bash" op status | grep -m3 -E "episode|goal"
