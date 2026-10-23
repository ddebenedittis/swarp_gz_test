#!/bin/bash
# Session bringup in one call; nothing moves (the bridges end in shadow: nothing forwarded).
#
#   scripts/real/bringup.bash start       (re)start both robot stacks in dry first (resets odometry), then all below
#   scripts/real/bringup.bash             stacks already up: laptop side + localization
#   scripts/real/bringup.bash localize    localization only (robots moved by hand; laptop side untouched)
#
# The localization (scan match tee -> AMCL seeded there -> check verify) runs while the laptop side
# (caps shadow, RViz + viz, operator, policies, op release) starts in parallel: they do not depend on each other.
# Then read out/real/state/overlay.png. Exit 1 if the localization failed.
set -uo pipefail
source "$(dirname "$(realpath "$0")")/config.bash"
R="${REAL_DIR}"
LOGS="${WS_DIR}/out/real/logs"; mkdir -p "${LOGS}"
what=${1:-}
t0=${SECONDS}

if [ "$what" = start ]; then "$R/robots.bash" start dry || exit 1; fi

if [ "$what" != localize ]; then
    (
        "$R/robots.bash" caps shadow &
        "$R/laptop.bash" rviz &
        "$R/laptop.bash" policies &
        "$R/laptop.bash" operator && "$R/laptop.bash" op release > /dev/null
        wait
    ) > "${LOGS}/bringup_laptop.log" 2>&1 &
    laptop=$!
fi

loc() {
    "$R/laptop.bash" py scripts/real/localize.py "$@" 2>&1 | grep -vE "Warning|warnings.warn|nanmedian"
    return "${PIPESTATUS[0]}"
}
ok=1
echo "== localization (map ${MAP##*/}, arena ${ARENA_CENTRE} at ${ARENA_YAW} deg, ARENA_HALF ${ARENA_HALF})"
if loc scan match tee; then
    read -r P M < <("$R/laptop.bash" py scripts/real/localize.py args 2>/dev/null | paste -sd' ')
    "$R/laptop.bash" amcl "$P" "$M" || ok=0
    out=$(loc check verify) || ok=0
    echo "${out}" | grep -vE "scans, "
    echo "${out}" | grep -qE "CHECK|no amcl" && ok=0
else
    ok=0
fi

if [ "$what" != localize ]; then
    wait "${laptop}"
    echo "== laptop side"
    grep -E "set dry_run|operator up|loaded after" "${LOGS}/bringup_laptop.log"
    "$R/laptop.bash" op status | grep -E "^\s+robot_.( gate)?:" | sort -u
fi
echo "== bringup $((SECONDS - t0)) s; overlay: ${WS_DIR}/out/real/state/overlay.png"
[ "$ok" = 1 ] || { echo "LOCALIZATION FAILED: see above"; exit 1; }
