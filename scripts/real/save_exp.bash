#!/bin/bash
# Copy one real episode to the experiment archive and add its entry to the day's summary.
#
#   scripts/real/save_exp.bash EXP_NAME BAG [VIDEO] [EPISODE]
#
# EXP_NAME: short name of the experiment (e.g. l9c_a125); BAG: the episode's bag name in RUN_DIR (episode.bash NAME);
# VIDEO: the user's video number (or -); EPISODE: the operator's episode index (default: the last row of episodes.csv).
# Writes EXP_DIR/<date>/<EXP_NAME>_<id> (id: next free number of the day, 01, 02, ...) with the bag, the per-step
# records, the episodes.csv row, the localization state (overlay, poses, T, arena), the laptop logs and meta.txt,
# and appends an entry to EXP_DIR/<date>/summary.md whose `Synthesis:` line is to be written afterwards.
set -euo pipefail
source "$(dirname "$(realpath "$0")")/config.bash"
EXP_DIR="${EXP_DIR:-${HOME}/Documents/swarp_exp}"
name=${1:?usage: save_exp.bash EXP_NAME BAG [VIDEO] [EPISODE]}; bag=${2:?bag name}; video=${3:--}; ep=${4:-}
csv="${RUN_DIR}/episodes.csv"
[ -f "${csv}" ] || { echo "no ${csv} (RUN_DIR?)"; exit 1; }
[ -n "${ep}" ] || ep=$(python3 -c "import csv,sys; print(list(csv.DictReader(open(sys.argv[1])))[-1]['episode'])" "${csv}")

day="${EXP_DIR}/$(date +%F)"; mkdir -p "${day}"
last=$(find "${day}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sed -nE 's/.*_([0-9]+)$/\1/p' | sort -n | tail -1)
id=$(printf '%02d' $((10#${last:-0} + 1)))
out="${day}/${name}_${id}"; mkdir -p "${out}/rec" "${out}/state" "${out}/logs"

[ -d "${RUN_DIR}/${bag}" ] && cp -r "${RUN_DIR}/${bag}" "${out}/bag" || echo "no bag ${RUN_DIR}/${bag}"
cp "${RUN_DIR}"/rec/ep"$(printf '%04d' "${ep}")"_* "${out}/rec/" 2>/dev/null || echo "no rec/ for episode ${ep}"
[ -f "${RUN_DIR}/${bag}_monitor.log" ] && cp "${RUN_DIR}/${bag}_monitor.log" "${out}/monitor.log"
python3 - "${csv}" "${ep}" > "${out}/episode.csv" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
w = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys()); w.writeheader()
w.writerows(r for r in rows if r["episode"] == sys.argv[2])
EOF
cp "${STATE_DIR}"/{overlay.png,poses.json,tee.json,arena.json} "${out}/state/" 2>/dev/null || true
cp "${WS_DIR}"/out/real/logs/*.log "${out}/logs/"
cat > "${out}/meta.txt" <<EOF
experiment ${name}_${id}, saved $(date '+%F %T')
video ${video}
episode ${ep} of ${RUN_DIR}, bag ${bag}
checkpoint ${CHECKPOINT}
ARENA_HALF ${ARENA_HALF}, arena centre ${ARENA_CENTRE} at ${ARENA_YAW} deg, map ${MAP}
LIDAR_FOV ${LIDAR_FOV}, caps ${EPISODE_MAX_V} m/s / ${EPISODE_MAX_W} rad/s
swarp_gz_test $(git -C "${WS_DIR}" rev-parse --short HEAD)$(git -C "${WS_DIR}" diff --quiet || echo +dirty)
EOF

# summary.md entry: the facts from episodes.csv; the synthesis is written afterwards (skill, step 4)
python3 - "${out}/episode.csv" "${day}/summary.md" "${name}_${id}" "${video}" "${CHECKPOINT##*/}" "${ARENA_HALF}" <<'EOF'
import csv, os, sys, time
src, md, exp, video, ckpt, half = sys.argv[1:]
r = next(csv.DictReader(open(src)), None) or {"tee_start": ""}
new = not os.path.exists(md)
with open(md, "a") as f:
    if new:
        f.write(f"# Real two-Limo Push-T experiments, {time.strftime('%Y-%m-%d')}\n")
    f.write(f"\n## {exp}\n\n"
            f"- Video: {video}\n"
            f"- Saved: {time.strftime('%H:%M')}, episode {r.get('episode', '?')}, {float(r.get('time') or 'nan'):.0f} s\n"
            f"- Policy: {ckpt}, ARENA_HALF {half} m\n"
            f"- Goal (arena x, y, yaw rad): {r.get('goal', '?')}\n"
            f"- Start: robot_1 {r.get('robot_1_start', '?')}, robot_2 {r.get('robot_2_start', '?')}, T {r.get('tee_start').strip('()') or 'not tracked'}\n"
            f"- Result: {r.get('result', '?')} (final error {r.get('final_pos_err', '?')} m / {r.get('final_ang_err', '?')} rad)\n"
            f"- Data: `{exp}/`\n"
            f"- Synthesis: TODO\n")
EOF
echo "saved ${out}; entry ${name}_${id} in ${day}/summary.md (fill in its Synthesis line)"
