#!/bin/bash
# Run scenarios on the bundled local simulator: no CARLA, no GPU, no server.
#
#   ./run_local_demo.sh                 # window if a display exists, else MP4s
#   ./run_local_demo.sh --headless      # force off-screen rendering + MP4s
#   OUT_DIR=/tmp/localrun ./run_local_demo.sh --headless
#
# Writes, per scenario, into OUT_DIR (default: local_run/):
#   <name>.mp4          bird's-eye recording
#   <name>.mp4_frames/  the PNGs it was encoded from
#   <name>.json         the same metrics summary the CARLA path writes
#   <name>.log          stderr trace
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT}/local_run}"
MODE="auto"
[ "${1:-}" = "--headless" ] && MODE="headless"

# shellcheck disable=SC1091
source "${ROOT}/env.sh"
PYTHON="${PYTHON:-python3}"

if ! "${PYTHON}" -c "import pygame" >/dev/null 2>&1; then
    echo "ERROR: this interpreter ($("${PYTHON}" -V 2>&1)) has no pygame." >&2
    echo "       pip install pygame     # or set PYTHON=/path/to/python" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

# scenario:sim_duration:camera binding
# local_crossing is written against the local `grid` town, so its conflict
# actually develops here. The paper demos use map spawn points and relative
# placement, so they transfer as-is. The scenarios/benchmark/*.osc files are
# deliberately absent: they hard-code Town10HD_Opt coordinates and would run
# without staging anything (see README, "Local simulator backend").
SCENARIOS=(
    "scenarios/local/local_crossing.osc:18:ego"
    "scenarios/closed_loop_demo.osc:22:hero"
    "scenarios/scenario_collision.osc:22:ego"
    "scenarios/simple_example.osc:16:hero"
)

fail=0
for entry in "${SCENARIOS[@]}"; do
    path="${entry%%:*}"
    rest="${entry#*:}"
    dur="${rest%%:*}"
    actor="${rest##*:}"
    name="$(basename "${path}" .osc)"
    echo "===== ${name} (${dur}s, camera on ${actor}) ====="
    "${PYTHON}" -m osc2carla "${path}" \
        --backend pygame \
        --render-mode "${MODE}" \
        --record-actor "${actor}" \
        --record-video "${OUT_DIR}/${name}.mp4" \
        --metrics-out "${OUT_DIR}/${name}.json" \
        --sim-duration "${dur}" \
        > "${OUT_DIR}/${name}.log" 2>&1
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "  FAILED rc=${rc}; tail of log:"
        tail -12 "${OUT_DIR}/${name}.log" | sed 's/^/    /'
        fail=1
    else
        grep -E "metrics ->|wrote " "${OUT_DIR}/${name}.log" | sed 's/^/  /'
    fi
done

echo
echo "results in ${OUT_DIR}"
exit "${fail}"
