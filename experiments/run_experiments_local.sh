#!/bin/bash
# Run every scenario in experiments/benchmark_local.json under every policy
# arm, on the bundled local simulator. No CARLA, no GPU, no server.
#
#   ./experiments/run_experiments_local.sh [OUT_DIR] [--no-video]
#   REPEATS=5 ./experiments/run_experiments_local.sh results/local --no-video
#
# The local simulator is deterministic -- fixed step, no physics substepping,
# no server -- so REPEATS>1 reproduces identical numbers rather than sampling
# a distribution. It exists here only to mirror the CARLA runner's interface;
# use it to check that, not to average.
#
# For statistics, sample the POLICY instead of repeating the simulator:
#     python experiments/sweep_idm_local.py -n 64 --check-determinism
# which draws a Latin hypercube over the six IDM parameters and feeds the same
# report generator.
#
# Writes, per run:
#   OUT_DIR/<scenario>__<policy>[__rN].json   metrics summary
#   OUT_DIR/<scenario>__<policy>[__rN].mp4    bird's-eye recording
#   OUT_DIR/<scenario>__<policy>[__rN].log    stderr trace
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT}/experiments/results_local}"
VIDEO=1
[ "${2:-}" = "--no-video" ] && VIDEO=0

# shellcheck disable=SC1091
source "${ROOT}/env.sh"
PYTHON="${PYTHON:-python3}"
cd "${ROOT}"

if ! "${PYTHON}" -c "import pygame" >/dev/null 2>&1; then
    echo "ERROR: this interpreter ($("${PYTHON}" -V 2>&1)) has no pygame." >&2
    echo "       pip install pygame     # or set PYTHON=/path/to/python" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

# scenario:duration:junction-turn  (see benchmark_local.json for why each
# scenario needs its own manoeuvre preference)
SCENARIOS=(
    "red_light:18:straight"
    "right_turn:24:right"
    "left_turn:20:left"
    "stop_sign:26:straight"
)
POLICIES=(
    "scripted:"
    "idm:--ego-policy idm"
)

REPEATS="${REPEATS:-1}"
fail=0
for entry in "${SCENARIOS[@]}"; do
    name="${entry%%:*}"
    rest="${entry#*:}"
    dur="${rest%%:*}"
    turn="${rest##*:}"
    for pol in "${POLICIES[@]}"; do
        pid="${pol%%:*}"
        pargs="${pol#*:}"
        for rep in $(seq 1 "${REPEATS}"); do
            if [ "${REPEATS}" -gt 1 ]; then
                tag="${name}__${pid}__r${rep}"
            else
                tag="${name}__${pid}"
            fi
            echo "=== ${tag} (${dur}s, --junction-turn ${turn}) ==="
            # shellcheck disable=SC2086
            set -- "${PYTHON}" -m osc2carla "scenarios/local/benchmark/${name}.osc" \
                --backend pygame --town grid --junction-turn "${turn}" \
                --sim-duration "${dur}" \
                --record-actor ego \
                --metrics-out "${OUT_DIR}/${tag}.json" \
                ${pargs}
            if [ "${VIDEO}" -eq 1 ]; then
                set -- "$@" --render-mode headless \
                    --record-video "${OUT_DIR}/${tag}.mp4" --record-fps 20 \
                    --record-width 1280 --record-height 720
            else
                set -- "$@" --render-mode off
            fi
            "$@" > "${OUT_DIR}/${tag}.log" 2>&1
            rc=$?
            if [ "${rc}" -ne 0 ]; then
                echo "    FAILED rc=${rc}; tail of log:"
                tail -15 "${OUT_DIR}/${tag}.log" | sed 's/^/      /'
                fail=1
            else
                grep -E "metrics ->" "${OUT_DIR}/${tag}.log" \
                    | sed 's|.*metrics -> [^ ]* |    |' | sed 's/^/   /'
            fi
        done
    done
done

echo
echo "results in ${OUT_DIR}"
exit "${fail}"
