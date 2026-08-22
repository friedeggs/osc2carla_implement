#!/bin/bash -l
# Run every scenario in experiments/benchmark.json under every policy arm.
#
#   ./experiments/run_experiments.sh [OUT_DIR] [--no-video]
#   REPEATS=5 ./experiments/run_experiments.sh results/repeats --no-video
#   RESUME=1 REPEATS=20 ./experiments/run_experiments.sh <same args>
#
# RESUME=1 skips any run whose metrics JSON already exists, so a sweep
# interrupted by a CARLA server crash can be restarted without redoing work.
#
# Assumes a CARLA 0.9.16 server is already listening on $PORT. Writes, per run:
#   OUT_DIR/<scenario>__<policy>[__rN].json   metrics summary
#   OUT_DIR/<scenario>__<policy>[__rN].mp4    chase-cam recording (unless --no-video)
#   OUT_DIR/<scenario>__<policy>[__rN].log    stderr trace
#
# REPEATS>1 exists because these outcomes are not fully repeatable: CARLA's
# physics substepping makes a marginal conflict flip between runs of an
# identical file, so a single sample per cell is not a measurement.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT}/experiments/results}"
VIDEO=1
[ "${2:-}" = "--no-video" ] && VIDEO=0
PORT="${PORT:-2000}"
HOST="${HOST:-127.0.0.1}"

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

module --force purge >/dev/null 2>&1
module load StdEnv/2020 python/3.8.10 >/dev/null 2>&1
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

# Guard the interpreter. `bash experiments/run_experiments.sh` ignores the
# `-l` in the shebang, so the module load above silently does nothing and the
# system Python 3.11 is used instead -- which has no CARLA wheel (cp38 only)
# and an ANTLR runtime that disagrees with the generated parser. Fail here
# with a usable message rather than eight identical tracebacks later.
if ! python -c "import carla" >/dev/null 2>&1; then
    echo "ERROR: this interpreter ($(python -V 2>&1), $(command -v python)) cannot import carla." >&2
    echo "       Run it as a login shell so the module load takes effect:" >&2
    echo "           ./experiments/run_experiments.sh        # uses the shebang" >&2
    echo "       or  bash -l experiments/run_experiments.sh" >&2
    exit 2
fi

# scenario:duration
# Ordered by map, not by family: the map comes from each .osc and the server
# reloads whenever it changes, which costs ~20 s a time. Three junction
# scenarios are on Town10HD_Opt, left_turn is on Town05 (see the header of
# scenarios/benchmark/left_turn.osc for why it cannot be staged on
# Town10HD_Opt), and the three highway scenarios are on Town04 -- so this order
# pays for two reloads instead of four. The report's ordering comes from
# benchmark.json and is unaffected.
SCENARIOS=(
    "red_light:18"
    "right_turn:24"
    "stop_sign:26"
    "left_turn:20"
    "lane_change:20"
    "cut_in:20"
    "overtake:16"
)
# policy_id:extra CLI args
POLICIES=(
    "scripted:"
    "idm:--ego-policy idm"
)

REPEATS="${REPEATS:-1}"
fail=0
for entry in "${SCENARIOS[@]}"; do
    name="${entry%%:*}"
    dur="${entry##*:}"
    for pol in "${POLICIES[@]}"; do
        pid="${pol%%:*}"
        pargs="${pol#*:}"
        for rep in $(seq 1 "${REPEATS}"); do
            if [ "${REPEATS}" -gt 1 ]; then
                tag="${name}__${pid}__r${rep}"
            else
                tag="${name}__${pid}"
            fi
            if [ "${RESUME:-0}" -eq 1 ] && [ -s "${OUT_DIR}/${tag}.json" ]; then
                echo "=== ${tag} — already present, skipping ==="
                continue
            fi
            echo "=== ${tag} (sim_duration=${dur}s) ==="
            # shellcheck disable=SC2086
            set -- python -m osc2carla "scenarios/benchmark/${name}.osc" \
                --host "${HOST}" --port "${PORT}" \
                --carla-timeout 300 --timeout 400 \
                --sim-duration "${dur}" \
                --metrics-out "${OUT_DIR}/${tag}.json" \
                ${pargs}
            if [ "${VIDEO}" -eq 1 ]; then
                set -- "$@" --record-video "${OUT_DIR}/${tag}.mp4" \
                    --record-actor ego --record-fps 20 \
                    --record-width 1280 --record-height 720
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
