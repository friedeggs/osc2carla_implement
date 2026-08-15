#!/bin/bash
# Parse + print behaviour trees for the paper-relevant scenarios.
# No CARLA server, no GPU.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${ROOT}/../install/env.sh" ]; then
    # shellcheck disable=SC1091
    source "${ROOT}/../install/env.sh"
fi
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

if [ -n "${AV_VENV:-}" ] && [ -x "${AV_VENV}/bin/python" ]; then
    PYTHON="${AV_VENV}/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

SCENARIOS=(
    scenarios/closed_loop_demo.osc
    scenarios/scenario_collision.osc
    scenarios/simple_example.osc
    scenarios/hello_world.osc
)

fail=0
for rel in "${SCENARIOS[@]}"; do
    echo "===== ${rel} ====="
    "${PYTHON}" -m osc2carla "${ROOT}/${rel}" --dry-run
    rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "[fail] ${rel} rc=${rc}" >&2
        fail=1
    fi
    echo
done
exit "${fail}"
