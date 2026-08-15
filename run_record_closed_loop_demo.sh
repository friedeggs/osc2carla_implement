#!/bin/bash -l
# Boot CARLA, then compile & record the two paper demos:
#   1. closed_loop_demo.osc   (lead-vehicle brake check)
#   2. scenario_collision.osc (relative spawn + ram pursuit)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../install/env.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

LOG="${AV_INSTALL}/carla_server_closed_loop_demo.log"
PORT="${AV_PORT}"
mkdir -p "${AV_RUN}"

echo "[host] $(hostname)"
echo "[gpu]  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"

module --force purge >/dev/null 2>&1
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
source "${AV_VENV}/bin/activate" 2>/dev/null || true

pkill -u "$USER" -f CarlaUE4-Linux-Shipping 2>/dev/null || true
sleep 3

cd "${AV_RUN}"
nohup "${AV_SERVER}/CarlaUE4.sh" -RenderOffScreen -carla-rpc-port="${PORT}" -nosound -quality-level=Low \
    >"${LOG}" 2>&1 &
SERVER_PID=$!

LISTENING=0
for i in $(seq 1 90); do
    ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$" && LISTENING=1 && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -80 "${LOG}"; exit 2; }
    sleep 2
done
[ "${LISTENING}" -eq 1 ] || { tail -120 "${LOG}"; kill -9 "${SERVER_PID}" 2>/dev/null; exit 3; }

"${AV_VENV}/bin/python" - <<PY || { kill -9 "${SERVER_PID}" 2>/dev/null; exit 4; }
import carla, time, sys
c = carla.Client('127.0.0.1', ${PORT}); c.set_timeout(180.0)
for attempt in range(30):
    try:
        w = c.get_world()
        print('[ready]', w.get_map().name)
        sys.exit(0)
    except RuntimeError as e:
        print(f'[wait {attempt+1}]', e)
        time.sleep(5)
sys.exit(1)
PY

sleep 5

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

run_one() {
    local name="$1" record_actor="$2" sim_duration="$3"
    local out_dir="${AV_RUN}/${name}"
    local scenario="${OSC2CARLA_ROOT}/scenarios/${name}.osc"
    local video="${out_dir}/${name}.mp4"
    mkdir -p "${out_dir}"

    echo "[step] dry-run parse ${scenario}"
    "${AV_VENV}/bin/python" -m osc2carla "${scenario}" --dry-run \
        2>&1 | tee "${out_dir}/dry_run.log"
    local dry_rc=${PIPESTATUS[0]}
    [ "${dry_rc}" -eq 0 ] || return "${dry_rc}"

    echo "[step] recording ${name}.osc (~${sim_duration} s, ${record_actor} camera)"
    "${AV_VENV}/bin/python" -m osc2carla "${scenario}" \
        --host 127.0.0.1 --port "${PORT}" \
        --carla-timeout 300 \
        --sim-duration "${sim_duration}" \
        --timeout 600 \
        --record-video "${video}" \
        --record-actor "${record_actor}" \
        --record-fps 20 \
        --record-width 1280 \
        --record-height 720 \
        2>&1 | tee "${out_dir}/record.log"
    local rc=${PIPESTATUS[0]}
    echo "[step] ${name} rc=${rc}"
    [ -f "${video}" ] && echo "[done] video: ${video}"
    return "${rc}"
}

run_one closed_loop_demo hero 22 || exit $?
run_one scenario_collision ego 22 || exit $?

exit 0
