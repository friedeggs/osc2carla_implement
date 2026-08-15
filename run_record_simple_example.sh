#!/bin/bash -l
# Boot CARLA, run simple_example.osc through osc2carla, record MP4.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../install/env.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

LOG="${AV_INSTALL}/carla_server_record_simple.log"
PORT="${AV_PORT}"
OUT_DIR="${AV_RUN}/simple_example"
SCENARIO="${OSC2CARLA_ROOT}/scenarios/simple_example.osc"
VIDEO="${OUT_DIR}/simple_example.mp4"

mkdir -p "${AV_RUN}" "${OUT_DIR}"

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

echo "[step] dry-run parse ${SCENARIO}"
"${AV_VENV}/bin/python" -m osc2carla "${SCENARIO}" --dry-run \
    2>&1 | tee "${OUT_DIR}/dry_run.log"
DRY_RC=${PIPESTATUS[0]}
[ "${DRY_RC}" -eq 0 ] || exit "${DRY_RC}"

echo "[step] recording simple_example.osc (~15 s, hero camera)"
"${AV_VENV}/bin/python" -m osc2carla "${SCENARIO}" \
    --host 127.0.0.1 --port "${PORT}" \
    --carla-timeout 300 \
    --sim-duration 15 \
    --timeout 300 \
    --record-video "${VIDEO}" \
    --record-actor hero \
    --record-fps 20 \
    --record-width 1280 \
    --record-height 720 \
    2>&1 | tee "${OUT_DIR}/record.log"
RC=${PIPESTATUS[0]}
echo "[step] recorder rc=${RC}"

if [ -f "${VIDEO}" ]; then
    echo "[done] video: ${VIDEO}"
fi

exit "${RC}"
