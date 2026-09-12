#!/bin/bash -l
# Boot CARLA and record the six harness scenario families as MP4s.
#
# Restarts the CARLA server between scenarios (map loads / RPC timeouts are
# common when chaining Town10 → Town05 → Town04 on one process).
#
#   ./run_record_benchmark6.sh
#   ONLY=cut_in,lane_change,overtake ./run_record_benchmark6.sh   # resume
#   sbatch sbatch_benchmark6_videos.sh
set -u
OSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AV_ROOT="${AV_ROOT:-}"
if [ -z "${AV_ROOT}" ]; then
  d="${OSC_ROOT}"
  while [ "${d}" != "/" ]; do
    if [ -f "${d}/install/env.sh" ]; then
      AV_ROOT="${d}"
      break
    fi
    d="$(dirname "${d}")"
  done
fi
if [ -z "${AV_ROOT}" ] || [ ! -f "${AV_ROOT}/install/env.sh" ]; then
  echo "AV_ROOT not found (set AV_ROOT to the tree with install/env.sh)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${AV_ROOT}/install/env.sh"
# shellcheck disable=SC1091
source "${OSC_ROOT}/env.sh"

OUT_ROOT="${OUT_ROOT:-${AV_RUN}/osc2runner_benchmark6}"
PORT="${AV_PORT_OVERRIDE:-$(( 2000 + (${SLURM_JOB_ID:-0} % 200) * 4 ))}"
LOG="${AV_INSTALL}/carla_server_osc2_benchmark6_${SLURM_JOB_ID:-local}.log"
PY="${AV_VENV}/bin/python"
export PATH="/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/bin:${PATH}"
CARLA_TIMEOUT_MS="${CARLA_TIMEOUT_MS:-600000}"

# name:sim_duration:record_actor:town
ALL_SCENARIOS=(
  "red_light:18:ego:Town10HD_Opt"
  "right_turn:24:ego:Town10HD_Opt"
  "left_turn:20:ego:Town05"
  "cut_in:20:ego:Town04"
  "lane_change:20:ego:Town04"
  "overtake:16:ego:Town04"
)

ONLY="${ONLY:-}"
ONLY="${ONLY//:/,}"
SCENARIOS=()
for entry in "${ALL_SCENARIOS[@]}"; do
  name="${entry%%:*}"
  if [ -z "${ONLY}" ] || [[ ",${ONLY}," == *",${name},"* ]]; then
    SCENARIOS+=("${entry}")
  fi
done

echo "[host] $(hostname)"
echo "[gpu]  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true)"
echo "[port] ${PORT}"
echo "[out]  ${OUT_ROOT}"
echo "[set]  ${SCENARIOS[*]}"
mkdir -p "${OUT_ROOT}" "${AV_RUN}"

module --force purge >/dev/null 2>&1 || true
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5 >/dev/null 2>&1
# shellcheck disable=SC1091
source "${AV_VENV}/bin/activate"

SERVER_PID=""
stop_carla() {
  if [ -n "${SERVER_PID}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -9 "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
  pkill -u "$USER" -f CarlaUE4-Linux-Shipping 2>/dev/null || true
  sleep 2
}

start_carla() {
  stop_carla
  cd "${AV_RUN}"
  nohup "${AV_SERVER}/CarlaUE4.sh" -RenderOffScreen -carla-rpc-port="${PORT}" -nosound -quality-level=Low \
    >"${LOG}" 2>&1 &
  SERVER_PID=$!
  local listening=0
  local i
  for i in $(seq 1 120); do
    ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$" && listening=1 && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { echo "[fail] CARLA died during boot"; tail -80 "${LOG}"; return 2; }
    sleep 2
  done
  [ "${listening}" -eq 1 ] || { echo "[fail] CARLA not listening"; tail -120 "${LOG}"; stop_carla; return 3; }
  "${PY}" - <<PY
import carla, time, sys
c = carla.Client("127.0.0.1", int("${PORT}"))
c.set_timeout(180.0)
for attempt in range(40):
    try:
        w = c.get_world()
        print("[ready]", w.get_map().name)
        sys.exit(0)
    except RuntimeError as e:
        print(f"[wait {attempt+1}]", e)
        time.sleep(5)
sys.exit(1)
PY
}

trap stop_carla EXIT

FAIL=0
for entry in "${SCENARIOS[@]}"; do
  IFS=':' read -r name dur actor town <<< "${entry}"
  scenario="${OSC_ROOT}/scenarios/benchmark/${name}.osc"
  out_dir="${OUT_ROOT}/${name}"
  video="${out_dir}/${name}.mp4"
  mkdir -p "${out_dir}"
  echo
  echo "===== ${name} (town=${town}, ${dur}s, camera=${actor}) ====="
  if [ ! -f "${scenario}" ]; then
    echo "[skip] missing ${scenario}"
    FAIL=1
    continue
  fi
  if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "${video}" ] && [ -s "${video}" ]; then
    echo "[skip] existing video $(ls -lh "${video}" | awk '{print $5}')"
    continue
  fi

  start_carla || { FAIL=1; continue; }

  "${PY}" -m osc2carla "${scenario}" --dry-run \
    >"${out_dir}/dry_run.log" 2>&1 || {
      echo "[fail] dry-run ${name}"; tail -20 "${out_dir}/dry_run.log"; FAIL=1; continue
    }

  set +e
  "${PY}" -m osc2carla "${scenario}" \
    --host 127.0.0.1 --port "${PORT}" \
    --carla-timeout 600 \
    --sim-duration "${dur}" \
    --timeout 1200 \
    --record-video "${video}" \
    --record-actor "${actor}" \
    --record-fps 20 \
    --record-width 1280 \
    --record-height 720 \
    --metrics-out "${out_dir}/metrics.json" \
    >"${out_dir}/record.log" 2>&1
  rc=$?
  set -e

  # A post-run RPC timeout during cleanup still often leaves a good video.
  if [ -f "${video}" ] && [ -s "${video}" ]; then
    ls -lh "${video}" | awk '{print "[done]", $NF, $5}'
    grep -E 'metrics|collision|wrote' "${out_dir}/record.log" | tail -5 | sed 's/^/  /' || true
    if [ "${rc}" -ne 0 ]; then
      echo "  [note] osc2carla exited ${rc} after writing the video (cleanup timeout is OK)"
    fi
  else
    echo "[fail] record ${name} rc=${rc} (no video)"
    tail -40 "${out_dir}/record.log" | sed 's/^/  /'
    FAIL=1
  fi
  stop_carla
done

echo
echo "videos in ${OUT_ROOT}"
find "${OUT_ROOT}" -name '*.mp4' -printf '%p %s\n' | sort
# Require all requested scenarios to have a video
MISSING=0
for entry in "${SCENARIOS[@]}"; do
  name="${entry%%:*}"
  if [ ! -s "${OUT_ROOT}/${name}/${name}.mp4" ]; then
    echo "[missing] ${name}"
    MISSING=1
  fi
done
if [ "${MISSING}" -ne 0 ]; then
  exit 1
fi
exit 0
