# Source after the parent CARLA env (optional, for live runs):
#   source ../install/env.sh
#   source env.sh
export OSC2CARLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OSC2CARLA_THIRD_PARTY="${OSC2CARLA_ROOT}/third_party"

_osc2carla_prepend_pythonpath() {
    local dir="$1"
    case ":${PYTHONPATH:-}:" in
        *":${dir}:"*) ;;
        *) export PYTHONPATH="${dir}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
}

_osc2carla_prepend_pythonpath "${OSC2CARLA_ROOT}"
if [ -n "${AV_SERVER:-}" ] && [ -d "${AV_SERVER}/PythonAPI/carla" ]; then
    _osc2carla_prepend_pythonpath "${AV_SERVER}/PythonAPI/carla"
fi
unset -f _osc2carla_prepend_pythonpath
