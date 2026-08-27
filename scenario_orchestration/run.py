#!/usr/bin/env python3
"""Standardized execution entry point for the scenario_orchestration harness.

The harness (``../scenario_orchestration``, DESIGN.md section 5) never imports
anything from this repository. It writes two JSON documents, runs

    python scenario_orchestration/run.py \
        --scenario-request request.json \
        --policy-request policy.json \
        --output-dir <results>/raw/<experiment_id>

and reads ``method_result.json`` back out of the output directory. Everything
between those two points is this file's job:

    ScenarioRequest  ->  which .osc file, on which backend, for how long
    PolicyRequest    ->  --ego-policy and its parameters, in this repo's names
    metrics summary  ->  the canonical metric vocabulary

Nothing here reaches into the harness either: the two requests are the whole
input, so this file runs unchanged against any checkout of it.

What the harness gets back
--------------------------
``status`` is ``success`` when osc2carla ran the scenario to its horizon,
``failure`` when the run or the translation of the request failed, ``timeout``
when it outlived its budget, ``error`` when a result could not be produced at
all. ``incompatible`` is deliberately absent: compatibility is the harness's
call, not ours (third_party/README.md).

Metrics
-------
``scenario_realized`` is this benchmark's intent proxy: the run produced the
outcome the scenario was written to produce, with the antagonist it was written
to produce it with (``collision_occurred == expect_collision`` and, for the six
conflict families, ``intended_partner`` among the contacted roles). The intent
lives in ``experiments/benchmark.json`` / ``benchmark_local.json``, beside the
scenarios, exactly as it does for this repository's own experiment runner.

``scenario_success`` follows the harness's own success criteria
(``scenario_realized`` and ``no_collision``), which for the six adversarial
families means the conflict developed *and* the ego was not the one hit. Read
it knowing that these families measure scenario execution through the
collision, so a realized conflict and an unharmed ego rarely coincide;
``stop_sign`` is the one family where success is the ordinary reading.

Environment overrides
---------------------
Operator-side escape hatches, all optional; each wins over the corresponding
request parameter, because it describes the machine rather than the experiment:

    OSC2CARLA_BACKEND            carla (default) | pygame | localsim
    OSC2CARLA_PYTHON             interpreter for the run (CARLA needs cp38)
    OSC2CARLA_CARLA_HOST/PORT    CARLA RPC endpoint (127.0.0.1 / 2000)
    OSC2CARLA_TOWN               local road network (local backend only)
    OSC2CARLA_FIXED_DT           simulation step; default 1 / tick_rate_hz
    OSC2CARLA_SIM_DURATION       simulated seconds; default per scenario
    OSC2CARLA_RUN_TIMEOUT_S      wall-clock budget for the child process
    OSC2CARLA_RECORD_VIDEO       1 to write <family>.mp4 into the output dir
    OSC2CARLA_NEAR_COLLISION_M   near-miss gap threshold, metres (default 2.0)
    OSC2CARLA_HARNESS_ROOT       where third_party/<policy repo> lives
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHOD_RESULT_FILE = "method_result.json"
METRICS_FILE = "osc2carla_metrics.json"
#: What the policy bridge writes about the observation it actually served.
POLICY_NOTES_FILE = "policy_bridge.json"
LOG_FILE = "osc2carla.log"

#: Backend -> where its scenarios and their declared intent live.
BACKENDS = {
    "carla": {
        "scenario_dir": os.path.join("scenarios", "benchmark"),
        "benchmark": os.path.join("experiments", "benchmark.json"),
    },
    "pygame": {
        "scenario_dir": os.path.join("scenarios", "local", "benchmark"),
        "benchmark": os.path.join("experiments", "benchmark_local.json"),
    },
}
BACKEND_ALIASES = {"localsim": "pygame", "local": "pygame", "pygame": "pygame",
                   "carla": "carla"}

#: Road networks the local backend can synthesise. A requested town outside
#: this set is a CARLA map name, which the local backend would alias onto some
#: other network -- see resolve_town.
LOCAL_TOWNS = ("grid", "highway", "loop", "two_lane", "wide_grid")

#: Native ids other declarations use for the same .osc file.
NATIVE_ALIASES = {"red_light_violation": "red_light"}

#: PolicyRequest parameter names (SI, method-agnostic) -> this repo's names.
IDM_PARAMETERS = {
    "desired_speed_mps": "v0",
    "time_headway_s": "T",
    "min_gap_m": "s0",
    "max_accel_mps2": "a_max",
    "comfort_decel_mps2": "b",
}
CONSTANT_PARAMETERS = {"desired_speed_mps": "v0"}

#: Policies this repository realizes natively, by the class its implementation
#: names. Analytic policies are fully described by their parameters, so they
#: need no policy repository (harness configs/policy/*.yaml say as much).
NATIVE_POLICIES = {
    "idmpolicy": ("idm", IDM_PARAMETERS),
    "constantspeedpolicy": ("constant", CONSTANT_PARAMETERS),
}
NATIVE_POLICY_NAMES = {
    "idm": ("idm", IDM_PARAMETERS),
    "idm_assertive": ("idm", IDM_PARAMETERS),
    "idm_conservative": ("idm", IDM_PARAMETERS),
    "constant": ("constant", CONSTANT_PARAMETERS),
}
#: Names that mean "leave the ego to the compiled behaviour tree".
SCRIPTED_POLICY_NAMES = ("scripted", "none", "compiled", "behaviour_tree")

#: Loaded by module name, from the directory this file lives in; see
#: child_environment, which puts that directory on the child's PYTHONPATH.
BRIDGE_POLICY = "osc2carla_policy_bridge:BridgedPolicy"
BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FIXED_DT = 0.05
DEFAULT_NEAR_COLLISION_M = 2.0


class RequestError(Exception):
    """The request cannot be turned into an osc2carla invocation."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object, got %s" % type(payload).__name__)
    return payload


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_float(name: str) -> Optional[float]:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_flag(name: str) -> bool:
    raw = (_env(name) or "").lower()
    return raw in ("1", "true", "yes", "on")


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: A traceback's last line: the exception that actually stopped the run.
_EXCEPTION_LINE = re.compile(r"^\w+(?:Error|Exception|Exit)\b.*: .+")


def _first_line_of_error(stderr: Optional[str]) -> str:
    """The most quotable line of a failed run's stderr.

    Bottom-up, because the cause is at the end: the raised exception first,
    then osc2carla's own diagnosis, then whatever was printed last. The ANTLR
    version banner is printed once per parse and is never the reason.
    """
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()
             and "ANTLR runtime and generated code versions disagree" not in ln]
    for line in reversed(lines):
        if _EXCEPTION_LINE.match(line):
            return line
    for line in reversed(lines):
        if "unavailable" in line or "ERROR" in line or "error:" in line:
            return line
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# scenario resolution
# ---------------------------------------------------------------------------

def resolve_backend(parameters: Dict[str, Any]) -> str:
    """carla unless the operator or the request asks for the local simulator."""
    requested = _env("OSC2CARLA_BACKEND") or parameters.get("backend") or "carla"
    key = str(requested).strip().lower()
    if key not in BACKEND_ALIASES:
        raise RequestError(
            "unknown backend %r; osc2carla_implement runs on %s"
            % (requested, " or ".join(sorted(set(BACKEND_ALIASES.values()))))
        )
    return BACKEND_ALIASES[key]


def scenario_candidates(request: Dict[str, Any]) -> List[str]:
    """Names to try, in order, when locating this family's .osc file."""
    implementation = request.get("implementation") or {}
    parameters = implementation.get("parameters") or {}
    out: List[str] = []
    for value in (parameters.get("scenario"),
                  implementation.get("native_id"),
                  request.get("scenario_family"),
                  request.get("semantic_id")):
        if not value:
            continue
        stem = os.path.basename(str(value))
        if stem.endswith(".osc"):
            stem = stem[:-4]
        for candidate in (stem, NATIVE_ALIASES.get(stem)):
            if candidate and candidate not in out:
                out.append(candidate)
    return out


def resolve_scenario(request: Dict[str, Any], backend: str) -> Tuple[str, str]:
    """``(family name, absolute .osc path)`` for this request on this backend."""
    parameters = (request.get("implementation") or {}).get("parameters") or {}
    explicit = parameters.get("scenario_file")
    if explicit:
        path = str(explicit) if os.path.isabs(str(explicit)) \
            else os.path.join(REPO_ROOT, str(explicit))
        if not os.path.exists(path):
            raise RequestError("scenario_file %r does not exist" % explicit)
        return os.path.splitext(os.path.basename(path))[0], path

    directory = os.path.join(REPO_ROOT, BACKENDS[backend]["scenario_dir"])
    tried = scenario_candidates(request)
    for name in tried:
        path = os.path.join(directory, name + ".osc")
        if os.path.exists(path):
            return name, path
    available = sorted(f[:-4] for f in os.listdir(directory) if f.endswith(".osc")) \
        if os.path.isdir(directory) else []
    raise RequestError(
        "no %s scenario for family %r (tried %s in %s; available: %s)"
        % (backend, request.get("scenario_family"), ", ".join(tried) or "nothing",
           BACKENDS[backend]["scenario_dir"], ", ".join(available) or "none")
    )


def load_intent(backend: str, family: str) -> Dict[str, Any]:
    """The scenario's declared intent, from this repository's benchmark config.

    Empty when the scenario is not part of the benchmark: the run still happens,
    but ``scenario_realized`` cannot be judged and is left unreported rather
    than guessed.
    """
    path = os.path.join(REPO_ROOT, BACKENDS[backend]["benchmark"])
    try:
        config = _read_json(path)
    except (OSError, ValueError):
        return {}
    for entry in config.get("scenarios") or []:
        if entry.get("name") == family:
            return dict(entry)
    return {}


def scenario_ego_binding(path: str) -> Optional[str]:
    """The binding this run measures and, under a policy, drives.

    Every benchmark scenario names it ``ego``. A scenario reached through
    ``parameters.scenario_file`` need not, and osc2carla's own default -- the
    first vehicle binding in the file -- is the better answer than a forced
    name that does not exist.
    """
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return None
    return "ego" if re.search(r"^\s*ego\s*:\s*vehicle", text, re.M) else None


def scenario_map_file(path: str) -> Optional[str]:
    """The map an .osc pins itself to, via ``keep(it.map_file == "...")``."""
    try:
        with open(path) as fh:
            match = re.search(r'map_file\s*==\s*"([^"]+)"', fh.read())
    except OSError:
        return None
    return match.group(1) if match else None


def resolve_town(backend: str, intent: Dict[str, Any], parameters: Dict[str, Any],
                 osc_map: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``(--town value, note)``.

    Each .osc hard-codes the geometry of one map, so a town is a property of the
    scenario and not of the request. A request town is honoured only where it
    can be: on the local backend, and only when it names a bundled road network.
    """
    requested = _env("OSC2CARLA_TOWN") or parameters.get("town")
    if backend == "carla":
        note = None
        if requested and osc_map and str(requested) != osc_map:
            note = ("requested town %r ignored: this scenario stages its "
                    "geometry on %s" % (requested, osc_map))
        return None, note
    default = intent.get("town") or osc_map
    if requested is None:
        return default, None
    if str(requested).lower() in LOCAL_TOWNS:
        return str(requested).lower(), None
    return default, ("requested town %r is not a local road network; using %r, "
                     "which is what this scenario's coordinates were written "
                     "against" % (requested, default))


def resolve_duration(evaluation: Dict[str, Any], intent: Dict[str, Any],
                     parameters: Dict[str, Any]) -> Tuple[float, str]:
    """Simulated seconds to run, and where the number came from.

    The evaluation protocol's horizon bounds the episode; the scenario's own
    tuned duration bounds it too, since the choreography ends there. The shorter
    of the two is the one that respects both.
    """
    override = _env_float("OSC2CARLA_SIM_DURATION") or _as_float(parameters.get("sim_duration"))
    if override and override > 0:
        return override, "override"
    horizon = _as_float(evaluation.get("horizon_s")) or 0.0
    tuned = _as_float(intent.get("sim_duration")) or 0.0
    if horizon > 0 and tuned > 0:
        return (min(horizon, tuned),
                "min(evaluation.horizon_s, benchmark sim_duration)")
    if horizon > 0:
        return horizon, "evaluation.horizon_s"
    if tuned > 0:
        return tuned, "benchmark sim_duration"
    return 0.0, "scenario (largest wait elapsed)"


def resolve_fixed_dt(evaluation: Dict[str, Any], parameters: Dict[str, Any]) -> float:
    """Step size, from the evaluation protocol's tick rate where it gives one."""
    override = _env_float("OSC2CARLA_FIXED_DT") or _as_float(parameters.get("fixed_dt"))
    if override and override > 0:
        return override
    tick_rate = _as_float(evaluation.get("tick_rate_hz")) or 0.0
    if tick_rate > 0:
        return 1.0 / tick_rate
    return DEFAULT_FIXED_DT


# ---------------------------------------------------------------------------
# policy translation (DESIGN.md section 6)
# ---------------------------------------------------------------------------

def harness_root(request_path: str) -> Optional[str]:
    """Where the harness that issued this request lives.

    Needed only to resolve a policy repository's relative ``repository`` and
    ``entry_point`` paths; nothing is imported from the harness itself.
    """
    override = _env("OSC2CARLA_HARNESS_ROOT")
    if override:
        return override if os.path.isdir(override) else None
    path = os.path.dirname(os.path.abspath(request_path))
    while True:
        if os.path.isdir(os.path.join(path, "configs", "algorithm")) \
                or os.path.isdir(os.path.join(path, "third_party")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def translate_parameters(parameters: Dict[str, Any], mapping: Dict[str, str],
                         native: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
    """PolicyRequest parameters in this repository's parameter names.

    ``--policy-param`` takes numbers only, so anything else is reported as
    ignored rather than dropped silently.
    """
    translated: Dict[str, float] = {}
    ignored: List[str] = []
    # Native names first, then the standardized ones, so that a request
    # carrying both spellings of one quantity resolves to the standardized
    # value rather than to whichever came last in the document.
    for standardized in (False, True):
        for key, value in (parameters or {}).items():
            if (key in mapping) is not standardized:
                continue
            target = mapping.get(key, key if key in native else None)
            number = _as_float(value)
            if target is None or number is None:
                ignored.append(key)
                continue
            translated[target] = number
    return translated, sorted(ignored)


#: Mirrors ``DEFAULTS`` in osc2carla/backend/policy.py, for the case where this
#: process cannot import it -- run.py may run under the harness's interpreter
#: while the run itself uses another one (OSC2CARLA_PYTHON).
NATIVE_PARAMETERS = {
    "idm": ("v0", "T", "a_max", "b", "delta", "s0", "lookahead", "a_throttle",
            "a_brake"),
    "constant": ("v0", "kp", "lookahead"),
}


def native_parameter_names(policy: str) -> Sequence[str]:
    """The parameter names the built-in policy accepts, from the source itself."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        from osc2carla.backend.policy import BUILTIN_POLICIES
        return tuple(getattr(BUILTIN_POLICIES.get(policy), "DEFAULTS", {}) or ())
    except Exception:  # noqa: BLE001 - not importable here, use the mirror
        return NATIVE_PARAMETERS.get(policy, ())


def build_policy_plan(policy_request: Dict[str, Any], request_path: str,
                      output_dir: str
                      ) -> Tuple[List[str], Dict[str, str], Dict[str, Any]]:
    """``(cli args, extra env, notes)`` for the requested ego policy.

    Three outcomes, in the order they are tried:

    * a policy this repository realizes natively (the analytic IDM family and
      the constant-speed reference), reached through ``--ego-policy`` with the
      request's parameters translated into this repository's names;
    * the compiled behaviour tree, when the request asks for no ego policy;
    * any other ``ego_policy_v1`` policy, loaded from its own repository's
      ``scenario_orchestration/policy.py`` through the bridge.
    """
    name = str(policy_request.get("name") or "").strip().lower()
    implementation = str(policy_request.get("implementation") or "").strip()
    parameters = dict(policy_request.get("parameters") or {})
    interface = str(policy_request.get("interface") or "ego_policy_v1")
    observation_space = str(policy_request.get("observation_space") or "state")
    action_space = str(policy_request.get("action_space") or "control")

    if interface != "ego_policy_v1":
        raise RequestError(
            "policy %r declares interface %r; osc2carla_implement speaks "
            "'ego_policy_v1'" % (name or implementation, interface)
        )
    if observation_space != "state":
        raise RequestError(
            "policy %r wants a %r observation space; this runtime describes the "
            "scene as state -- ego pose and speed, object-centric actors, route, "
            "signals and a BEV raster on the CARLA backend -- and renders no "
            "sensor stream" % (name or implementation, observation_space)
        )
    if action_space not in ("control", "waypoints"):
        raise RequestError(
            "policy %r emits %r; this runtime applies normalised control "
            "(throttle, brake, steer), and accepts a waypoint policy through the "
            "control its own controllers return alongside its waypoints. There is "
            "no %s follower here"
            % (name or implementation, action_space, action_space)
        )

    if name in SCRIPTED_POLICY_NAMES:
        return [], {}, {"policy_mode": "scripted",
                        "policy_note": "the compiled behaviour tree drives the ego"}

    class_name = implementation.rsplit(".", 1)[-1].rsplit(":", 1)[-1].lower()
    native = NATIVE_POLICIES.get(class_name) or NATIVE_POLICY_NAMES.get(name)
    if native is not None:
        policy, mapping = native
        translated, ignored = translate_parameters(
            parameters, mapping, native_parameter_names(policy))
        args = ["--ego-policy", policy]
        for key in sorted(translated):
            args += ["--policy-param", "%s=%r" % (key, translated[key])]
        notes = {
            "policy_mode": "native",
            "policy_resolved": policy,
            "policy_parameters_translated": translated,
        }
        if ignored:
            notes["policy_parameters_ignored"] = ignored
        return args, {}, notes

    entry_point = locate_policy_entry_point(policy_request, request_path)
    if entry_point is None:
        raise RequestError(
            "policy %r (%s) is not one this repository realizes natively (%s), "
            "and its repository's entry point %s/%s is not present next to the "
            "harness; install it, or point OSC2CARLA_HARNESS_ROOT at the "
            "checkout that has it"
            % (name or implementation, implementation or "no implementation",
               ", ".join(sorted(NATIVE_POLICY_NAMES)),
               policy_request.get("repository") or "?",
               policy_request.get("entry_point") or "scenario_orchestration/policy.py")
        )
    return (
        ["--ego-policy", BRIDGE_POLICY],
        {"OSC2CARLA_POLICY_REQUEST": os.path.abspath(request_path),
         "OSC2CARLA_POLICY_ENTRY_POINT": entry_point,
         "OSC2CARLA_POLICY_NOTES": os.path.join(os.path.abspath(output_dir),
                                                POLICY_NOTES_FILE)},
        {"policy_mode": "bridged",
         "policy_entry_point": entry_point,
         "policy_note": "loaded through "
                        "scenario_orchestration/osc2carla_policy_bridge.py"},
    )


def locate_policy_entry_point(policy_request: Dict[str, Any],
                              request_path: str) -> Optional[str]:
    """The external policy's ``scenario_orchestration/policy.py``, if it exists."""
    entry_point = policy_request.get("entry_point") or "scenario_orchestration/policy.py"
    repository = policy_request.get("repository")
    candidates: List[str] = []
    if os.path.isabs(str(entry_point)):
        candidates.append(str(entry_point))
    else:
        root = harness_root(request_path)
        for base in (os.path.join(root, str(repository)) if root and repository else None,
                     root, REPO_ROOT):
            if base:
                candidates.append(os.path.join(base, str(entry_point)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


# ---------------------------------------------------------------------------
# invocation
# ---------------------------------------------------------------------------

def build_command(scenario_path: str, backend: str, town: Optional[str],
                  intent: Dict[str, Any], duration: float, fixed_dt: float,
                  policy_args: Sequence[str], output_dir: str, family: str,
                  ego_binding: Optional[str]) -> List[str]:
    interpreter = _env("OSC2CARLA_PYTHON") or sys.executable
    command = [interpreter, "-m", "osc2carla", scenario_path,
               "--backend", "pygame" if backend == "pygame" else "carla",
               "--fixed-dt", "%r" % fixed_dt,
               "--metrics-out", os.path.join(output_dir, METRICS_FILE)]
    if ego_binding:
        # Which binding the metrics measure, and which one an ego policy takes
        # over; osc2carla falls back to the first vehicle in the file.
        command += ["--record-actor", ego_binding]
    if duration > 0:
        command += ["--sim-duration", "%r" % duration]
    if backend == "carla":
        command += ["--host", _env("OSC2CARLA_CARLA_HOST") or "127.0.0.1",
                    "--port", _env("OSC2CARLA_CARLA_PORT") or "2000",
                    "--carla-timeout", "300"]
    else:
        # A whole-map preference: which exit drive() takes at a junction. The
        # highway ports never cross one, hence the harmless default.
        command += ["--junction-turn", str(intent.get("junction_turn") or "straight")]
        if town:
            command += ["--town", town]
    if _env_flag("OSC2CARLA_RECORD_VIDEO"):
        command += ["--record-video", os.path.join(output_dir, family + ".mp4"),
                    # One frame is captured per simulation tick, so the
                    # container's frame rate has to be the tick rate or the video
                    # misrepresents time: at fixed_dt=0.1 the sim runs at 10 Hz,
                    # and stamping 20 fps made an 18.1 s episode play in 9 s. For
                    # a recording whose point is to show whether the ego braked in
                    # time, playing at 2x is not a cosmetic problem.
                    "--record-fps", "%g" % max(1.0, round(1.0 / fixed_dt, 3)),
                    # Per *pane*. The recorder composites a top-down and a chase
                    # view side by side, so the file is twice this wide -- 1440x540,
                    # which is the geometry the orchestration method's recorder
                    # produces. Matching it means the two methods' videos can be
                    # put next to each other without rescaling one of them.
                    "--record-width", "720",
                    "--record-height", "540"]
        if backend == "pygame":
            command += ["--render-mode", "headless"]
    elif backend == "pygame":
        command += ["--render-mode", "off"]
    return command + list(policy_args)


def child_environment(extra: Dict[str, str], seed: int) -> Dict[str, str]:
    env = dict(os.environ)
    python_path = [REPO_ROOT, BRIDGE_DIR] + [
        p for p in (env.get("PYTHONPATH") or "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["OSC2CARLA_ROOT"] = REPO_ROOT
    env["PYTHONUNBUFFERED"] = "1"
    # Not a simulation seed -- neither backend takes one -- but it does pin the
    # one source of run-to-run variation Python itself contributes.
    env.setdefault("PYTHONHASHSEED", str(seed))
    env.update(extra)
    return env


def run_osc2carla(command: Sequence[str], env: Dict[str, str], timeout_s: float,
                  output_dir: str) -> Tuple[Optional[int], str, str, bool]:
    started = time.time()
    timed_out = False
    try:
        completed = subprocess.run(list(command), cwd=REPO_ROOT, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True, timeout=timeout_s)
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        returncode = None
    # The harness captures whatever this process prints; pass the child's
    # streams through so stdout.log and stderr.log read as if it ran directly,
    # and keep a copy beside the results.
    sys.stdout.write(stdout or "")
    sys.stderr.write(stderr or "")
    try:
        with open(os.path.join(output_dir, LOG_FILE), "w") as fh:
            fh.write(stdout or "")
            fh.write(stderr or "")
    except OSError:
        pass
    sys.stderr.write("[run.py] osc2carla finished in %.1fs (returncode=%s)\n"
                     % (time.time() - started, returncode))
    return returncode, stdout, stderr, timed_out


# ---------------------------------------------------------------------------
# metrics (DESIGN.md section 11)
# ---------------------------------------------------------------------------

def canonical_metrics(summary: Dict[str, Any], intent: Dict[str, Any],
                      near_gap_m: float) -> Dict[str, Any]:
    """The canonical vocabulary, derived from this run's summary and intent."""
    collision = bool(summary.get("collision_occurred"))
    roles = [r for r in (summary.get("collision_partner_roles") or []) if r]
    metrics: Dict[str, Any] = {"collision": collision}

    if "expect_collision" in intent:
        expected = bool(intent.get("expect_collision"))
        partner = intent.get("intended_partner")
        realized = collision == expected
        if realized and expected and partner:
            realized = partner in roles
        metrics["scenario_realized"] = realized
        # The harness's own success criteria: the family's target interaction
        # occurred and the ego was not hit.
        metrics["scenario_success"] = bool(realized and not collision)

    first = _as_float(summary.get("first_collision_time"))
    if first is not None:
        metrics["time_to_event"] = first

    # Near-miss is read off the closest approach to the leader on the ego's own
    # path, which the run summary records only when an ego policy is driving --
    # the compiled arm does no leader perception. Left unreported when there is
    # nothing to read, and blind by construction to a crossing conflict, which
    # is not a leader.
    gap = _as_float(summary.get("min_leader_gap_m"))
    if collision:
        metrics["near_collision"] = False
    elif gap is not None:
        metrics["near_collision"] = bool(gap <= near_gap_m)

    duration = _as_float(summary.get("sim_duration"))
    if duration is not None:
        metrics["scenario_duration"] = duration
    return metrics


def write_result(output_dir: str, status: str, metrics: Optional[Dict[str, Any]] = None,
                 method_metrics: Optional[Dict[str, Any]] = None,
                 reason: Optional[str] = None,
                 trace_path: Optional[str] = None) -> int:
    """Write ``method_result.json``; its ``status`` is the run's verdict."""
    report = {
        "status": status,
        "metrics": dict(metrics or {}),
        "method_metrics": dict(method_metrics or {}),
        "trace_path": trace_path,
        "reason": reason,
    }
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, METHOD_RESULT_FILE), "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    except OSError as exc:
        sys.stderr.write("[run.py] could not write %s: %s\n" % (METHOD_RESULT_FILE, exc))
        return 1
    sys.stderr.write("[run.py] %s -> status=%s%s\n"
                     % (METHOD_RESULT_FILE, status,
                        " (%s)" % reason if reason else ""))
    # The report carries the verdict; a non-zero exit would have the harness
    # overwrite it with 'failure' (third_party/README.md, Outputs).
    return 0


# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one scenario_orchestration experiment on osc2carla_implement")
    parser.add_argument("--scenario-request", required=True)
    parser.add_argument("--policy-request", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    output_dir = os.path.abspath(args.output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        sys.stderr.write("[run.py] cannot create %s: %s\n" % (output_dir, exc))
        return 1

    try:
        request = _read_json(args.scenario_request)
    except (OSError, ValueError) as exc:
        return write_result(output_dir, "error",
                            reason="unreadable scenario request: %s" % exc)
    try:
        policy_request = _read_json(args.policy_request)
    except (OSError, ValueError) as exc:
        return write_result(output_dir, "error",
                            reason="unreadable policy request: %s" % exc)

    experiment_id = str(request.get("experiment_id") or "")
    seed = int(request.get("seed") or 0)
    evaluation = dict(request.get("evaluation") or {})
    implementation = dict(request.get("implementation") or {})
    parameters = dict(implementation.get("parameters") or {})

    context: Dict[str, Any] = {
        "experiment_id": experiment_id,
        "seed": seed,
        "seed_effect": ("none: the local simulator is deterministic and CARLA's "
                        "variation comes from physics substepping, which no seed "
                        "here controls"),
        "evaluation_horizon_s": evaluation.get("horizon_s"),
        "evaluation_tick_rate_hz": evaluation.get("tick_rate_hz"),
        "requested_native_id": implementation.get("native_id"),
        "requested_parameters": parameters,
        "policy_requested": policy_request.get("name"),
    }

    try:
        backend = resolve_backend(parameters)
        family, scenario_path = resolve_scenario(request, backend)
        intent = load_intent(backend, family)
        osc_map = scenario_map_file(scenario_path)
        ego_binding = scenario_ego_binding(scenario_path)
        town, town_note = resolve_town(backend, intent, parameters, osc_map)
        duration, duration_source = resolve_duration(evaluation, intent, parameters)
        fixed_dt = resolve_fixed_dt(evaluation, parameters)
        policy_args, policy_env, policy_notes = build_policy_plan(
            policy_request, args.policy_request, output_dir)
    except RequestError as exc:
        return write_result(output_dir, "failure", method_metrics=context,
                            reason=str(exc))

    context.update(policy_notes)
    context.update({
        "backend": backend,
        "scenario_family_resolved": family,
        "scenario_file": os.path.relpath(scenario_path, REPO_ROOT),
        "scenario_map_file": osc_map,
        "ego_binding": ego_binding or "(first vehicle in the scenario)",
        "town": town or osc_map,
        "junction_turn": intent.get("junction_turn") if backend == "pygame" else None,
        "sim_duration_requested": duration,
        "sim_duration_source": duration_source,
        "fixed_dt": fixed_dt,
        "expect_collision": intent.get("expect_collision"),
        "intended_partner": intent.get("intended_partner"),
    })
    if town_note:
        context["town_note"] = town_note
    if not intent:
        context["intent_note"] = (
            "%s is not declared in %s, so scenario_realized and scenario_success "
            "cannot be judged and are not reported"
            % (family, BACKENDS[backend]["benchmark"]))

    command = build_command(scenario_path, backend, town, intent, duration,
                            fixed_dt, policy_args, output_dir, family,
                            ego_binding)
    timeout_s = _env_float("OSC2CARLA_RUN_TIMEOUT_S") \
        or (900.0 if backend == "carla" else max(300.0, 30.0 * max(duration, 1.0)))
    # The wall-clock bound osc2carla puts on its own tick loop, inside ours.
    command += ["--timeout", "%r" % max(60.0, timeout_s - 60.0)]
    context["command"] = list(command)
    context["timeout_s"] = timeout_s

    sys.stderr.write("[run.py] %s: %s on %s (%s, %.1fs @ %.0f Hz)\n"
                     % (experiment_id or family, family, backend,
                        context.get("policy_mode", "?"), duration,
                        1.0 / fixed_dt if fixed_dt else 0.0))

    started = time.time()
    returncode, _stdout, stderr, timed_out = run_osc2carla(
        command, child_environment(policy_env, seed), timeout_s, output_dir)
    context["wall_time_s"] = round(time.time() - started, 3)
    context["returncode"] = returncode

    metrics_path = os.path.join(output_dir, METRICS_FILE)
    summary: Dict[str, Any] = {}
    if os.path.exists(metrics_path):
        try:
            summary = _read_json(metrics_path)
        except (OSError, ValueError) as exc:
            context["metrics_read_error"] = str(exc)

    if timed_out:
        return write_result(
            output_dir, "timeout", metrics=dict(summary), method_metrics=context,
            reason="osc2carla exceeded its %.0fs budget on %s" % (timeout_s, family))
    if returncode not in (0, None):
        detail = _first_line_of_error(stderr)
        return write_result(
            output_dir, "failure", metrics=dict(summary), method_metrics=context,
            reason="osc2carla exited with code %s%s"
                   % (returncode, ": " + detail if detail else ""))
    if not summary:
        return write_result(
            output_dir, "error", method_metrics=context,
            reason="osc2carla exited cleanly but wrote no run summary to %s"
                   % os.path.relpath(metrics_path, output_dir))

    near_gap = _env_float("OSC2CARLA_NEAR_COLLISION_M") or DEFAULT_NEAR_COLLISION_M
    context["near_collision_gap_m"] = near_gap
    context["metrics_path"] = METRICS_FILE
    # What the bridge served the policy: which observation shape, which BEV
    # raster, whether the route ran short. A BEV substitution changes what a
    # number means, so it travels with the number.
    notes_path = os.path.join(output_dir, POLICY_NOTES_FILE)
    if os.path.exists(notes_path):
        try:
            with open(notes_path) as fh:
                context["policy_bridge"] = json.load(fh)
        except (OSError, ValueError) as exc:
            context["policy_bridge"] = {"unreadable": str(exc)}
    context["log_path"] = LOG_FILE
    if not _env_flag("OSC2CARLA_RECORD_VIDEO"):
        context["video_path"] = None
    else:
        context["video_path"] = family + ".mp4"

    metrics = canonical_metrics(summary, intent, near_gap)
    # Canonical names first, then the run summary verbatim: the harness splits
    # them and keeps everything it does not recognise under method_metrics.
    for key, value in summary.items():
        metrics.setdefault(key, value)
    return write_result(output_dir, "success", metrics=metrics, method_metrics=context)


if __name__ == "__main__":
    sys.exit(main())
