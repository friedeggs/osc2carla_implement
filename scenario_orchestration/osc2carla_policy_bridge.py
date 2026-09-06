"""Load an external ``ego_policy_v1`` policy into this repository's runtime.

DESIGN.md section 6 puts the cost of policy integration at ``M + P``: a policy
repository exposes one ``scenario_orchestration/policy.py`` and knows nothing
about any execution method, and each method translates the standardized
``PolicyRequest`` into its own policy plumbing. This file is that translation
for osc2carla_implement.

``run.py`` selects it with::

    python -m osc2carla <scenario>.osc \
        --ego-policy osc2carla_policy_bridge:BridgedPolicy

and points it at the request through the environment, because
``--ego-policy`` names a class and ``--policy-param`` carries numbers only::

    OSC2CARLA_POLICY_REQUEST      path to the harness's policy.json
    OSC2CARLA_POLICY_ENTRY_POINT  path to the policy repository's policy.py

Analytic policies (the IDM family, the constant-speed reference) never reach
this file: ``run.py`` realizes those natively, which is what their harness
configs say every method should do.

What the external policy sees
-----------------------------
One JSON-able mapping per decision -- no CARLA types, no types from this
repository, so a policy is testable against a recorded observation::

    {"t": 4.2,
     "ego": {"speed_mps": 7.9, "x": .., "y": .., "heading_rad": ..},
     "route": [[x, y], ...],            # EGO frame, +x forward, +y right
     "speed_limit_kph": 50.0,
     "sensor": {"cameras": {"rgb_front": HxWx3 uint8 RGB,
                            "lidar": Nx4, "radar1": Nx4, ...}},
     "leader": {"gap_m": 12.4, "speed_mps": 6.1, ...} | None,
     # the flat pose keys this bridge has always sent, unchanged
     "speed_mps": 7.9, "x": .., "y": .., "heading_rad": ..}

Three details in there are load-bearing, and each was a silent failure before
it was a line of code.

**The route is in the ego frame, and resampled.** Every route-conditioned model
in this family reads ``[[x, y], ...]`` with +x forward and +y right, one point
per metre from ~2.5 m ahead -- that is the sampling their target-point indexing
assumes (SimLingo takes points 7 and -1; TFv6 takes 0, 7 and 15 and measures the
heading change across the whole polyline). ``ExternalEgoController`` samples the
road every 2 m because that is what its own controllers want, so :func:`_resample`
puts it back on the grid the policies index against. Handing over the raw
sampling is not an error anything reports: the model simply conditions on a
target point twice as far away as it was trained to.

**Every sensor is filed under ``sensor.cameras``, LiDAR and radar included.**
That reads oddly and is deliberate: it is the key the installed policy adapters
already look their sensors up in, by the name they declared. Renaming it here
would be renaming it in two repositories that are not ours.

**A sensor that delivered nothing is absent, not zeroed.** A model cannot tell an
all-zero LiDAR raster from a clear road, so the policy has to be the one that
decides what a missing sensor means.

What comes back
---------------
The declared action space is ``control``. A modern policy returns
``{"control": {"throttle": .., "steer": .., "brake": ..}, ...}``; the older flat
form, and an ``acceleration_mps2``, are still accepted. An ``acceleration`` is
converted with the same pedal mapping the built-in IDM uses.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from osc2carla.backend.policy import Command, EgoPolicy, Observation

#: Factory functions tried on the policy module, in order.
FACTORIES = ("build_policy", "make_policy", "create_policy", "load_policy",
             "policy_from_request", "get_policy")

#: Class names tried when the module exposes no factory.
CLASS_NAMES = ("Policy", "EgoPolicy", "Agent")

#: Methods tried on the constructed policy, in order.
ACT_METHODS = ("act", "step", "run_step", "__call__")

#: Same pedal mapping as osc2carla's built-in IDM, for policies that return an
#: acceleration rather than pedals.
ACCEL_TO_THROTTLE = 3.0
ACCEL_TO_BRAKE = 5.0

#: Route sampling handed to the policy: one point per metre, starting this far
#: ahead, this many of them. The reference agents in this family are trained and
#: evaluated on exactly this grid, and their adapters index it positionally.
ROUTE_FIRST_M = 2.5
ROUTE_STEP_M = 1.0
ROUTE_POINTS = 20


class PolicyBridgeError(RuntimeError):
    """The external policy could not be loaded or spoken to."""


def _load_module(path: str):
    """Import a policy repository's ``policy.py`` by path, not by package name.

    Its own repository goes on ``sys.path`` first so the file's internal
    imports resolve, but the module itself is loaded under a private name: two
    policy repositories may both call theirs ``policy``.
    """
    if not os.path.exists(path):
        raise PolicyBridgeError("policy entry point %r does not exist" % path)
    repository = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for candidate in (repository, os.path.dirname(os.path.abspath(path))):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    spec = importlib.util.spec_from_file_location("_osc2carla_external_policy", path)
    if spec is None or spec.loader is None:
        raise PolicyBridgeError("cannot import a policy from %r" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


#: Constructor/factory parameter names that mean "the whole PolicyRequest".
REQUEST_PARAMETERS = ("request", "policy_request", "spec", "policy_spec")


def _call_for_request(func, request: Dict[str, Any]) -> Any:
    """Call ``func`` the way its own signature asks to be called.

    The contract says a policy repository constructs its policy *from a
    PolicyRequest*, but says nothing about the shape of the callable that does
    it. Rather than guess an order and risk handing a request dict to a
    parameter that wanted a float, read the signature and pick.
    """
    parameters = dict(request.get("parameters") or {})
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        formal = [p for p in signature.parameters.values() if p.name != "self"]
        names = [p.name for p in formal]
        takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in formal)
        required = [p for p in formal
                    if p.default is inspect.Parameter.empty
                    and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                   inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        for name in REQUEST_PARAMETERS:
            if name in names:
                return func(**{name: request})
        if takes_kwargs:
            return func(**parameters)
        shared = {k: v for k, v in parameters.items() if k in names}
        if shared:
            return func(**shared)
        if len(required) == 1:
            return func(request)
        if not required:
            return func()
    # No usable signature (a C-level or heavily decorated callable): try the
    # documented form, then the empty one.
    try:
        return func(request)
    except TypeError:
        return func()


def _construct(module, request: Dict[str, Any]) -> Any:
    """Build the policy object from whatever the module exposes."""
    for name in FACTORIES:
        factory = getattr(module, name, None)
        if callable(factory):
            return _call_for_request(factory, request)
    implementation = str(request.get("implementation") or "")
    named = implementation.rsplit(".", 1)[-1].rsplit(":", 1)[-1]
    for name in [named] + list(CLASS_NAMES):
        cls = getattr(module, name, None) if name else None
        if isinstance(cls, type):
            return _call_for_request(cls, request)
    raise PolicyBridgeError(
        "policy module %s exposes none of %s, and no class among %s: it does "
        "not implement the ego_policy_v1 entry point this method knows how to "
        "load"
        % (getattr(module, "__file__", "?"), ", ".join(FACTORIES),
           ", ".join([named] + list(CLASS_NAMES))))


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------

def _resample(points: Sequence[Sequence[float]], first_m: float = ROUTE_FIRST_M,
              step_m: float = ROUTE_STEP_M, count: int = ROUTE_POINTS
              ) -> List[List[float]]:
    """``count`` points along ``points``, ``step_m`` apart from ``first_m``.

    Arc-length resampling of a polyline that already starts at the ego. The
    controller samples the road every 2 m; the reference agents index a 1 m
    grid, so without this the same index means a different distance and every
    target point lands twice as far out as the model expects.

    Returned SHORT rather than padded when the polyline runs out: the policy
    adapters pad by repeating their last point, and a short route is a fact
    about the road worth being able to see. Never returned longer than the
    source can support, so no point is invented.
    """
    if len(points) < 2:
        return [list(p) for p in points]
    # Cumulative arc length, with the ego itself at s = 0.
    spans: List[Tuple[float, float, Sequence[float], Sequence[float]]] = []
    travelled = 0.0
    previous: Sequence[float] = (0.0, 0.0)
    for point in points:
        length = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if length > 1e-9:
            spans.append((travelled, length, previous, point))
            travelled += length
        previous = point
    out: List[List[float]] = []
    target = first_m
    for start, length, a, b in spans:
        while target <= start + length + 1e-9 and len(out) < count:
            f = (target - start) / length
            out.append([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f])
            target += step_m
        if len(out) >= count:
            break
    return out


def _observation_payload(obs: Observation) -> Dict[str, Any]:
    """The mapping handed to the external policy for one decision."""
    payload: Dict[str, Any] = {
        "t": obs.t,
        # The flat pose keys this bridge has always sent. Kept because a policy
        # written against the state-only contract must keep working: adding a
        # sensor stream is not a reason to break the arm that never wanted one.
        "speed_mps": obs.speed,
        "x": obs.x,
        "y": obs.y,
        "heading_rad": obs.heading,
        "ego": {"speed_mps": obs.speed, "x": obs.x, "y": obs.y,
                "heading_rad": obs.heading},
        "route": _resample(obs.route_ego()),
        "leader": None if obs.leader is None else {
            "gap_m": obs.leader.gap,
            "speed_mps": obs.leader.speed,
            "actor_id": obs.leader.actor_id,
            "type_id": obs.leader.type_id,
        },
    }
    if obs.speed_limit_kph is not None:
        payload["speed_limit_kph"] = obs.speed_limit_kph
    if obs.sensors:
        # Every sensor under `cameras`, by the name the policy declared: that is
        # the key the installed adapters read, LiDAR and radar included.
        payload["sensor"] = {"cameras": dict(obs.sensors), "frame": obs.frame}
    return payload


# ---------------------------------------------------------------------------
# action
# ---------------------------------------------------------------------------

def _field(action: Any, *names: str) -> Optional[float]:
    for name in names:
        value = None
        if isinstance(action, dict):
            value = action.get(name)
        elif hasattr(action, name):
            value = getattr(action, name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _control_block(action: Any) -> Any:
    """The part of the action carrying the pedals.

    ``ego_policy_v1``'s ``control`` action space is a nested ``control`` block,
    which is what both installed sensorimotor adapters return alongside their
    predicted waypoints and their commentary. The flat form -- pedals at the top
    level -- is what this bridge accepted before, and analytic policies still
    use it, so both are read and the nested one wins.
    """
    if isinstance(action, dict):
        block = action.get("control")
        if isinstance(block, dict):
            return block
    control = getattr(action, "control", None)
    if isinstance(control, dict):
        return control
    return action


def _to_command(action: Any) -> Command:
    """Normalise whatever the policy returned into a pedal command."""
    if action is None:
        raise PolicyBridgeError("the external policy returned no action")
    control = _control_block(action)
    steer = _field(control, "steer", "steering", "steer_norm") or 0.0
    throttle = _field(control, "throttle")
    brake = _field(control, "brake")
    if throttle is None and brake is None:
        accel = _field(control, "acceleration_mps2", "acceleration", "accel",
                       "accel_mps2", "a")
        if accel is None:
            raise PolicyBridgeError(
                "the external policy returned %r, which carries no "
                "throttle/brake/steer and no acceleration; the declared action "
                "space is 'control'" % (action,))
        throttle = accel / ACCEL_TO_THROTTLE if accel >= 0.0 else 0.0
        brake = 0.0 if accel >= 0.0 else -accel / ACCEL_TO_BRAKE
    return Command(throttle=throttle or 0.0, brake=brake or 0.0,
                   steer=steer).clamped()


class BridgedPolicy(EgoPolicy):
    """This repository's ``EgoPolicy``, backed by an external policy repository."""

    name = "bridged"

    def __init__(self, **params: float) -> None:
        self._params = dict(params)
        self._policy: Any = None
        self._act = None
        self._loaded = False
        self.request: Dict[str, Any] = {}
        #: The last action the policy returned, verbatim. Read only by
        #: `metadata()` and the video overlay; nothing in the control path.
        self.last_action: Optional[Dict[str, Any]] = None

    # -- loading -----------------------------------------------------------

    def _ensure_constructed(self) -> None:
        """Import the policy repository and build its policy object.

        Split out of ``setup`` because ``ExternalEgoController`` asks for
        ``sensors()`` BEFORE it calls ``setup`` -- it has to, since the rig is
        attached to the ego before the policy is set up against it -- and only
        the external policy knows what rig it wants.
        """
        if self._loaded:
            return
        request_path = os.environ.get("OSC2CARLA_POLICY_REQUEST")
        entry_point = os.environ.get("OSC2CARLA_POLICY_ENTRY_POINT")
        if not request_path or not entry_point:
            raise PolicyBridgeError(
                "OSC2CARLA_POLICY_REQUEST and OSC2CARLA_POLICY_ENTRY_POINT must "
                "both be set; this policy is selected by "
                "scenario_orchestration/run.py, not from the command line")
        with open(request_path) as fh:
            self.request = json.load(fh)
        self.name = str(self.request.get("name") or "bridged")

        module = _load_module(entry_point)
        self._policy = _construct(module, self.request)
        for method in ACT_METHODS:
            candidate = getattr(self._policy, method, None)
            if callable(candidate):
                self._act = candidate
                break
        if self._act is None:
            raise PolicyBridgeError(
                "policy %r exposes none of %s, so it cannot be asked for an "
                "action" % (self.name, ", ".join(ACT_METHODS)))
        self._loaded = True
        sys.stderr.write("[policy_bridge] %s loaded from %s\n"
                         % (self.name, entry_point))

    # -- osc2carla's EgoPolicy contract ------------------------------------

    def sensors(self):
        """The rig the external policy asks for, or None for a state-only one.

        Returned as the policy gave it -- plain dicts, in the documented form --
        so ``osc2carla.backend.sensors.specs_from`` does the normalising and no
        policy repository has to import this method's spec types.
        """
        self._ensure_constructed()
        declared = getattr(self._policy, "sensors", None)
        if not callable(declared):
            return None
        rig = declared()
        return list(rig) if rig else None

    def setup(self, *, world, carla_map, actor, binding, params) -> None:
        self._ensure_constructed()

        # The policy's own setup, if it has one. Its signature is its business:
        # try the informative call first, then the empty one.
        setup = getattr(self._policy, "setup", None)
        if callable(setup):
            try:
                setup(request=self.request, binding=binding,
                      seed=self.request.get("seed"),
                      parameters=dict(self.request.get("parameters") or {}))
            except TypeError:
                setup()

        # Build the network here, not on the first tick. A checkpoint that
        # cannot be loaded, a missing CUDA device or an unimportable inference
        # stack should stop the run while the scenario is still being set up --
        # otherwise it surfaces as a policy that drove badly for one tick and
        # then crashed, which reads as a scenario failure.
        loader = getattr(self._policy, "load", None)
        if callable(loader):
            loader()
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def act(self, obs: Observation) -> Command:
        action = self._act(_observation_payload(obs))
        if action is None:
            raise PolicyBridgeError(
                "policy %r returned no action for t=%.2fs" % (self.name, obs.t))
        self.last_action = action if isinstance(action, dict) else None
        return _to_command(action)

    def metadata(self) -> Dict[str, Any]:
        """Whatever the external policy reports about itself, plus the request.

        Ends up in this run's summary under ``ego_policy_detail.policy_metadata``
        -- which is where a reader looks to find out that the checkpoint was the
        one they meant, or what the VLA said about the frame it braked on.
        """
        out: Dict[str, Any] = {
            "name": self.name,
            "implementation": self.request.get("implementation"),
            "entry_point": os.environ.get("OSC2CARLA_POLICY_ENTRY_POINT"),
            "checkpoint": self.request.get("checkpoint"),
            "observation_space": self.request.get("observation_space"),
            "action_space": self.request.get("action_space"),
        }
        described = getattr(self._policy, "metadata", None)
        if callable(described):
            try:
                out["policy_metadata"] = described()
            except Exception as exc:  # noqa: BLE001 - a report must not fail a run
                out["policy_metadata_error"] = str(exc)
        return out

    def teardown(self) -> None:
        for method in ("close", "teardown"):
            closer = getattr(self._policy, method, None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - teardown must not fail a run
                    pass
