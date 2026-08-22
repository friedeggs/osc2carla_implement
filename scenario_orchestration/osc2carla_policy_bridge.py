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
The declared observation space is ``state``, so the observation handed across
is a plain JSON-able dict -- no CARLA types, no types from this repository::

    {"t": 4.2, "speed_mps": 7.9, "x": .., "y": .., "heading_rad": ..,
     "route": [{"x": .., "y": .., "heading_rad": .., "s": 2.0}, ...],
     "leader": {"gap_m": 12.4, "speed_mps": 6.1, "actor_id": 42,
                "type_id": "vehicle.audi.a2"}}

``leader`` is ``None`` when nothing is ahead on the ego's own path. The
declared action space is ``control``; an action may come back as a mapping or
an object carrying ``throttle`` / ``brake`` / ``steer`` in the usual [0,1] and
[-1,1] ranges, or as ``acceleration_mps2`` (plus ``steer``), which is converted
with the same pedal mapping the built-in IDM uses.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from typing import Any, Dict, Optional

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


def _observation_payload(obs: Observation) -> Dict[str, Any]:
    return {
        "t": obs.t,
        "speed_mps": obs.speed,
        "x": obs.x,
        "y": obs.y,
        "heading_rad": obs.heading,
        "route": [{"x": p.x, "y": p.y, "heading_rad": p.heading, "s": p.s}
                  for p in obs.route],
        "leader": None if obs.leader is None else {
            "gap_m": obs.leader.gap,
            "speed_mps": obs.leader.speed,
            "actor_id": obs.leader.actor_id,
            "type_id": obs.leader.type_id,
        },
    }


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


def _to_command(action: Any) -> Command:
    """Normalise whatever the policy returned into a pedal command."""
    if action is None:
        raise PolicyBridgeError("the external policy returned no action")
    steer = _field(action, "steer", "steering", "steer_norm") or 0.0
    throttle = _field(action, "throttle")
    brake = _field(action, "brake")
    if throttle is None and brake is None:
        accel = _field(action, "acceleration_mps2", "acceleration", "accel",
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
        self.request: Dict[str, Any] = {}

    # -- osc2carla's EgoPolicy contract ------------------------------------

    def setup(self, *, world, carla_map, actor, binding, params) -> None:
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
        sys.stderr.write("[policy_bridge] %s loaded from %s\n"
                         % (self.name, entry_point))

    def act(self, obs: Observation) -> Command:
        return _to_command(self._act(_observation_payload(obs)))

    def teardown(self) -> None:
        teardown = getattr(self._policy, "teardown", None)
        if callable(teardown):
            try:
                teardown()
            except Exception:  # noqa: BLE001 - teardown must not fail a run
                pass
