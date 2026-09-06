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
    OSC2CARLA_POLICY_NOTES        where to write what this bridge actually did,
                                  so the run report can carry it
    OSC2CARLA_POLICY_OBS_DUMP     where to write the first observations and the
                                  actions they produced, for diagnosis. The BEV
                                  raster is summarised rather than dumped.

Analytic policies (the IDM family, the constant-speed reference) never reach
this file: ``run.py`` realizes those natively, which is what their harness
configs say every method should do.

What the external policy sees
-----------------------------
The declared observation space is ``state``, and that space is one contract
shared with every other method in the harness -- not a per-method dialect. So on
the CARLA backend the observation is the full object-centric document
(``scenario_orchestration/carla_state_obs.py``): ego speed, ego-frame object
tokens, a 20-point ego-frame route, the posted speed limit, and -- when a BEV
source is available -- the semantic raster an object-centric planner needs::

    {"ego": {"speed_mps": 7.9},
     "objects": [{"type": "car", "position": [12.0, 0.2], "yaw_rad": 0.01,
                  "speed_mps": 8.0, "extent": [2.45, 1.06, 0.75], "id": "hero"},
                 {"type": "traffic_light", "position": [20.0, 0.0],
                  "state": "Red", ...}],
     "route": [[2.5, 0.0], [3.5, 0.0], ...],
     "speed_limit_kph": 50,
     "bev": {"semantic_classes": [[...]]}}

This repository's own car-following view is carried alongside rather than
replaced, so a policy written against it keeps working::

    {"t": 4.2, "speed_mps": 7.9, "x": .., "y": .., "heading_rad": ..,
     "route_world": [{"x": .., "y": .., "heading_rad": .., "s": 2.0}, ...],
     "leader": {"gap_m": 12.4, "speed_mps": 6.1, "actor_id": 42,
                "type_id": "vehicle.audi.a2"}}

Note ``route_world``: ``route`` is the harness contract's ego-frame ``[x, y]``
list, so the world-frame route this repository samples for its own IDM moves to
its own key rather than colliding with it.

On the local backends there is no CARLA world to describe, so only the
car-following view is sent and ``objects``/``bev`` are absent. A policy that
needs them fails with its own explicit error, which is the correct outcome: a
substituted observation would be a silently wrong result.

A policy that needs to SEE gets one more block. It declares its own rig through
``sensors()`` and ``osc2carla.backend.sensors`` attaches exactly that, so the
observation additionally carries::

    {"sensor": {"cameras": {"rgb_front": HxWx3 uint8 RGB,
                            "lidar": Nx4, "radar1": Nx4, ...},
                "frame": 91823}}

Every sensor is filed under ``cameras`` by its declared name, range sensors
included -- one oddly named namespace, and the one the installed adapters
already read. A sensor that delivered nothing on a tick is ABSENT rather than
zero-filled: a model cannot tell an all-zero raster from a clear road, so the
policy has to be the one that decides what a missing sensor means.

What comes back
---------------
The declared action space is ``control`` or ``waypoints``. A ``control`` action
may be a mapping or an object carrying ``throttle`` / ``brake`` / ``steer`` in
the usual [0,1] and [-1,1] ranges, or ``acceleration_mps2`` (plus ``steer``),
converted with the same pedal mapping the built-in IDM uses.

A nested ``control`` block is read whichever action space is declared, because
that is what a policy predicting waypoints returns alongside them and both
installed sensorimotor policies declare ``control`` for exactly that reason.

A ``waypoints`` policy is accepted through the ``control`` block its *own*
lateral and longitudinal controllers return alongside its waypoints. That is
deliberate: it reproduces the policy's published closed-loop behaviour instead of
re-deriving a controller here, and it is the same actuation path the
orchestration method uses, so the two methods drive such a policy the same way. A
waypoint policy that returns no control cannot be actuated and says so.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from typing import Any, Dict, List, Optional

from osc2carla.backend.policy import Command, EgoPolicy, Observation

#: This file's own directory. ``--ego-policy`` names this module by bare name, so
#: it is imported without package context and its siblings have to be reached by
#: path rather than by ``from . import``.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _sibling(name: str):
    """Import a module that lives next to this file, under a private name.

    Private, because these names are generic enough to collide with a policy
    repository's own modules once its directory is on ``sys.path``.
    """
    path = os.path.join(_HERE, name + ".py")
    spec = importlib.util.spec_from_file_location("_osc2carla_" + name, path)
    if spec is None or spec.loader is None:         # pragma: no cover
        raise PolicyBridgeError("cannot import %s from %s" % (name, _HERE))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

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


def _carla_python_api() -> Optional[str]:
    """``$CARLA_ROOT/PythonAPI/carla``, when that is a real directory.

    That tree ships with the CARLA *server* and carries the ``agents`` package
    -- ``agents.navigation``, ``GlobalRoutePlanner``, ``RoadOption`` -- which the
    pip ``carla`` wheel does not. Every route-conditioned policy in this family
    imports something from it, directly or through its own controller, and the
    installed ones disagree about whose job it is to find it: SimLingo's adapter
    puts it on ``sys.path`` itself, TFv6's expects a symlink inside its own
    checkout. Doing it here means neither has to be right.

    ``CARLA_ROOT`` is upstream's own name for this and is a property of the
    machine, which is why it is read from the environment rather than declared
    in a config.
    """
    root = (os.environ.get("CARLA_ROOT") or "").strip()
    if not root:
        return None
    api = os.path.join(root, "PythonAPI", "carla")
    return api if os.path.isdir(os.path.join(api, "agents")) else None


def _load_module(path: str):
    """Import a policy repository's ``policy.py`` by path, not by package name.

    Its own repository goes on ``sys.path`` first so the file's internal
    imports resolve, but the module itself is loaded under a private name: two
    policy repositories may both call theirs ``policy``.
    """
    if not os.path.exists(path):
        raise PolicyBridgeError("policy entry point %r does not exist" % path)
    repository = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for candidate in (repository, os.path.dirname(os.path.abspath(path)),
                      _carla_python_api()):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)
    name = "_osc2carla_external_policy"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PolicyBridgeError("cannot import a policy from %r" % path)
    module = importlib.util.module_from_spec(spec)
    # Registered *before* execution, not after. A module body that defines a
    # dataclass -- which a policy declaring its own PolicyRequest mirror does --
    # makes `dataclasses` look itself up as `sys.modules[cls.__module__]`, and an
    # unregistered module fails there with an AttributeError on None that says
    # nothing about the real cause. Removed again if the body raises, so a failed
    # import cannot leave a half-built module for the next attempt to find.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except BaseException:
        sys.modules.pop(name, None)
        raise
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
    """This repository's own car-following view, as plain JSON-able data.

    ``route_world`` rather than ``route``: the harness's ``state`` space defines
    ``route`` as an ego-frame ``[x, y]`` list, and the object-centric builder owns
    that key. Keeping the world-frame samples under their own name means neither
    view has to be dropped.
    """
    payload: Dict[str, Any] = {
        "t": obs.t,
        "speed_mps": obs.speed,
        "x": obs.x,
        "y": obs.y,
        "heading_rad": obs.heading,
        "route_world": [{"x": p.x, "y": p.y, "heading_rad": p.heading, "s": p.s}
                        for p in obs.route],
        "leader": None if obs.leader is None else {
            "gap_m": obs.leader.gap,
            "speed_mps": obs.leader.speed,
            "actor_id": obs.leader.actor_id,
            "type_id": obs.leader.type_id,
        },
    }
    if obs.sensors:
        # Every sensor under `cameras`, by the name the policy declared: that is
        # the key the installed adapters read, LiDAR and radar included. Absent
        # entirely when the rig delivered nothing, so "the sensor is missing" and
        # "the sensor sees nothing" stay different statements.
        payload["sensor"] = {"cameras": dict(obs.sensors), "frame": obs.frame}
    return payload


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
    """A policy's own ``control``, when it returned one.

    An object-centric planner emits waypoints and, alongside them, the control
    its *own* lateral and longitudinal controllers derive from those waypoints.
    Consuming that block is what reproduces the policy's published closed-loop
    behaviour rather than re-deriving a controller here.

    Read whichever action space is declared, not only ``waypoints``. Both
    installed sensorimotor policies declare ``control`` precisely because they
    return the control their own controllers produced -- so the nested block is
    the normal shape of a ``control`` action, not a waypoint policy's quirk, and
    a reader that only unwrapped it for ``waypoints`` rejected them as carrying
    no control at all.
    """
    if isinstance(action, dict):
        block = action.get("control")
    else:
        block = getattr(action, "control", None)
    return block


def _to_command(action: Any, action_space: str = "control") -> Command:
    """Normalise whatever the policy returned into a pedal command."""
    if action is None:
        raise PolicyBridgeError("the external policy returned no action")
    block = _control_block(action)
    if block is None and action_space == "waypoints":
        raise PolicyBridgeError(
            "policy declares action space 'waypoints' and returned %r, which "
            "carries no 'control' block. This runtime actuates a waypoint "
            "policy through the control its own controllers return alongside "
            "its waypoints; a waypoint list on its own has no actuation path "
            "here, and inventing a follower would make the result a "
            "measurement of that follower rather than of the policy"
            % (sorted(action) if isinstance(action, dict) else type(action).__name__,)
        )
    if block is not None:
        action = block
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


def _summarize_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    """The observation with the raster replaced by a description of it.

    A 256x256 raster per tick would bury the thing being diagnosed, and its
    class histogram answers the only question worth asking of it: whether the
    ego is standing on road.
    """
    out = {k: v for k, v in observation.items() if k != "bev"}
    bev = observation.get("bev")
    if isinstance(bev, dict) and "semantic_classes" in bev:
        classes = bev["semantic_classes"]
        try:
            import numpy as np

            array = np.asarray(classes)
            values, counts = np.unique(array, return_counts=True)
            centre = array[array.shape[0] // 2, array.shape[1] // 2]
            out["bev"] = {
                "shape": list(array.shape),
                "class_histogram": {int(v): int(c) for v, c in zip(values, counts)},
                "class_at_centre": int(centre),
            }
        except Exception as exc:  # noqa: BLE001
            out["bev"] = {"unsummarizable": str(exc)}
    elif bev is not None:
        out["bev"] = {"present": True}
    return out


def _summarize_action(action: Any) -> Any:
    if isinstance(action, dict):
        return {k: v for k, v in action.items() if k != "meta"}
    return repr(action)[:400]


class BridgedPolicy(EgoPolicy):
    """This repository's ``EgoPolicy``, backed by an external policy repository."""

    name = "bridged"

    def __init__(self, **params: float) -> None:
        self._params = dict(params)
        self._policy: Any = None
        self._act = None
        self._loaded = False
        self.request: Dict[str, Any] = {}
        self.action_space = "control"
        #: The last observation handed to the policy, without the frames. Read
        #: only by the video overlay; nothing in the control path touches it.
        self.last_observation_payload: Optional[Dict[str, Any]] = None
        self._observer: Any = None
        self._bev: Any = None
        self._notes: Dict[str, Any] = {}
        self._dump_path = os.environ.get("OSC2CARLA_POLICY_OBS_DUMP") or None
        self._dump: List[Dict[str, Any]] = []
        self._dump_limit = 12
        self._acts = 0

    # -- osc2carla's EgoPolicy contract ------------------------------------

    def _ensure_constructed(self) -> None:
        """Read the request and build the policy object.

        Split out of ``setup`` because ``ExternalEgoController`` asks for
        ``sensors()`` BEFORE it calls ``setup`` -- it has to, since the rig is
        attached to the ego before the policy is set up against it -- and only
        the external policy knows what rig it wants. Nothing in here needs a
        world, which is what makes the split possible.
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
        self.action_space = str(self.request.get("action_space") or "control")
        self._absolutize(os.environ.get("OSC2CARLA_POLICY_ROOT"))

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

    #: Request keys whose value is a path when it is one. ``checkpoint`` is in
    #: the contract; the rest are the conventional names a policy repository
    #: uses for a file inside its checkpoint, and are only rewritten when the
    #: rewrite actually lands on something.
    PATH_PARAMETERS = ("weights", "checkpoint", "config", "config_path",
                       "model_path", "weights_path")

    def _absolutize(self, root: Optional[str]) -> None:
        """Make the request's relative checkpoint paths absolute.

        The harness writes paths relative to its own root -- ``checkpoint:
        third_party/checkpoints/simlingo/simlingo`` -- because that is where the
        declaration lives and where an operator reads it. The policy resolves
        them against the process's working directory, which is the method's, not
        the harness's, and which this runner deliberately lets an operator move
        (a vision policy caches a multi-gigabyte backbone under it). Those two
        facts only agreed by accident.

        Rewritten only when the rewrite lands on something that exists, so a
        genuinely missing checkpoint still fails naming the path the harness
        asked for rather than a path this file invented.
        """
        if not root or not os.path.isdir(root):
            return
        rewritten = []

        def resolve(value):
            if not isinstance(value, str) or not value or os.path.isabs(value):
                return value
            candidate = os.path.join(root, value)
            if not os.path.exists(candidate):
                return value
            rewritten.append(value)
            return os.path.abspath(candidate)

        self.request["checkpoint"] = resolve(self.request.get("checkpoint"))
        parameters = self.request.get("parameters")
        if isinstance(parameters, dict):
            for key in self.PATH_PARAMETERS:
                if key in parameters:
                    parameters[key] = resolve(parameters[key])
        if rewritten:
            sys.stderr.write(
                "[policy_bridge] resolved %d request path(s) against the "
                "harness root %s: %s\n"
                % (len(rewritten), root, ", ".join(rewritten)))

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
        entry_point = os.environ.get("OSC2CARLA_POLICY_ENTRY_POINT") or ""

        # The object-centric half of the `state` space, when there is a CARLA
        # world to describe. Built before the policy's own setup so a missing
        # BEV raster or dependency fails here, where the run can still report it,
        # rather than on the first tick.
        self._build_observer(world, carla_map, actor, entry_point)

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

        sys.stderr.write("[policy_bridge] %s loaded from %s (action space %s, "
                         "observation %s)\n"
                         % (self.name, entry_point, self.action_space,
                            "state+objects" if self._observer else "state"))

    # -- the observation ---------------------------------------------------

    def _build_observer(self, world, carla_map, actor, entry_point: str) -> None:
        """Prepare the object-centric observation builder, if this backend can.

        The local backends have no CARLA world, so there is nothing to describe
        object-centrically and the car-following view is all a policy gets. That
        is recorded rather than worked around: a policy needing objects will
        raise its own error, which is the right outcome.
        """
        if world is None or carla_map is None:
            self._notes["observation"] = (
                "car-following view only: this backend exposes no CARLA world, so "
                "no object-centric scene or BEV raster can be described")
            return
        state_obs = _sibling("carla_state_obs")

        # The BEV raster comes from the policy repository's own renderer. Its
        # absence is not fatal here: a checkpoint that needs one refuses the
        # observation itself, with its own message.
        repository = os.path.dirname(os.path.dirname(os.path.abspath(entry_point)))
        try:
            self._bev = _sibling("bev").build(repository).prepare(actor)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            self._bev = None
            self._notes["bev"] = "unavailable: %s: %s" % (type(exc).__name__, exc)
            sys.stderr.write("[policy_bridge] BEV renderer unavailable: %s\n" % exc)
        else:
            # Described at teardown rather than here: at setup the renderer has
            # produced nothing, and a snapshot taken now would report zero
            # rasters for a run that rendered one per tick.
            self._notes["bev"] = "prepared"

        self._observer = state_obs.StateObservationBuilder(
            world=world, carla_map=carla_map,
            bev=self._bev if self._bev is not None else None,
        )
        self._ego_actor = actor

    def observation(self, obs: Observation) -> Dict[str, Any]:
        """The document handed to the policy: both views, one dict."""
        payload = _observation_payload(obs)
        if self._observer is not None:
            payload.update(self._observer.build(self._ego_actor, obs.speed))
        return payload

    def describe(self) -> Dict[str, Any]:
        """What this bridge did, for the run report."""
        notes = dict(self._notes)
        if self._bev is not None:
            notes["bev"] = self._bev.describe()
        if self._observer is not None:
            notes["state"] = self._observer.describe()
        notes["action_space"] = self.action_space
        notes["acted"] = self._acts
        return notes

    def metadata(self) -> Dict[str, Any]:
        """Whatever the external policy reports about itself, plus the request.

        Distinct from ``describe()`` and read by a different consumer:
        ``describe`` feeds ``$OSC2CARLA_POLICY_NOTES``, which records what this
        BRIDGE did, while this is picked up by
        ``ExternalEgoController.describe()`` into the run summary's
        ``ego_policy_detail`` -- where a reader looks to confirm the checkpoint
        was the one they meant, or to read what a VLA said about the frame it
        braked on.
        """
        out: Dict[str, Any] = {
            "name": self.name,
            "implementation": self.request.get("implementation"),
            "entry_point": os.environ.get("OSC2CARLA_POLICY_ENTRY_POINT"),
            "checkpoint": self.request.get("checkpoint"),
            "observation_space": self.request.get("observation_space"),
            "action_space": self.action_space,
        }
        described = getattr(self._policy, "metadata", None)
        if callable(described):
            try:
                out["policy_metadata"] = described()
            except Exception as exc:  # noqa: BLE001 - a report must not fail a run
                out["policy_metadata_error"] = str(exc)
        return out

    def act(self, obs: Observation) -> Command:
        observation = self.observation(obs)
        # Kept without the images: the overlay wants the route and the numbers,
        # and holding a reference to every frame would keep the whole run's
        # camera data alive for the sake of a video band.
        self.last_observation_payload = {k: v for k, v in observation.items()
                                         if k != "sensor"}
        action = self._act(observation)
        self._acts += 1
        command = _to_command(action, self.action_space)
        if self._dump_path and len(self._dump) < self._dump_limit:
            self._dump.append({
                "t": obs.t,
                "observation": _summarize_observation(observation),
                "action": _summarize_action(action),
                "command": {"throttle": command.throttle, "brake": command.brake,
                            "steer": command.steer},
            })
        return command

    def teardown(self) -> None:
        self._write_notes()
        self._write_dump()
        if self._bev is not None:
            try:
                self._bev.close()
            except Exception:  # noqa: BLE001 - teardown must not fail a run
                pass
        for method in ("close", "teardown"):
            closer = getattr(self._policy, method, None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - teardown must not fail a run
                    pass

    def _write_dump(self) -> None:
        if not self._dump_path or not self._dump:
            return
        try:
            with open(self._dump_path, "w") as fh:
                json.dump(self._dump, fh, indent=2, default=str)
        except OSError as exc:  # noqa: BLE001
            sys.stderr.write("[policy_bridge] could not write obs dump: %s\n" % exc)

    def _write_notes(self) -> None:
        """Record which observation and which raster the policy actually got.

        A BEV substitution or a short route changes what a number means, so it
        belongs in the result rather than only in a log line that nobody reads.
        Failing to write it must not fail the run.
        """
        path = os.environ.get("OSC2CARLA_POLICY_NOTES")
        if not path:
            return
        try:
            with open(path, "w") as fh:
                json.dump(self.describe(), fh, indent=2, default=str)
        except OSError as exc:  # noqa: BLE001
            sys.stderr.write("[policy_bridge] could not write notes: %s\n" % exc)
