"""External ego-control entry point.

The compiled behaviour tree normally actuates every vehicle, the ego included.
This module provides the alternative: a documented hand-off point where the
ego's *actuation* is taken over by an external driving policy, while the rest
of the scenario -- NPC timelines, ``emit``/``wait`` events, monitors -- keeps
running exactly as compiled.

That split is the whole point.  The scenario still defines what the world does
around the ego and when the adversarial triggers fire (they key off the ego's
live position, not off its scripted timeline), so the same ``.osc`` file can be
used to exercise any policy under test.

Contract
--------
A policy is any object implementing :class:`EgoPolicy`::

    class MyPolicy(EgoPolicy):
        def setup(self, *, world, carla_map, actor, binding, params): ...
        def act(self, obs: Observation) -> Command: ...
        def teardown(self): ...

Selected on the command line by name or import path::

    python -m osc2carla scenario.osc --ego-policy idm
    python -m osc2carla scenario.osc --ego-policy mypkg.drivers:MyPolicy \
                                     --policy-param v0=8.3 --policy-param T=1.2

Perception (route sampling, leader detection) is done by
:class:`ExternalEgoController` and handed to the policy as a plain
:class:`Observation`; the policy returns a plain :class:`Command`.  Neither
type touches CARLA, so a policy is testable without a simulator.

Sensors
-------
A policy that needs to *see* declares its own rig -- ``sensors()`` or
``camera_rig`` (see :class:`EgoPolicy`) -- and :mod:`osc2carla.backend.sensors`
attaches exactly that to the ego, capturing each sensor's measurement for the
tick the observation describes.  ``Observation.sensors`` then carries those
measurements and ``Observation.route_ego()`` gives the route in the frame those
models are conditioned on.  A policy that declares no rig sees exactly what it
saw before: pose, speed, the sampled route and the closest leader.

That is the whole of what used to be missing, and it is why
``capabilities.json`` can now declare a ``sensor`` observation space on the
CARLA backend.  The local backend has no camera blueprints, so a sensor policy
is refused there rather than driven blind.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .simapi import sim as carla


# --------------------------------------------------------------------------
# Plain data handed across the boundary
# --------------------------------------------------------------------------

@dataclass
class RoutePoint:
    """A point on the ego's upcoming path, in world coordinates."""
    x: float
    y: float
    heading: float          # radians
    s: float                # arc length ahead of the ego, metres


@dataclass
class Leader:
    """The closest vehicle ahead of the ego on its own path."""
    gap: float              # bumper-to-bumper metres, clamped at >= 0
    speed: float            # m/s, projected on the ego's heading
    actor_id: int
    type_id: str


@dataclass
class Observation:
    t: float                        # simulated seconds
    speed: float                    # m/s
    x: float
    y: float
    heading: float                  # radians
    route: List[RoutePoint] = field(default_factory=list)
    leader: Optional[Leader] = None
    #: This tick's sensor measurements, keyed by the name the policy declared in
    #: ``sensors()``. Empty for a ``state``-only policy, which is every policy
    #: that ran before this repository grew a rig. A sensor that delivered
    #: nothing this tick is ABSENT rather than zero-filled: a model cannot tell
    #: an all-zero raster from a clear road, so the distinction has to survive.
    sensors: Dict[str, Any] = field(default_factory=dict)
    #: The world frame these measurements are stamped with, when the simulator
    #: reports one. Carried so a run can prove the images and the pose came from
    #: the same tick rather than assert it.
    frame: Optional[int] = None
    #: The ego's current road speed limit, km/h, or None where the simulator
    #: does not report one.
    speed_limit_kph: Optional[float] = None

    def route_ego(self) -> List[List[float]]:
        """The sampled route in the EGO frame: +x forward, +y right, metres.

        World coordinates are what this repository's own controllers use, but
        every route-conditioned driving model in this family is trained on the
        ego-frame form -- it is what ``get_relative_transform`` returns upstream.
        Converting here, in CARLA's own handedness, keeps the one conversion in
        the one place that knows the ego pose, and keeps a policy from having to
        rediscover the convention (and get the sign of ``y`` wrong).
        """
        c, s = math.cos(self.heading), math.sin(self.heading)
        out: List[List[float]] = []
        for rp in self.route:
            dx, dy = rp.x - self.x, rp.y - self.y
            out.append([dx * c + dy * s, -dx * s + dy * c])
        return out


@dataclass
class Command:
    """Normalised actuation. throttle/brake in [0,1], steer in [-1,1]."""
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0

    def clamped(self) -> "Command":
        return Command(
            throttle=min(1.0, max(0.0, self.throttle)),
            brake=min(1.0, max(0.0, self.brake)),
            steer=min(1.0, max(-1.0, self.steer)),
        )


# --------------------------------------------------------------------------
# Policy interface
# --------------------------------------------------------------------------

class EgoPolicy:
    """Base class for an external ego controller.

    Two optional hooks declare a sensor rig, and a policy that uses neither is a
    ``state``-only policy that never sees one:

    ``sensors()``
        Returns the rig to attach, as :mod:`~.sensors` specs or as plain dicts.
        This is the form an external policy repository should use, because a
        policy describing its own rig must not have to import an execution
        method to do it.

    ``camera_rig``
        A string naming one of :data:`~.sensors.RIGS`, for a policy that is
        trained behind a rig this repository already knows.

    A sensorimotor policy declares its own rig for a reason worth restating: it
    was trained behind one specific set of intrinsics and mountings, and a rig
    chosen by the runner instead would publish the runner's behaviour under the
    model's name.
    """

    name = "policy"

    def setup(self, *, world: Any, carla_map: Any, actor: Any,
              binding: str, params: Dict[str, float]) -> None:
        """Called once, after the ego has been spawned."""

    def act(self, obs: Observation) -> Command:  # pragma: no cover - abstract
        raise NotImplementedError

    def teardown(self) -> None:
        """Called once, when the run finishes."""


class IDMPolicy(EgoPolicy):
    """Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000).

    Longitudinal acceleration::

        a = a_max * [ 1 - (v/v0)^delta - (s*(v, dv) / s)^2 ]
        s* = s0 + max(0, v*T + v*dv / (2*sqrt(a_max*b)))

    with ``s`` the bumper-to-bumper gap to the leader and ``dv = v - v_lead``
    the approach rate.  With no leader the interaction term vanishes and the
    model is a plain free-flow controller converging on ``v0``.

    IDM is a *car-following* model: it is defined only along the lane and only
    with respect to a leader in front.  Two consequences matter for reading the
    benchmark results, and neither is a defect in this implementation:

      * lateral control is not part of IDM.  Steering here reuses the same
        pure-pursuit rule as the compiled ``drive()`` behaviour, so the policy
        follows the identical route through each junction and the comparison
        isolates the longitudinal policy.
      * a vehicle approaching from the side is not a leader.  IDM will not
        brake for a crossing conflict, because the model has no term for one.
    """

    name = "idm"

    # Defaults are the conventional urban-car parameter set.
    DEFAULTS: Dict[str, float] = {
        "v0": 8.333,        # desired free-flow speed, m/s (30 kph)
        "T": 1.5,           # desired time headway, s
        "a_max": 1.5,       # maximum acceleration, m/s^2
        "b": 2.0,           # comfortable deceleration, m/s^2
        "delta": 4.0,       # free-flow acceleration exponent
        "s0": 2.0,          # minimum standstill gap, m
        "lookahead": 5.0,   # steering lookahead, m
        "a_throttle": 3.0,  # accel that maps to full throttle, m/s^2
        "a_brake": 5.0,     # decel that maps to full brake, m/s^2
    }

    def __init__(self, **params: float):
        self.p = dict(self.DEFAULTS)
        self.p.update({k: float(v) for k, v in params.items() if k in self.DEFAULTS})
        self._unknown = sorted(set(params) - set(self.DEFAULTS))
        self.last_accel = 0.0

    def setup(self, *, world, carla_map, actor, binding, params) -> None:
        self.p.update({k: float(v) for k, v in params.items() if k in self.DEFAULTS})
        self._unknown = sorted(set(params) - set(self.DEFAULTS))

    # -- the model ---------------------------------------------------------

    def acceleration(self, v: float, leader: Optional[Leader]) -> float:
        p = self.p
        v0 = max(p["v0"], 1e-3)
        free = 1.0 - (max(v, 0.0) / v0) ** p["delta"]
        interaction = 0.0
        if leader is not None:
            s = max(leader.gap, 0.1)
            dv = v - leader.speed
            s_star = p["s0"] + max(
                0.0, v * p["T"] + (v * dv) / (2.0 * math.sqrt(p["a_max"] * p["b"]))
            )
            interaction = (s_star / s) ** 2
        return p["a_max"] * (free - interaction)

    def act(self, obs: Observation) -> Command:
        a = self.acceleration(obs.speed, obs.leader)
        self.last_accel = a
        cmd = Command()
        if a >= 0.0:
            cmd.throttle = a / self.p["a_throttle"]
        else:
            cmd.brake = -a / self.p["a_brake"]
        cmd.steer = _pure_pursuit_steer(obs, self.p["lookahead"])
        return cmd.clamped()


class ConstantSpeedPolicy(EgoPolicy):
    """Reference policy: hold a fixed speed, ignore everything.

    Present mainly to show that the entry point is not IDM-specific.
    """

    name = "constant"
    DEFAULTS = {"v0": 8.333, "kp": 0.6, "lookahead": 5.0}

    def __init__(self, **params: float):
        self.p = dict(self.DEFAULTS)
        self.p.update({k: float(v) for k, v in params.items() if k in self.DEFAULTS})

    def setup(self, *, world, carla_map, actor, binding, params) -> None:
        self.p.update({k: float(v) for k, v in params.items() if k in self.DEFAULTS})

    def act(self, obs: Observation) -> Command:
        out = self.p["kp"] * (self.p["v0"] - obs.speed)
        cmd = Command(steer=_pure_pursuit_steer(obs, self.p["lookahead"]))
        if out >= 0:
            cmd.throttle = out
        else:
            cmd.brake = -out
        return cmd.clamped()


def _pure_pursuit_steer(obs: Observation, lookahead: float) -> float:
    """Same steering rule as the compiled ``drive()`` behaviour."""
    target = None
    for rp in obs.route:
        if rp.s >= lookahead:
            target = rp
            break
    if target is None:
        if not obs.route:
            return 0.0
        target = obs.route[-1]
    err = math.atan2(target.y - obs.y, target.x - obs.x) - obs.heading
    err = (err + math.pi) % (2 * math.pi) - math.pi
    return max(-1.0, min(1.0, err))


# --------------------------------------------------------------------------
# Policy resolution
# --------------------------------------------------------------------------

BUILTIN_POLICIES: Dict[str, type] = {
    "idm": IDMPolicy,
    "constant": ConstantSpeedPolicy,
}


def resolve_policy(spec: str) -> type:
    """Resolve ``idm`` or ``package.module:ClassName`` to a policy class."""
    if spec in BUILTIN_POLICIES:
        return BUILTIN_POLICIES[spec]
    if ":" not in spec:
        raise ValueError(
            f"unknown ego policy {spec!r}; use one of "
            f"{sorted(BUILTIN_POLICIES)} or 'module:ClassName'"
        )
    mod_name, _, cls_name = spec.partition(":")
    module = importlib.import_module(mod_name)
    try:
        cls = getattr(module, cls_name)
    except AttributeError as exc:
        raise ValueError(f"{mod_name} has no attribute {cls_name!r}") from exc
    if not issubclass(cls, EgoPolicy):
        raise ValueError(f"{spec} is not an EgoPolicy subclass")
    return cls


def parse_policy_params(items: Sequence[str]) -> Dict[str, float]:
    """Parse repeated ``k=v`` CLI arguments into a float dict."""
    out: Dict[str, float] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"--policy-param expects k=v, got {item!r}")
        k, _, v = item.partition("=")
        out[k.strip()] = float(v)
    return out


# --------------------------------------------------------------------------
# The runtime hand-off
# --------------------------------------------------------------------------

class ExternalEgoController:
    """Drives one binding from an :class:`EgoPolicy` instead of the tree.

    Ticked once per simulation step, after the behaviour tree.  Does the
    perception the policy needs -- route ahead, closest leader on that route,
    and, for a policy that asks for one, the sensor rig's measurements for this
    exact tick -- and applies the returned command to the CARLA actor.

    Decision rate
    -------------
    ``decision_hz`` decouples the policy's decision rate from the simulation
    tick rate, holding the last command in between.  It defaults to 0, meaning
    "decide on every tick", which is what the analytic policies have always
    done and what keeps their results unchanged.  A network is the reason the
    knob exists: a VLA at 20 decisions per simulated second spends most of a
    run inside ``forward``, and the CARLA Leaderboard the sensorimotor
    checkpoints were trained under does not tick them that fast either.
    """

    def __init__(self, world, carla_map, ctx, binding: str, policy: EgoPolicy,
                 params: Optional[Dict[str, float]] = None,
                 route_step: float = 2.0, route_horizon: float = 60.0,
                 corridor: float = 2.2, decision_hz: float = 0.0):
        self.world = world
        self.map = carla_map
        self.ctx = ctx
        self.binding = binding
        self.policy = policy
        self.route_step = route_step
        self.route_horizon = route_horizon
        self.corridor = corridor
        self.decision_hz = float(decision_hz or 0.0)
        self.ticks = 0          # simulation steps this controller was asked for
        self.decisions = 0      # times the policy was actually consulted
        self.rig = None         # SensorRig, for a policy that declared one
        self._command: Optional[Command] = None
        self._last_decision_t: Optional[float] = None
        self._last_observation: Optional[Observation] = None
        self._actor = ctx.actor(binding)
        if self._actor is None:
            raise RuntimeError(f"ego policy: no spawned actor for binding {binding!r}")
        self._build_rig()
        self.policy.setup(world=world, carla_map=carla_map, actor=self._actor,
                          binding=binding, params=params or {})

    @property
    def actor(self):
        return self._actor

    # -- sensors -----------------------------------------------------------

    def _build_rig(self) -> None:
        """Attach whatever rig the policy asked for, before the run starts.

        Built before ``policy.setup`` so that a policy which loads a network in
        setup fails after the cheap step, not before it, and so that a policy
        may look at ``controller.rig`` from setup if it wants to.

        Nothing attached is not fatal here.  The local backend has no camera
        blueprints at all, and a policy that cannot see is the only thing that
        can say whether that ends the run -- so the failure is recorded, printed
        once, and left to the caller.  ``scenario_orchestration/run.py`` is where
        it becomes a refusal, because that is where the request says the policy
        needs a sensor observation space.
        """
        declared = None
        sensors = getattr(self.policy, "sensors", None)
        if callable(sensors):
            declared = sensors()
        else:
            named = getattr(self.policy, "camera_rig", None)
            if named:
                from .sensors import rig as named_rig
                declared = named_rig(str(named))
        if not declared:
            return
        from .sensors import SensorRig, specs_from
        self.rig = SensorRig(self.world, self._actor, specs_from(declared)).spawn()
        import sys
        if not self.rig.active:
            sys.stderr.write(
                "[osc2carla] sensor rig requested but nothing attached: %s\n"
                % ("; ".join(self.rig.failed) or "no sensors created"))
        else:
            sys.stderr.write(
                "[osc2carla] sensor rig: %s attached at %.0f Hz%s\n"
                % (", ".join(self.rig.names), self.rig.sensor_hz,
                   "; failed: " + "; ".join(self.rig.failed)
                   if self.rig.failed else ""))

    def _frame(self) -> Optional[int]:
        """The world frame this tick's measurements must be stamped with."""
        try:
            return self.world.get_snapshot().frame
        except (AttributeError, RuntimeError):
            return None

    # -- perception --------------------------------------------------------

    def observe(self, sim_time: float, sensors: bool = True) -> Observation:
        """The scene as of ``sim_time``.

        ``sensors=False`` builds the state half only. The state half is a
        handful of CARLA queries; the sensor half makes the simulator render,
        which is the expensive part of a rendered tick. So a tick that is only
        being measured, not decided on, skips it.
        """
        a = self._actor
        tf = a.get_transform()
        v = a.get_velocity()
        speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        route = self._sample_route()
        obs = Observation(
            t=sim_time,
            speed=speed,
            x=tf.location.x,
            y=tf.location.y,
            heading=math.radians(tf.rotation.yaw),
            route=route,
            frame=self._frame(),
            speed_limit_kph=self._speed_limit(),
        )
        obs.leader = self._find_leader(obs)
        if sensors and self.rig is not None and self.rig.active:
            # Captured after the pose is read and stamped with the same frame,
            # so the images and the state the policy reasons over are one tick.
            obs.sensors = self.rig.capture(obs.frame)
        return obs

    def _speed_limit(self) -> Optional[float]:
        try:
            limit = float(self._actor.get_speed_limit())
        except (RuntimeError, AttributeError, TypeError):
            return None
        # CARLA reports 0 until the vehicle has passed a speed-limit sign, and
        # a 0 limit read as a limit would tell a planner to stop.
        return limit if limit > 1.0 else None

    # -- the loop ----------------------------------------------------------

    def _due(self, sim_time: float) -> bool:
        if self.decision_hz <= 0 or self._command is None:
            return True
        return sim_time + 1e-9 >= (self._last_decision_t or 0.0) + 1.0 / self.decision_hz

    def tick(self, sim_time: float) -> Observation:
        """One simulation step: decide if due, otherwise hold the last command.

        Returns THIS tick's state, whether or not a decision was taken. The
        caller reads ``leader.gap`` off it for the run summary, and a run
        summary sampled at the policy's decision rate rather than the
        simulation's would report a closest approach that simply was not looked
        for -- the ego covers metres between two decisions of a slow policy.
        Only the sensors are skipped on a held tick, because only they cost a
        render. What the policy actually saw stays available separately, as
        :attr:`last_observation`.
        """
        self.ticks += 1
        if self._due(sim_time):
            obs = self.observe(sim_time)
            self._command = self.policy.act(obs).clamped()
            self._last_decision_t = sim_time
            self._last_observation = obs
            self.decisions += 1
        else:
            obs = self.observe(sim_time, sensors=False)
        if carla and self._command is not None:
            self._actor.apply_control(
                carla.VehicleControl(throttle=self._command.throttle,
                                     steer=self._command.steer,
                                     brake=self._command.brake)
            )
        return obs

    @property
    def command(self) -> Optional[Command]:
        """The command currently applied to the ego, held or fresh."""
        return self._command

    @property
    def last_observation(self) -> Optional[Observation]:
        """The observation the currently applied command was decided on.

        Not this tick's state -- see :meth:`tick`. This is what the policy saw,
        which is what a recording of the policy should show.
        """
        return self._last_observation

    def vision_panel(self, height: int = 0):
        """This tick's camera frames as one strip, or None.

        Reads the measurements the policy was ACTUALLY given rather than
        capturing again: a second capture would drain the sensor queues the
        controller is synchronising on, so the recording would silently change
        what the policy sees.
        """
        if self.rig is None or self._last_observation is None:
            return None
        return self.rig.panel(self._last_observation.sensors, height=height)

    def describe(self) -> Dict[str, Any]:
        """What this hand-off actually did, for the run summary."""
        out: Dict[str, Any] = {
            "policy": getattr(self.policy, "name", "policy"),
            "binding": self.binding,
            "observation_space": "state+sensor" if (
                self.rig is not None and self.rig.active) else "state",
            "decision_hz": self.decision_hz or None,
            "decisions": self.decisions,
            "control_steps": self.ticks,
            "sensor_rig": self.rig.describe() if self.rig is not None else None,
        }
        # Whatever the policy says about itself -- for a bridged policy that is
        # the checkpoint it loaded and, for a VLA, the text it generated. Passed
        # through a JSON filter because this ends up in a JSON summary and a
        # stray tensor in a report must not lose the whole run's metrics.
        described = getattr(self.policy, "metadata", None)
        if callable(described):
            try:
                out["policy_metadata"] = _jsonable(described())
            except Exception as exc:  # noqa: BLE001 - a report must not fail a run
                out["policy_metadata_error"] = str(exc)
        return out

    def teardown(self) -> None:
        if self.rig is not None:
            self.rig.destroy()
        try:
            self.policy.teardown()
        except Exception:  # noqa: BLE001
            pass

    # -- perception --------------------------------------------------------

    def _sample_route(self) -> List[RoutePoint]:
        """Path the ego would follow, sampled ahead of it.

        Uses the same ``waypoint.next()[0]`` rule the compiled ``drive()``
        behaviour uses, so an external policy inherits the identical route
        through a junction.
        """
        out: List[RoutePoint] = []
        if self.map is None:
            return out
        wp = self.map.get_waypoint(self._actor.get_location(), project_to_road=True)
        if wp is None:
            return out
        s = 0.0
        while s < self.route_horizon:
            nxt = wp.next(self.route_step)
            if not nxt:
                break
            wp = nxt[0]
            s += self.route_step
            t = wp.transform
            out.append(RoutePoint(x=t.location.x, y=t.location.y,
                                  heading=math.radians(t.rotation.yaw), s=s))
        return out

    def _find_leader(self, obs: Observation) -> Optional[Leader]:
        """Closest vehicle whose centre lies within the ego's path corridor.

        Distance is measured along the sampled route rather than straight
        ahead, so the leader is still found around a junction turn.
        """
        if self.world is None or not obs.route:
            return None
        ego = self._actor
        ego_half = _half_length(ego)
        fx, fy = math.cos(obs.heading), math.sin(obs.heading)
        best: Optional[Leader] = None
        for other in self.world.get_actors().filter("vehicle.*"):
            if other.id == ego.id:
                continue
            loc = other.get_location()
            lateral, s_at = self._project_on_route(obs.route, loc.x, loc.y)
            if lateral is None or lateral > self.corridor:
                continue
            gap = s_at - ego_half - _half_length(other)
            if gap < -2.0:
                continue
            if best is not None and gap >= best.gap:
                continue
            ov = other.get_velocity()
            best = Leader(
                gap=max(gap, 0.0),
                speed=ov.x * fx + ov.y * fy,
                actor_id=other.id,
                type_id=other.type_id,
            )
        return best

    @staticmethod
    def _project_on_route(route: Sequence[RoutePoint], x: float, y: float
                          ) -> Tuple[Optional[float], float]:
        best_d: Optional[float] = None
        best_s = 0.0
        for rp in route:
            d = math.hypot(x - rp.x, y - rp.y)
            if best_d is None or d < best_d:
                best_d = d
                best_s = rp.s
        return best_d, best_s


def _jsonable(value: Any, depth: int = 0) -> Any:
    """``value`` reduced to something ``json.dump`` will accept.

    Anything it does not recognise becomes its ``repr``, truncated. A policy's
    self-description is free-form by design -- it is the policy repository's
    text, not ours -- so the run summary has to survive whatever is in it.
    """
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if depth > 6:
        return repr(value)[:200]
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in value]
    return repr(value)[:200]


def _half_length(actor) -> float:
    try:
        return float(actor.bounding_box.extent.x)
    except Exception:  # noqa: BLE001
        return 2.4
