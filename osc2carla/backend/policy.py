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
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import carla  # type: ignore
except Exception:  # noqa: BLE001
    carla = None  # type: ignore


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
    """Base class for an external ego controller."""

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
    perception the policy needs (route ahead, closest leader on that route)
    and applies the returned command to the CARLA actor.
    """

    def __init__(self, world, carla_map, ctx, binding: str, policy: EgoPolicy,
                 params: Optional[Dict[str, float]] = None,
                 route_step: float = 2.0, route_horizon: float = 60.0,
                 corridor: float = 2.2):
        self.world = world
        self.map = carla_map
        self.ctx = ctx
        self.binding = binding
        self.policy = policy
        self.route_step = route_step
        self.route_horizon = route_horizon
        self.corridor = corridor
        self.ticks = 0
        self._actor = ctx.actor(binding)
        if self._actor is None:
            raise RuntimeError(f"ego policy: no spawned actor for binding {binding!r}")
        self.policy.setup(world=world, carla_map=carla_map, actor=self._actor,
                          binding=binding, params=params or {})

    @property
    def actor(self):
        return self._actor

    def observe(self, sim_time: float) -> Observation:
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
        )
        obs.leader = self._find_leader(obs)
        return obs

    def tick(self, sim_time: float) -> Observation:
        obs = self.observe(sim_time)
        cmd = self.policy.act(obs).clamped()
        if carla is not None:
            self._actor.apply_control(
                carla.VehicleControl(throttle=cmd.throttle, steer=cmd.steer,
                                     brake=cmd.brake)
            )
        self.ticks += 1
        return obs

    def teardown(self) -> None:
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


def _half_length(actor) -> float:
    try:
        return float(actor.bounding_box.extent.x)
    except Exception:  # noqa: BLE001
        return 2.4
