"""The object-centric ``state`` observation an external ego policy sees.

Why this file exists
--------------------
``osc2carla/backend/policy.py`` hands a policy an :class:`Observation` shaped for
a car-following controller: ego pose, speed, a world-frame route and the closest
leader. That is everything the built-in IDM needs and nothing an object-centric
planner needs.

``scenario_orchestration``'s ``state`` observation space is the wider document --
an ego-frame scene description -- and the shape is not ours to invent: it is
already pinned by what the policy repositories in that harness consume, and by
the reference realization in the orchestration method's own
``carla_port/carla_obs.py``. This module builds that same document from the CARLA
world this backend already owns, so a policy that runs under one method runs
under this one unchanged. That is the whole point of the observation space being
*one* contract rather than one per method.

Frame
-----
Metres, ``+x`` forward, ``+y`` right, yaw in radians relative to the ego heading
-- CARLA's own left-handed convention, which is what
``carla_garage.transfuser_utils.get_relative_transform`` returns and therefore
what the object-centric planners in this family are trained on. Getting the
handedness wrong hands a policy a mirrored world, which it will drive into.

What is deliberately NOT here
-----------------------------
The BEV raster. It is the policy repository's own representation, produced by its
own renderer from its own prebuilt town maps, so it arrives as an injected
callable; see ``scenario_orchestration/bev.py``. This module stays free of any one
policy's internals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Objects further than this from the ego are not serialized. The policy applies
#: its own, tighter, model-specific gate; this only keeps the document small on a
#: busy map.
RANGE_M = 75.0

#: Route sampling. Not free parameters: PlanT's route embedding is a fixed
#: ``Linear(20 * 2, ...)`` and the reference agents sample one point per metre
#: starting 2.5 m ahead, so matching that is what makes numbers comparable.
ROUTE_POINTS = 20
ROUTE_FIRST_M = 2.5
ROUTE_STEP_M = 1.0

#: A traffic light is only described while it is this close.
LIGHT_RANGE_M = 30.0

#: The urban default, and what the object-centric planners fall back to.
DEFAULT_SPEED_LIMIT_KPH = 50.0

#: ``carla.TrafficLightState`` -> the state string the observation carries. Only
#: red and amber are fed to the model: a green light carries no constraint, which
#: is the reference agents' own filtering.
LIGHT_STATES = {"Red": "Red", "Yellow": "Yellow", "Green": "Green",
                "Off": "Green", "Unknown": "Green"}


def _normalize_angle(rad: float) -> float:
    """Wrap to (-pi, pi]."""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class EgoFrame:
    """The rigid transform that maps CARLA world coordinates to the ego frame."""

    x: float
    y: float
    z: float
    yaw_rad: float

    @classmethod
    def of(cls, actor) -> "EgoFrame":
        tf = actor.get_transform()
        loc, rot = tf.location, tf.rotation
        return cls(x=float(loc.x), y=float(loc.y), z=float(loc.z),
                   yaw_rad=math.radians(float(rot.yaw)))

    def to_ego(self, X: float, Y: float, Z: Optional[float] = None
               ) -> Tuple[float, float, float]:
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        dx, dy = X - self.x, Y - self.y
        dz = 0.0 if Z is None else Z - self.z
        return (dx * c + dy * s, -dx * s + dy * c, dz)

    def relative_yaw(self, yaw_deg: float) -> float:
        return _normalize_angle(math.radians(float(yaw_deg)) - self.yaw_rad)


@dataclass
class StateObservationBuilder:
    """Builds one ``state`` observation per control step.

    ``bev`` is the policy-specific raster source: a callable taking the ego's
    CARLA actor and returning whatever that policy's ``bev`` field expects, or
    ``None``. Injected rather than imported, so this module never learns what a
    particular checkpoint was trained on.
    """

    world: Any
    carla_map: Any
    bev: Optional[Callable[[Any], Any]] = None
    range_m: float = RANGE_M
    route_points: int = ROUTE_POINTS
    route_first_m: float = ROUTE_FIRST_M
    route_step_m: float = ROUTE_STEP_M
    notes: List[str] = field(default_factory=list)
    #: Set once, so a run report can say the route was short rather than leaving
    #: the policy's own padding to look like a full route.
    short_routes: int = 0

    # ------------------------------------------------------------------ #
    def build(self, actor, speed_mps: float) -> Dict[str, Any]:
        ego = EgoFrame.of(actor)
        observation: Dict[str, Any] = {
            "ego": {"speed_mps": float(speed_mps)},
            "objects": self._objects(ego, actor),
            "route": self._route(ego, actor),
            "speed_limit_kph": self._speed_limit(actor),
        }
        if self.bev is not None:
            raster = self.bev(actor)
            if raster is not None:
                observation["bev"] = {"semantic_classes": raster}
        return observation

    # ------------------------------------------------------------------ #
    # Objects
    # ------------------------------------------------------------------ #
    def _objects(self, ego: EgoFrame, ego_actor) -> List[Dict[str, Any]]:
        """Every vehicle, walker and blocking light the ego could perceive.

        Read from CARLA rather than from the compiled scenario: the policy under
        test is being evaluated on what a driver could see, and the behaviour
        tree's own view does not include actors it did not spawn.
        """
        out: List[Dict[str, Any]] = []
        ego_id = getattr(ego_actor, "id", None)
        for actor in self._actors():
            if getattr(actor, "id", None) == ego_id:
                continue
            kind = self._classify(actor)
            if kind is None:
                continue
            tf = actor.get_transform()
            x, y, z = ego.to_ego(tf.location.x, tf.location.y, tf.location.z)
            if x * x + y * y > self.range_m ** 2:
                continue
            extent = self._extent(actor)
            out.append({
                "type": kind,
                "position": [x, y, z],
                "yaw_rad": ego.relative_yaw(tf.rotation.yaw),
                "speed_mps": self._speed(actor),
                "extent": list(extent),
                "type_id": getattr(actor, "type_id", None),
                # The scenario's own name for the actor when it has one. Not part
                # of the observation contract, but it is what makes a recorded
                # trace readable next to the scenario's monitors.
                "id": self._role_name(actor) or f"carla:{getattr(actor, 'id', '?')}",
            })
        out.extend(self._lights(ego, ego_actor))
        return out

    def _actors(self) -> List[Any]:
        try:
            actors = self.world.get_actors()
        except (RuntimeError, AttributeError):        # pragma: no cover
            return []
        try:
            return list(actors.filter("*vehicle*")) + list(actors.filter("*walker*"))
        except (AttributeError, TypeError):           # pragma: no cover
            return [a for a in actors if self._classify(a) is not None]

    @staticmethod
    def _classify(actor) -> Optional[str]:
        type_id = str(getattr(actor, "type_id", "") or "")
        if type_id.startswith("vehicle"):
            return "car"
        if type_id.startswith("walker"):
            return "walker"
        return None

    @staticmethod
    def _role_name(actor) -> Optional[str]:
        attributes = getattr(actor, "attributes", None) or {}
        try:
            role = attributes.get("role_name")
        except AttributeError:                        # pragma: no cover
            return None
        return str(role) if role else None

    @staticmethod
    def _extent(actor) -> Tuple[float, float, float]:
        """CARLA *half*-extents (length, width, height), which is what
        ``carla.BoundingBox.extent`` reports and what the policies expect."""
        box = getattr(actor, "bounding_box", None)
        extent = getattr(box, "extent", None)
        if extent is None:                            # pragma: no cover
            return (2.4, 1.0, 0.8)
        return (float(extent.x), float(extent.y), float(extent.z))

    @staticmethod
    def _speed(actor) -> float:
        try:
            v = actor.get_velocity()
        except (RuntimeError, AttributeError):        # pragma: no cover
            return 0.0
        return math.sqrt(float(v.x) ** 2 + float(v.y) ** 2 + float(v.z) ** 2)

    # ------------------------------------------------------------------ #
    # Traffic lights
    # ------------------------------------------------------------------ #
    def _lights(self, ego: EgoFrame, ego_actor) -> List[Dict[str, Any]]:
        """The light governing the ego, described at its stop line.

        ``actor.get_traffic_light()`` is CARLA's own answer to "which light
        applies to this vehicle", so the scenario's junction topology never has
        to be tabulated here. The position is the stop line rather than the light
        head, because the stop line is where a planner has to stop.
        """
        try:
            light = ego_actor.get_traffic_light()
        except (AttributeError, RuntimeError):        # pragma: no cover
            return []
        if light is None:
            return []
        state = LIGHT_STATES.get(str(getattr(light, "state", "Green")), "Green")
        if state == "Green":
            return []
        out: List[Dict[str, Any]] = []
        for wp in self._stop_lines(light):
            tf = wp.transform
            x, y, z = ego.to_ego(tf.location.x, tf.location.y, tf.location.z)
            if x * x + y * y > LIGHT_RANGE_M ** 2:
                continue
            out.append({
                "type": "traffic_light",
                "position": [x, y, z],
                "yaw_rad": ego.relative_yaw(tf.rotation.yaw),
                "state": state,
                "extent": [1.5, 1.5, 0.5],
            })
        return out

    @staticmethod
    def _stop_lines(light) -> List[Any]:
        try:
            return list(light.get_stop_waypoints())
        except (AttributeError, RuntimeError):        # pragma: no cover
            return []

    # ------------------------------------------------------------------ #
    # Route
    # ------------------------------------------------------------------ #
    def _route(self, ego: EgoFrame, actor) -> List[List[float]]:
        """``route_points`` ego-frame points along the ego's intended route.

        Sampled with the same ``waypoint.next()[0]`` rule the compiled ``drive()``
        behaviour uses, so a policy inherits the identical path through a
        junction -- including which exit it takes. Route conditioning is what
        tells a planner which way it is meant to go, so it has to be the route
        the scenario is about, not a straight line.

        Points are never invented. A route that runs out is returned short and
        counted, so the policy's own padding is visible in the report rather than
        looking like a full route.
        """
        if self.carla_map is None:
            return []
        try:
            wp = self.carla_map.get_waypoint(actor.get_location(),
                                             project_to_road=True)
        except (RuntimeError, AttributeError):        # pragma: no cover
            return []
        if wp is None:
            return []

        picked: List[List[float]] = []
        # Walk in route_step_m increments and start emitting at route_first_m,
        # so the spacing the policy was trained on is the spacing it gets.
        travelled = 0.0
        guard = 0
        limit = int((self.route_first_m
                     + self.route_points * self.route_step_m) / self.route_step_m) + 8
        while len(picked) < self.route_points and guard < limit:
            guard += 1
            try:
                nxt = wp.next(self.route_step_m)
            except (RuntimeError, AttributeError):    # pragma: no cover
                break
            if not nxt:
                break
            wp = nxt[0]
            travelled += self.route_step_m
            if travelled + 1e-9 < self.route_first_m:
                continue
            tf = wp.transform
            x, y, _z = ego.to_ego(tf.location.x, tf.location.y, None)
            picked.append([x, y])
        if len(picked) < self.route_points:
            self.short_routes += 1
        return picked

    # ------------------------------------------------------------------ #
    def _speed_limit(self, actor) -> float:
        """The posted limit in km/h, from CARLA's own answer for this vehicle."""
        try:
            limit = float(actor.get_speed_limit())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return DEFAULT_SPEED_LIMIT_KPH
        # CARLA reports 0 before the vehicle has passed any speed-limit sign.
        return limit if limit > 0.0 else DEFAULT_SPEED_LIMIT_KPH

    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        return {
            "source": "scenario_orchestration/carla_state_obs.py",
            "frame": "ego (+x forward, +y right), CARLA handedness",
            "range_m": self.range_m,
            "route": {"points": self.route_points,
                      "first_m": self.route_first_m,
                      "step_m": self.route_step_m,
                      "short": self.short_routes},
            "bev": "injected" if self.bev is not None else None,
        }
