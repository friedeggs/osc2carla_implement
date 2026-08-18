"""Actors: vehicles with a bicycle-model chassis, static props, sensors.

The read surface mirrors ``carla.Actor`` closely enough that
``atomic_behaviors``, ``policy`` and ``metrics`` cannot tell the difference:
``get_transform()``, ``get_location()``, ``get_velocity()``,
``apply_control()``, ``bounding_box.extent``, ``type_id``, ``attributes``,
``destroy()``.

Vehicle dynamics are a kinematic bicycle model with 2-D momentum: driving is
kinematic (speed follows the throttle/brake command, heading follows the
steering angle), but the velocity is carried as a vector so a collision
impulse can push a car sideways and spin it.  That is the cheapest model that
still makes the crash scenarios look like crashes.
"""
from __future__ import annotations

import fnmatch
import math
from typing import Any, Callable, Dict, Optional

from .blueprints import prop_spec, vehicle_spec
from .datatypes import (CollisionEvent, VehicleControl, VehicleLightState,
                        parse_color)
from .geometry import BoundingBox, Location, Rotation, Transform, Vector3D


class VehiclePhysics:
    """Per-model chassis constants, in SI units."""

    __slots__ = ("max_accel", "max_decel", "max_steer", "wheelbase",
                 "drag", "rolling", "lateral_damping", "spin_damping")

    def __init__(self, mass: float, half_length: float):
        # Heavy vehicles accelerate and stop less sharply, but not in
        # proportion to their mass: a loaded truck makes perhaps 1.5 m/s^2,
        # not an eighth of a car's. Scaling linearly gave a 12 t HGV
        # 0.48 m/s^2, slow enough that traffic behind it rear-ended it before
        # it had cleared its own spawn point.
        scale = max(0.45, min(1.0, math.sqrt(1600.0 / max(mass, 200.0))))
        self.max_accel = 3.6 * scale          # m/s^2 at throttle = 1
        self.max_decel = 8.5 * min(1.0, scale + 0.35)   # m/s^2 at brake = 1
        self.max_steer = math.radians(38.0)
        self.wheelbase = max(1.6, half_length * 1.45)
        self.drag = 0.0016                    # m^-1, sets the top speed
        self.rolling = 0.12                   # m/s^2
        self.lateral_damping = 7.0            # 1/s, kills sideslip
        self.spin_damping = 2.2               # 1/s


class Actor:
    """Base class for everything the world simulates."""

    is_solid = False
    immovable = True

    def __init__(self, world, actor_id: int, type_id: str, transform: Transform,
                 attributes: Optional[Dict[str, str]] = None):
        self.world = world
        self.id = actor_id
        self.type_id = type_id
        self.attributes: Dict[str, str] = dict(attributes or {})
        self.parent: Optional["Actor"] = None
        self.is_alive = True
        self._x = transform.location.x
        self._y = transform.location.y
        self._z = transform.location.z
        self._yaw_deg = transform.rotation.yaw
        self._vx = 0.0
        self._vy = 0.0
        self._spin = 0.0
        self.mass = 0.0
        #: decals report contact but never exchange momentum
        self.is_decal = False
        self.bounding_box = BoundingBox(extent=Vector3D(0.5, 0.5, 0.5))

    # -- CARLA read surface ----------------------------------------------

    def get_location(self) -> Location:
        return Location(self._x, self._y, self._z)

    def get_transform(self) -> Transform:
        return Transform(Location(self._x, self._y, self._z),
                         Rotation(yaw=self._yaw_deg))

    def set_transform(self, transform: Transform) -> None:
        self._x = transform.location.x
        self._y = transform.location.y
        self._z = transform.location.z
        self._yaw_deg = transform.rotation.yaw

    def set_location(self, location: Location) -> None:
        self._x, self._y, self._z = location.x, location.y, location.z

    def get_velocity(self) -> Vector3D:
        return Vector3D(self._vx, self._vy, 0.0)

    def get_angular_velocity(self) -> Vector3D:
        return Vector3D(0.0, 0.0, math.degrees(self._spin))

    def get_acceleration(self) -> Vector3D:
        return Vector3D(0.0, 0.0, 0.0)

    @property
    def collision_radius(self) -> float:
        e = self.bounding_box.extent
        return math.hypot(e.x, e.y)

    @property
    def speed(self) -> float:
        return math.hypot(self._vx, self._vy)

    @property
    def yaw(self) -> float:
        """Heading in radians."""
        return math.radians(self._yaw_deg)

    def destroy(self) -> bool:
        if not self.is_alive:
            return False
        self.is_alive = False
        self.world._remove_actor(self)
        return True

    # -- simulation -------------------------------------------------------

    def step(self, dt: float) -> None:
        """Advance one tick. Static actors do nothing."""

    def __repr__(self) -> str:
        role = self.attributes.get("role_name", "")
        return (f"<{type(self).__name__} id={self.id} type={self.type_id}"
                f"{' role=' + role if role else ''}>")


class Vehicle(Actor):
    is_solid = True
    immovable = False

    def __init__(self, world, actor_id: int, type_id: str, transform: Transform,
                 attributes: Optional[Dict[str, str]] = None):
        super().__init__(world, actor_id, type_id, transform, attributes)
        hl, hw, hh, mass, default_rgb = vehicle_spec(type_id)
        self.bounding_box = BoundingBox(extent=Vector3D(hl, hw, hh))
        self.mass = mass
        self.physics = VehiclePhysics(mass, hl)
        self.color = parse_color(self.attributes.get("color"), default_rgb)
        self._control = VehicleControl()
        self._light_state = VehicleLightState(VehicleLightState.NONE)
        #: True on the tick a control was applied; the renderer draws brake
        #: lights from the command rather than from a light_state the
        #: scenario may never set.
        self.braking = False

    # -- actuation --------------------------------------------------------

    def apply_control(self, control: VehicleControl) -> None:
        self._control = control

    def get_control(self) -> VehicleControl:
        return self._control

    def set_light_state(self, light_state) -> None:
        self._light_state = VehicleLightState(int(light_state))

    def get_light_state(self) -> VehicleLightState:
        return self._light_state

    def set_target_velocity(self, vector: Vector3D) -> None:
        self._vx, self._vy = vector.x, vector.y

    def set_simulate_physics(self, enabled: bool) -> None:  # accepted, ignored
        pass

    # -- dynamics ---------------------------------------------------------

    def step(self, dt: float) -> None:
        p = self.physics
        c = self._control
        yaw = math.radians(self._yaw_deg)
        hx, hy = math.cos(yaw), math.sin(yaw)
        v_long = self._vx * hx + self._vy * hy
        v_lat = -self._vx * hy + self._vy * hx

        throttle = min(1.0, max(0.0, c.throttle))
        brake = min(1.0, max(0.0, c.brake)) + (1.0 if c.hand_brake else 0.0)
        brake = min(1.0, brake)
        self.braking = brake > 0.05

        accel = throttle * p.max_accel
        accel -= p.drag * v_long * abs(v_long)
        accel -= p.rolling * (1.0 if v_long > 0.01 else 0.0)
        decel = brake * p.max_decel
        if v_long > 0.0:
            accel -= decel
        v_long += accel * dt
        if brake > 0.0 and v_long < 0.0:
            v_long = 0.0                      # brakes stop, they do not reverse
        if throttle <= 0.0 and abs(v_long) < 0.02:
            v_long = 0.0
        v_long = max(v_long, 0.0 if not c.reverse else -8.0)

        # heading: bicycle steering plus whatever spin a collision imparted
        delta = min(1.0, max(-1.0, c.steer)) * p.max_steer
        yaw_rate = (v_long / p.wheelbase) * math.tan(delta)
        self._spin *= math.exp(-p.spin_damping * dt)
        yaw += (yaw_rate + self._spin) * dt
        self._yaw_deg = math.degrees(yaw)

        v_lat *= math.exp(-p.lateral_damping * dt)
        hx, hy = math.cos(yaw), math.sin(yaw)
        self._vx = v_long * hx - v_lat * hy
        self._vy = v_long * hy + v_lat * hx
        self._x += self._vx * dt
        self._y += self._vy * dt
        # actors are spawned lifted clear of the ground; settle them back
        self._z = max(0.0, self._z - 4.0 * dt)


#: props flatter than this are treated as ground decals
DECAL_HEIGHT = 0.10


class StaticProp(Actor):
    """A ground marker or obstacle. Solid, but never moves.

    A prop flatter than :data:`DECAL_HEIGHT` is a *decal*: the scenarios use
    ``static.prop.dirtdebris01`` as a distance reference painted on the road,
    and in CARLA driving over one raises a collision event without moving the
    car.  Reproducing only the first half of that -- an immovable box that
    deflects whatever touches it -- would silently wreck any scenario whose
    reference marker sits on a driving line.
    """

    is_solid = True
    immovable = True

    def __init__(self, world, actor_id: int, type_id: str, transform: Transform,
                 attributes: Optional[Dict[str, str]] = None):
        super().__init__(world, actor_id, type_id, transform, attributes)
        hl, hw, hh = prop_spec(type_id)
        self.bounding_box = BoundingBox(extent=Vector3D(hl, hw, hh))
        self.mass = 0.0
        self.is_decal = hh < DECAL_HEIGHT
        self.color = (150, 120, 70)


class Sensor(Actor):
    """Base sensor: a callback, a parent, and no geometry."""

    def __init__(self, world, actor_id: int, type_id: str, transform: Transform,
                 attributes: Optional[Dict[str, str]] = None,
                 attach_to: Optional[Actor] = None):
        super().__init__(world, actor_id, type_id, transform, attributes)
        self.parent = attach_to
        self._callback: Optional[Callable[[Any], None]] = None
        self.is_listening = False

    def listen(self, callback: Callable[[Any], None]) -> None:
        self._callback = callback
        self.is_listening = True

    def stop(self) -> None:
        self.is_listening = False

    def _emit(self, data) -> None:
        if self.is_listening and self._callback is not None:
            self._callback(data)

    def get_transform(self) -> Transform:
        if self.parent is not None:
            return self.parent.get_transform()
        return super().get_transform()


class CollisionSensor(Sensor):
    """Fires once per contacting pair per tick, as CARLA does per substep."""

    def notify(self, frame: int, timestamp: float, other: Actor,
               impulse: Vector3D) -> None:
        if self.parent is None or not self.parent.is_alive:
            return
        self._emit(CollisionEvent(frame, timestamp, self.parent, other, impulse))


class ActorList(list):
    """``world.get_actors()`` result, with CARLA's wildcard ``filter``."""

    def filter(self, pattern: str) -> "ActorList":
        return ActorList(a for a in self if fnmatch.fnmatch(a.type_id, pattern))

    def find(self, actor_id: int) -> Optional[Actor]:
        for a in self:
            if a.id == actor_id:
                return a
        return None
