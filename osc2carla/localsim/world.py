"""The world: actor registry, fixed-step integrator, sensor dispatch.

``World`` is the object ``ScenarioInitializer``, ``ExecutionContext`` and the
atomic behaviours are handed.  It reproduces the parts of ``carla.World`` they
touch, and nothing else.  Time only moves when :meth:`tick` is called, i.e.
the local simulator is always in CARLA's synchronous mode -- which is how the
compiler drives CARLA anyway.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from . import collision as _collision
from .actors import Actor, ActorList, CollisionSensor, StaticProp, Vehicle
from .blueprints import BlueprintLibrary
from .datatypes import (Timestamp, WeatherParameters, WorldSettings,
                        WorldSnapshot)
from .geometry import Transform, Vector3D, corners_2d
from .roadmap import Map
from .towns import load_town


class World:
    def __init__(self, road_map: Map, fixed_delta_seconds: float = 0.05):
        self._map = road_map
        self._blueprints = BlueprintLibrary()
        self._settings = WorldSettings(synchronous_mode=True,
                                       fixed_delta_seconds=fixed_delta_seconds)
        self._weather = WeatherParameters()
        self._actors: List[Actor] = []
        self._next_id = 1
        self.frame = 0
        self.elapsed_seconds = 0.0
        self._on_tick: Dict[int, Callable[[WorldSnapshot], None]] = {}
        self._next_tick_id = 1
        #: contacts detected on the most recent tick, for the renderer
        self.last_contacts: List[_collision.Contact] = []

    # -- CARLA-shaped accessors ------------------------------------------

    def get_map(self) -> Map:
        return self._map

    def get_blueprint_library(self) -> BlueprintLibrary:
        return self._blueprints

    def get_settings(self) -> WorldSettings:
        return self._settings.copy()

    def apply_settings(self, settings: WorldSettings) -> int:
        if settings.fixed_delta_seconds:
            self._settings.fixed_delta_seconds = float(settings.fixed_delta_seconds)
        self._settings.synchronous_mode = True     # the only mode there is
        self._settings.no_rendering_mode = settings.no_rendering_mode
        return self.frame

    def get_weather(self) -> WeatherParameters:
        return self._weather.copy()

    def set_weather(self, weather: WeatherParameters) -> None:
        self._weather = weather.copy()

    def get_actors(self, actor_ids: Optional[List[int]] = None) -> ActorList:
        if actor_ids is None:
            return ActorList(a for a in self._actors if a.is_alive)
        wanted = set(actor_ids)
        return ActorList(a for a in self._actors if a.is_alive and a.id in wanted)

    def get_snapshot(self) -> WorldSnapshot:
        dt = self._settings.fixed_delta_seconds or 0.05
        return WorldSnapshot(self.frame,
                             Timestamp(self.frame, self.elapsed_seconds, dt))

    def on_tick(self, callback) -> int:
        tick_id = self._next_tick_id
        self._next_tick_id += 1
        self._on_tick[tick_id] = callback
        return tick_id

    def remove_on_tick(self, tick_id: int) -> None:
        self._on_tick.pop(tick_id, None)

    # -- spawning ---------------------------------------------------------

    def spawn_actor(self, blueprint, transform: Transform,
                    attach_to: Optional[Actor] = None) -> Actor:
        actor = self.try_spawn_actor(blueprint, transform, attach_to=attach_to)
        if actor is None:
            raise RuntimeError(
                f"spawn failed for {blueprint.id} at "
                f"({transform.location.x:.1f}, {transform.location.y:.1f}): "
                f"the spot is occupied")
        return actor

    def try_spawn_actor(self, blueprint, transform: Transform,
                        attach_to: Optional[Actor] = None) -> Optional[Actor]:
        bp_id = blueprint.id
        attrs = blueprint.attributes
        actor_id = self._next_id

        if bp_id.startswith("sensor."):
            if bp_id != "sensor.other.collision":
                return None
            sensor = CollisionSensor(self, actor_id, bp_id, transform, attrs,
                                     attach_to=attach_to)
            self._next_id += 1
            self._actors.append(sensor)
            return sensor

        if bp_id.startswith("static."):
            actor: Actor = StaticProp(self, actor_id, bp_id, transform, attrs)
        elif bp_id.startswith("vehicle."):
            actor = Vehicle(self, actor_id, bp_id, transform, attrs)
        else:
            return None

        if self._blocked(actor):
            return None
        self._next_id += 1
        self._actors.append(actor)
        return actor

    def _blocked(self, candidate: Actor) -> bool:
        """CARLA refuses a spawn whose collision box overlaps a live actor.

        ``ScenarioInitializer._spawn_at`` relies on that: it retries the same
        spot lifted 0.4 m at a time, so the vertical separation has to count.
        """
        ext = candidate.bounding_box.extent
        cand = corners_2d(candidate._x, candidate._y,
                          math.radians(candidate._yaw_deg), ext.x, ext.y)
        for other in self._actors:
            if not other.is_alive or not other.is_solid:
                continue
            if abs(other._z - candidate._z) > (other.bounding_box.extent.z + ext.z):
                continue
            if _collision.obb_overlap(cand, _collision.actor_corners(other)):
                return True
        return False

    def _remove_actor(self, actor: Actor) -> None:
        """Destroying an actor takes its attached sensors with it, as in CARLA."""
        for sensor in list(self._actors):
            if getattr(sensor, "parent", None) is actor:
                sensor.is_alive = False
                self._actors.remove(sensor)
        try:
            self._actors.remove(actor)
        except ValueError:
            pass

    # -- the integrator ---------------------------------------------------

    def tick(self, seconds: Optional[float] = None) -> int:
        dt = float(self._settings.fixed_delta_seconds or 0.05)
        self.frame += 1
        self.elapsed_seconds += dt

        for actor in list(self._actors):
            if actor.is_alive:
                actor.step(dt)

        self.last_contacts = _collision.detect_contacts(self._actors)
        for contact in self.last_contacts:
            impulse = _collision.resolve(contact)
            if impulse <= 0.0:
                continue
            nx, ny = contact.normal
            vec = Vector3D(nx * impulse, ny * impulse, 0.0)
            self._notify_collision(contact.a, contact.b, vec)
            self._notify_collision(contact.b, contact.a,
                                   Vector3D(-vec.x, -vec.y, 0.0))

        snapshot = self.get_snapshot()
        for cb in list(self._on_tick.values()):
            cb(snapshot)
        return self.frame

    def _notify_collision(self, actor: Actor, other: Actor,
                          impulse: Vector3D) -> None:
        for sensor in self._actors:
            if isinstance(sensor, CollisionSensor) and sensor.parent is actor \
                    and sensor.is_alive:
                sensor.notify(self.frame, self.elapsed_seconds, other, impulse)

    # -- convenience ------------------------------------------------------

    def vehicles(self) -> List[Vehicle]:
        return [a for a in self._actors if isinstance(a, Vehicle) and a.is_alive]

    def props(self) -> List[StaticProp]:
        return [a for a in self._actors if isinstance(a, StaticProp) and a.is_alive]

    def __repr__(self) -> str:
        return (f"World(map={self._map.name!r}, actors={len(self._actors)}, "
                f"t={self.elapsed_seconds:.2f}s)")


class Client:
    """Stand-in for ``carla.Client``: there is no server, the world is local."""

    def __init__(self, host: str = "127.0.0.1", port: int = 2000,
                 worker_threads: int = 0, turn_preference: str = "straight",
                 fixed_delta_seconds: float = 0.05):
        self.host = host
        self.port = port
        self.turn_preference = turn_preference
        self.fixed_delta_seconds = fixed_delta_seconds
        self._world: Optional[World] = None
        self._timeout = 10.0

    def set_timeout(self, seconds: float) -> None:
        self._timeout = float(seconds)

    def get_client_version(self) -> str:
        return "osc2carla-localsim"

    get_server_version = get_client_version

    def get_world(self) -> World:
        if self._world is None:
            self._world = World(load_town(None, self.turn_preference),
                                self.fixed_delta_seconds)
        return self._world

    def load_world(self, map_name: str, *args, **kwargs) -> World:
        self._world = World(load_town(map_name, self.turn_preference),
                            self.fixed_delta_seconds)
        return self._world

    def reload_world(self, *args, **kwargs) -> World:
        name = self._world.get_map().name if self._world else None
        return self.load_world(name)

    def get_available_maps(self) -> List[str]:
        from .towns import town_names
        return town_names()

    def get_trafficmanager(self, *args, **kwargs):
        raise NotImplementedError(
            "the local simulator has no traffic manager; every actor is driven "
            "by the compiled behaviour tree or by --ego-policy")
