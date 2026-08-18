"""The names ``import carla`` would provide, backed by the local simulator.

Backend modules reach the active simulator through
``osc2carla.backend.simapi``; binding the ``pygame`` backend points that
indirection at this module.  Everything here is the local implementation --
nothing in this package imports or requires CARLA.
"""
from __future__ import annotations

from .actors import (Actor, ActorList, CollisionSensor, Sensor, StaticProp,
                     Vehicle)
from .blueprints import ActorBlueprint, BlueprintLibrary
from .datatypes import (CollisionEvent, Timestamp, VehicleControl,
                        VehicleLightState, WeatherParameters, WorldSettings,
                        WorldSnapshot)
from .geometry import BoundingBox, Location, Rotation, Transform, Vector3D
from .roadmap import Map, Waypoint
from .world import Client, World

#: marks this namespace as the local stand-in rather than the real thing
is_localsim = True
__version__ = "0.1.0"

__all__ = [
    "Actor", "ActorBlueprint", "ActorList", "BlueprintLibrary", "BoundingBox",
    "Client", "CollisionEvent", "CollisionSensor", "Location", "Map",
    "Rotation", "Sensor", "StaticProp", "Timestamp", "Transform", "Vector3D",
    "Vehicle", "VehicleControl", "VehicleLightState", "WeatherParameters",
    "World", "WorldSettings", "WorldSnapshot", "Waypoint", "is_localsim",
]
