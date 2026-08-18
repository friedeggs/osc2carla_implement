"""Blueprint library for the local simulator.

Scenario files name CARLA blueprints (``vehicle.tesla.model3``,
``static.prop.dirtdebris01``) and ``ScenarioInitializer`` looks them up with
``world.get_blueprint_library().find(id)``.  There is no asset database here,
so a blueprint is just a dimension/mass record plus the attribute dictionary
the initializer writes ``role_name`` and ``color`` into.

Known models get their real CARLA extents so the bird's-eye view is to scale
and collision geometry is honest; anything else falls back to a generic car
of the right class.
"""
from __future__ import annotations

import fnmatch
from typing import Dict, List, Optional, Tuple

#: blueprint id -> (half_length, half_width, half_height, mass_kg, rgb)
VEHICLE_MODELS: Dict[str, Tuple[float, float, float, float, Tuple[int, int, int]]] = {
    "vehicle.tesla.model3":            (2.40, 1.08, 0.74, 1845.0, (220, 220, 225)),
    "vehicle.audi.tt":                 (2.09, 0.99, 0.70, 1400.0, (40, 120, 60)),
    "vehicle.audi.a2":                 (1.85, 0.89, 0.77, 1200.0, (60, 60, 60)),
    "vehicle.audi.etron":              (2.43, 1.01, 0.82, 2200.0, (70, 90, 130)),
    "vehicle.mercedes.coupe":          (2.51, 1.07, 0.81, 1900.0, (200, 200, 200)),
    "vehicle.dodge.charger_2020":      (2.50, 1.06, 0.75, 1900.0, (160, 30, 30)),
    "vehicle.dodge.charger_police":    (2.49, 1.06, 0.77, 1900.0, (30, 30, 90)),
    "vehicle.seat.leon":               (2.09, 0.90, 0.74, 1300.0, (30, 90, 160)),
    "vehicle.nissan.patrol":           (2.30, 0.94, 0.93, 2200.0, (80, 80, 80)),
    "vehicle.nissan.micra":            (1.83, 0.79, 0.75, 1100.0, (150, 150, 160)),
    "vehicle.citroen.c3":              (1.99, 0.93, 0.81, 1200.0, (0, 90, 160)),
    "vehicle.bmw.grandtourer":         (2.31, 1.12, 0.82, 1700.0, (60, 60, 70)),
    "vehicle.ford.mustang":            (2.36, 1.05, 0.65, 1700.0, (170, 40, 40)),
    "vehicle.lincoln.mkz_2020":        (2.45, 1.06, 0.75, 1900.0, (190, 190, 195)),
    "vehicle.carlamotors.european_hgv": (5.65, 1.30, 1.90, 12000.0, (40, 100, 60)),
    "vehicle.carlamotors.firetruck":   (4.23, 1.42, 1.68, 9000.0, (180, 30, 30)),
    "vehicle.mitsubishi.fusorosa":     (6.13, 1.34, 1.79, 8000.0, (220, 200, 60)),
    "vehicle.harley-davidson.low_rider": (1.18, 0.38, 0.64, 300.0, (60, 60, 60)),
    "vehicle.diamondback.century":     (0.82, 0.19, 0.55, 100.0, (30, 30, 30)),
}

DEFAULT_VEHICLE = (2.30, 1.00, 0.75, 1500.0, (170, 170, 175))

#: props are rendered as flat markers and never move
PROP_MODELS: Dict[str, Tuple[float, float, float]] = {
    "static.prop.dirtdebris01":   (0.60, 0.60, 0.02),
    "static.prop.trafficwarning": (1.20, 0.40, 0.90),
    "static.prop.streetbarrier":  (1.02, 0.20, 0.55),
    "static.prop.constructioncone": (0.25, 0.25, 0.50),
}
DEFAULT_PROP = (0.50, 0.50, 0.30)

SENSOR_IDS = ("sensor.other.collision",)


class ActorAttribute:
    __slots__ = ("id", "_value")

    def __init__(self, name: str, value):
        self.id = name
        self._value = value

    def __str__(self) -> str:
        return str(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def __float__(self) -> float:
        return float(self._value)

    def __bool__(self) -> bool:
        return str(self._value).lower() not in ("", "false", "0")

    def __repr__(self) -> str:
        return f"ActorAttribute({self.id}={self._value!r})"


class ActorBlueprint:
    """Mutable spawn recipe, as returned by ``BlueprintLibrary.find``."""

    def __init__(self, bp_id: str, attributes: Dict[str, str],
                 tags: Optional[List[str]] = None):
        self.id = bp_id
        self._attributes = dict(attributes)
        self.tags = list(tags or bp_id.split("."))

    def has_attribute(self, name: str) -> bool:
        return name in self._attributes

    def get_attribute(self, name: str) -> ActorAttribute:
        return ActorAttribute(name, self._attributes[name])

    def set_attribute(self, name: str, value) -> None:
        self._attributes[name] = str(value)

    @property
    def attributes(self) -> Dict[str, str]:
        return dict(self._attributes)

    def copy(self) -> "ActorBlueprint":
        return ActorBlueprint(self.id, self._attributes, self.tags)

    def __repr__(self) -> str:
        return f"ActorBlueprint(id={self.id!r})"


def _vehicle_blueprint(bp_id: str) -> ActorBlueprint:
    spec = VEHICLE_MODELS.get(bp_id, DEFAULT_VEHICLE)
    r, g, b = spec[4]
    return ActorBlueprint(bp_id, {
        "role_name": "autopilot",
        "color": f"{r},{g},{b}",
        "number_of_wheels": "2" if "harley" in bp_id or "diamondback" in bp_id else "4",
    })


def _prop_blueprint(bp_id: str) -> ActorBlueprint:
    return ActorBlueprint(bp_id, {"role_name": "prop"})


def _sensor_blueprint(bp_id: str) -> ActorBlueprint:
    return ActorBlueprint(bp_id, {"role_name": "sensor"})


class BlueprintLibrary:
    """``find`` raises ``IndexError`` for unknown ids, as CARLA's does."""

    def __init__(self):
        self._explicit = sorted(set(VEHICLE_MODELS) | set(PROP_MODELS)
                                | set(SENSOR_IDS))

    def find(self, bp_id: str) -> ActorBlueprint:
        bp_id = str(bp_id)
        if bp_id.startswith("vehicle."):
            return _vehicle_blueprint(bp_id)
        if bp_id.startswith("static."):
            return _prop_blueprint(bp_id)
        if bp_id in SENSOR_IDS:
            return _sensor_blueprint(bp_id)
        raise IndexError(
            f"blueprint {bp_id!r} is not available in the local simulator "
            f"(it provides vehicle.*, static.prop.* and sensor.other.collision)")

    def filter(self, pattern: str) -> List[ActorBlueprint]:
        return [self.find(i) for i in self._explicit
                if fnmatch.fnmatch(i, pattern)]

    def __iter__(self):
        return iter(self.filter("*"))

    def __len__(self) -> int:
        return len(self._explicit)


def vehicle_spec(bp_id: str):
    """``(half_len, half_wid, half_hgt, mass, default_rgb)`` for a vehicle id."""
    return VEHICLE_MODELS.get(bp_id, DEFAULT_VEHICLE)


def prop_spec(bp_id: str):
    """``(half_len, half_wid, half_hgt)`` for a static prop id."""
    return PROP_MODELS.get(bp_id, DEFAULT_PROP)
