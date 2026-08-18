"""Control, weather, settings and sensor-event types mirroring CARLA's."""
from __future__ import annotations

from typing import Any, Optional

from .geometry import Vector3D


class VehicleControl:
    """Normalised actuation, identical in meaning to ``carla.VehicleControl``."""

    __slots__ = ("throttle", "steer", "brake", "hand_brake", "reverse",
                 "manual_gear_shift", "gear")

    def __init__(self, throttle: float = 0.0, steer: float = 0.0,
                 brake: float = 0.0, hand_brake: bool = False,
                 reverse: bool = False, manual_gear_shift: bool = False,
                 gear: int = 0):
        self.throttle = float(throttle)
        self.steer = float(steer)
        self.brake = float(brake)
        self.hand_brake = bool(hand_brake)
        self.reverse = bool(reverse)
        self.manual_gear_shift = bool(manual_gear_shift)
        self.gear = int(gear)

    def __repr__(self) -> str:
        return (f"VehicleControl(throttle={self.throttle:.2f}, "
                f"steer={self.steer:+.2f}, brake={self.brake:.2f})")


class VehicleLightState:
    """Bit flags, same names and roles as ``carla.VehicleLightState``.

    Instances are constructed from an int (``VehicleLightState(int(flag))``)
    the way ``atomic_behaviors.SetLights`` does it.
    """

    NONE = 0
    Position = 1
    LowBeam = 2
    HighBeam = 4
    Brake = 8
    RightBlinker = 16
    LeftBlinker = 32
    Reverse = 64
    Fog = 128
    Interior = 256
    Special1 = 512
    Special2 = 1024
    All = 0xFFFFFFFF

    __slots__ = ("value",)

    def __init__(self, value: int = 0):
        self.value = int(value)

    def __int__(self) -> int:
        return self.value

    def __or__(self, other) -> "VehicleLightState":
        return VehicleLightState(self.value | int(other))

    def __and__(self, other) -> int:
        return self.value & int(other)

    def __eq__(self, other) -> bool:
        return int(self) == int(other) if isinstance(other, (int, VehicleLightState)) \
            else NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"VehicleLightState({self.value})"


class WeatherParameters:
    """Only the sun angles are simulated; the renderer tints the scene by them."""

    def __init__(self, cloudiness: float = 0.0, precipitation: float = 0.0,
                 sun_azimuth_angle: float = 0.0, sun_altitude_angle: float = 45.0,
                 fog_density: float = 0.0):
        self.cloudiness = float(cloudiness)
        self.precipitation = float(precipitation)
        self.sun_azimuth_angle = float(sun_azimuth_angle)
        self.sun_altitude_angle = float(sun_altitude_angle)
        self.fog_density = float(fog_density)

    def copy(self) -> "WeatherParameters":
        return WeatherParameters(self.cloudiness, self.precipitation,
                                 self.sun_azimuth_angle, self.sun_altitude_angle,
                                 self.fog_density)

    def __repr__(self) -> str:
        return (f"WeatherParameters(sun_azimuth_angle={self.sun_azimuth_angle:.1f}, "
                f"sun_altitude_angle={self.sun_altitude_angle:.1f})")


class WorldSettings:
    """``synchronous_mode`` is always effectively on: the world only advances
    when :meth:`World.tick` is called."""

    def __init__(self, synchronous_mode: bool = True,
                 fixed_delta_seconds: Optional[float] = 0.05,
                 no_rendering_mode: bool = False):
        self.synchronous_mode = bool(synchronous_mode)
        self.fixed_delta_seconds = fixed_delta_seconds
        self.no_rendering_mode = bool(no_rendering_mode)

    def copy(self) -> "WorldSettings":
        return WorldSettings(self.synchronous_mode, self.fixed_delta_seconds,
                             self.no_rendering_mode)


class SensorData:
    __slots__ = ("frame", "timestamp")

    def __init__(self, frame: int, timestamp: float):
        self.frame = int(frame)
        self.timestamp = float(timestamp)


class CollisionEvent(SensorData):
    """What ``sensor.other.collision`` delivers to its callback."""

    __slots__ = ("actor", "other_actor", "normal_impulse")

    def __init__(self, frame: int, timestamp: float, actor: Any,
                 other_actor: Any, normal_impulse: Vector3D):
        super().__init__(frame, timestamp)
        self.actor = actor
        self.other_actor = other_actor
        self.normal_impulse = normal_impulse

    def __repr__(self) -> str:
        return (f"CollisionEvent(frame={self.frame}, "
                f"other={getattr(self.other_actor, 'type_id', '?')}, "
                f"impulse={self.normal_impulse.length():.0f})")


class Timestamp:
    __slots__ = ("frame", "elapsed_seconds", "delta_seconds")

    def __init__(self, frame: int, elapsed_seconds: float, delta_seconds: float):
        self.frame = int(frame)
        self.elapsed_seconds = float(elapsed_seconds)
        self.delta_seconds = float(delta_seconds)


class WorldSnapshot:
    __slots__ = ("frame", "timestamp")

    def __init__(self, frame: int, timestamp: Timestamp):
        self.frame = frame
        self.timestamp = timestamp


def parse_color(text: str, default=(200, 200, 200)) -> tuple:
    """``"0,128,0"`` -> ``(0, 128, 0)``; anything unparseable -> ``default``."""
    try:
        parts = [int(p) for p in str(text).split(",")]
    except (TypeError, ValueError):
        return default
    if len(parts) != 3:
        return default
    return tuple(max(0, min(255, p)) for p in parts)


__all__ = [
    "CollisionEvent", "SensorData", "Timestamp", "VehicleControl",
    "VehicleLightState", "WeatherParameters", "WorldSettings", "WorldSnapshot",
    "parse_color",
]
