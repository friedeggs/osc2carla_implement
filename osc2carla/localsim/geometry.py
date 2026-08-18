"""Vector / transform types, shaped like the ones CARLA's Python API exposes.

The backend modules (``atomic_behaviors``, ``initializer``, ``policy``,
``metrics``) are written against ``carla.Location`` / ``carla.Rotation`` /
``carla.Transform``.  Re-declaring the same surface here is what lets those
modules run unchanged against the local simulator: they never learn which
backend they are talking to.

Conventions follow CARLA so the scenario files keep their meaning:

* ``x`` grows east, ``y`` grows *south*, ``z`` grows up (left-handed).
* ``yaw`` is in **degrees**, measured from +x toward +y, so the forward
  vector is ``(cos yaw, sin yaw)`` and a left turn *decreases* yaw when the
  bird's-eye view is drawn with +x right and +y down.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple


class Vector3D:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def squared_length(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def __add__(self, o: "Vector3D") -> "Vector3D":
        return Vector3D(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vector3D") -> "Vector3D":
        return Vector3D(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, k: float) -> "Vector3D":
        return Vector3D(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"Vector3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class Location(Vector3D):
    """A point in world space."""

    def distance(self, other: "Location") -> float:
        return math.sqrt((self.x - other.x) ** 2
                         + (self.y - other.y) ** 2
                         + (self.z - other.z) ** 2)

    def __repr__(self) -> str:
        return f"Location(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"


class Rotation:
    """Euler angles in degrees; only ``yaw`` affects the planar simulation."""

    __slots__ = ("pitch", "yaw", "roll")

    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

    def get_forward_vector(self) -> Vector3D:
        r = math.radians(self.yaw)
        p = math.radians(self.pitch)
        return Vector3D(math.cos(r) * math.cos(p), math.sin(r) * math.cos(p),
                        math.sin(p))

    def get_right_vector(self) -> Vector3D:
        r = math.radians(self.yaw)
        return Vector3D(-math.sin(r), math.cos(r), 0.0)

    def __repr__(self) -> str:
        return (f"Rotation(pitch={self.pitch:.3f}, yaw={self.yaw:.3f}, "
                f"roll={self.roll:.3f})")


class Transform:
    __slots__ = ("location", "rotation")

    def __init__(self, location: Location = None, rotation: Rotation = None):
        self.location = location if location is not None else Location()
        self.rotation = rotation if rotation is not None else Rotation()

    def get_forward_vector(self) -> Vector3D:
        return self.rotation.get_forward_vector()

    def get_right_vector(self) -> Vector3D:
        return self.rotation.get_right_vector()

    def transform(self, in_point: Location) -> Location:
        """In-place CARLA-style local -> world transform of ``in_point``."""
        r = math.radians(self.rotation.yaw)
        c, s = math.cos(r), math.sin(r)
        x = in_point.x * c - in_point.y * s + self.location.x
        y = in_point.x * s + in_point.y * c + self.location.y
        in_point.x, in_point.y = x, y
        in_point.z += self.location.z
        return in_point

    def copy(self) -> "Transform":
        return Transform(Location(self.location.x, self.location.y, self.location.z),
                         Rotation(self.rotation.pitch, self.rotation.yaw,
                                  self.rotation.roll))

    def __repr__(self) -> str:
        return f"Transform({self.location!r}, {self.rotation!r})"


class BoundingBox:
    """Axis-aligned half-extents in the actor's own frame, as in CARLA."""

    __slots__ = ("location", "extent", "rotation")

    def __init__(self, location: Location = None, extent: Vector3D = None,
                 rotation: Rotation = None):
        self.location = location if location is not None else Location()
        self.extent = extent if extent is not None else Vector3D(2.4, 1.0, 0.75)
        self.rotation = rotation if rotation is not None else Rotation()


# --------------------------------------------------------------------------
# planar helpers used by the road map, physics and renderer
# --------------------------------------------------------------------------

def normalise_angle(a: float) -> float:
    """Wrap radians to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def corners_2d(cx: float, cy: float, yaw_rad: float,
               half_len: float, half_wid: float) -> Tuple[Tuple[float, float], ...]:
    """The four world-space corners of an oriented box, counter-clockwise."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    out = []
    for dx, dy in ((half_len, half_wid), (-half_len, half_wid),
                   (-half_len, -half_wid), (half_len, -half_wid)):
        out.append((cx + dx * c - dy * s, cy + dx * s + dy * c))
    return tuple(out)


def polyline_length(points: Iterable[Tuple[float, float]]) -> float:
    pts = list(points)
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
