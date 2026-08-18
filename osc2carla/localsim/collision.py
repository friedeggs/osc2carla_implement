"""Oriented-box overlap and the impulse response used on contact.

CARLA reports a collision event per physics substep for as long as two bodies
stay in contact, which is why ``metrics.py`` treats the event *count* as a
measure of how long a crash lasted rather than how many crashes there were.
This module keeps that behaviour: one event per contacting pair per tick.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .geometry import corners_2d

#: bounce-back fraction on impact; cars are close to inelastic
RESTITUTION = 0.12
#: fraction of the overlap resolved per tick when pushing bodies apart
SEPARATION = 0.8


class Contact:
    __slots__ = ("a", "b", "normal", "depth", "point")

    def __init__(self, a, b, normal: Tuple[float, float], depth: float,
                 point: Tuple[float, float]):
        self.a = a
        self.b = b
        self.normal = normal      # unit vector pointing from a to b
        self.depth = depth
        self.point = point

    def __repr__(self) -> str:
        return f"Contact({self.a.type_id} <-> {self.b.type_id}, depth={self.depth:.2f})"


def _axes(corners: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out = []
    for i in (0, 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy)
        if n > 1e-9:
            out.append((-dy / n, dx / n))
    return out


def _project(corners: Sequence[Tuple[float, float]],
             axis: Tuple[float, float]) -> Tuple[float, float]:
    dots = [c[0] * axis[0] + c[1] * axis[1] for c in corners]
    return min(dots), max(dots)


def obb_overlap(ca: Sequence[Tuple[float, float]],
                cb: Sequence[Tuple[float, float]]
                ) -> Optional[Tuple[Tuple[float, float], float]]:
    """Separating-axis test: ``(normal, depth)`` if the boxes overlap.

    The normal points from ``ca`` toward ``cb``.
    """
    best_axis: Optional[Tuple[float, float]] = None
    best_depth = float("inf")
    for axis in _axes(ca) + _axes(cb):
        amin, amax = _project(ca, axis)
        bmin, bmax = _project(cb, axis)
        if amax < bmin or bmax < amin:
            return None
        depth = min(amax, bmax) - max(amin, bmin)
        if depth < best_depth:
            best_depth = depth
            best_axis = axis
    if best_axis is None:
        return None
    # orient the normal from a to b
    ax = sum(c[0] for c in ca) / len(ca)
    ay = sum(c[1] for c in ca) / len(ca)
    bx = sum(c[0] for c in cb) / len(cb)
    by = sum(c[1] for c in cb) / len(cb)
    if (bx - ax) * best_axis[0] + (by - ay) * best_axis[1] < 0:
        best_axis = (-best_axis[0], -best_axis[1])
    return best_axis, best_depth


def actor_corners(actor) -> Tuple[Tuple[float, float], ...]:
    ext = actor.bounding_box.extent
    return corners_2d(actor._x, actor._y, math.radians(actor._yaw_deg),
                      ext.x, ext.y)


def detect_contacts(actors: Sequence) -> List[Contact]:
    """All pairwise overlaps among solid actors, this tick."""
    solid = [a for a in actors if getattr(a, "is_solid", False) and a.is_alive]
    out: List[Contact] = []
    for i in range(len(solid)):
        a = solid[i]
        ca = actor_corners(a)
        for j in range(i + 1, len(solid)):
            b = solid[j]
            # cheap radius reject before the full SAT
            dx, dy = b._x - a._x, b._y - a._y
            if dx * dx + dy * dy > (a.collision_radius + b.collision_radius) ** 2:
                continue
            if abs(a._z - b._z) > (a.bounding_box.extent.z + b.bounding_box.extent.z):
                continue                       # separated vertically
            hit = obb_overlap(ca, actor_corners(b))
            if hit is None:
                continue
            normal, depth = hit
            point = ((a._x + b._x) * 0.5, (a._y + b._y) * 0.5)
            out.append(Contact(a, b, normal, depth, point))
    return out


def resolve(contact: Contact) -> float:
    """Push the pair apart, exchange momentum, return the impulse magnitude.

    Returns the scalar impulse in kg m/s, the same quantity CARLA reports as
    ``event.normal_impulse``.
    """
    a, b = contact.a, contact.b
    nx, ny = contact.normal
    inv_a = 0.0 if a.mass <= 0 or a.immovable else 1.0 / a.mass
    inv_b = 0.0 if b.mass <= 0 or b.immovable else 1.0 / b.mass
    inv_sum = inv_a + inv_b
    if inv_sum <= 0.0:
        return 0.0

    # positional correction so the boxes stop interpenetrating
    push = contact.depth * SEPARATION / inv_sum
    a._x -= nx * push * inv_a
    a._y -= ny * push * inv_a
    b._x += nx * push * inv_b
    b._y += ny * push * inv_b

    rel = ((b._vx - a._vx) * nx) + ((b._vy - a._vy) * ny)
    if rel > 0.0:
        return 0.0                                  # already separating
    j = -(1.0 + RESTITUTION) * rel / inv_sum
    a._vx -= j * nx * inv_a
    a._vy -= j * ny * inv_a
    b._vx += j * nx * inv_b
    b._vy += j * ny * inv_b

    # a crude yaw kick: off-centre hits spin the struck car
    for actor, sign in ((a, -1.0), (b, +1.0)):
        if actor.immovable:
            continue
        rx = contact.point[0] - actor._x
        ry = contact.point[1] - actor._y
        torque = rx * ny - ry * nx
        inertia = max(actor.mass * (actor.bounding_box.extent.x ** 2), 1.0)
        actor._spin += sign * j * torque / inertia
        actor._spin = max(-3.0, min(3.0, actor._spin))
    return abs(j)
