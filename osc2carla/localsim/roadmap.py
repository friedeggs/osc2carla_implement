"""A lane-graph road map with the slice of CARLA's map API the backend uses.

``atomic_behaviors``, ``initializer`` and ``policy`` only ever ask a map for
five things::

    map.get_waypoint(location, project_to_road=True)
    map.get_spawn_points() / map.generate_waypoints(d)
    waypoint.next(d) / waypoint.previous(d)
    waypoint.get_left_lane() / waypoint.get_right_lane()
    waypoint.transform / .road_id / .lane_id / .s

so that is what this implements.  A map is a set of :class:`Lane` polylines
wired into a directed graph by successor / predecessor links, plus left/right
neighbour links inside a carriageway.  Junction arms are ordinary lanes
flagged ``is_junction``.

Where CARLA returns the successors of a lane in whatever order the OpenDRIVE
file happens to list them, this map classifies each successor as a straight /
left / right manoeuvre and orders them by :attr:`Map.turn_preference`.  The
compiled ``drive()`` behaviour takes ``next(d)[0]``, so that preference is what
decides which way a vehicle goes through a junction.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import Location, Rotation, Transform, normalise_angle

#: |turn angle| below this is a straight-ahead manoeuvre.
STRAIGHT_TOLERANCE = 0.35  # radians

_TURN_ORDER = {
    "straight": ("straight", "right", "left"),
    "right": ("right", "straight", "left"),
    "left": ("left", "straight", "right"),
}


class Lane:
    """One directed lane, described by its centreline polyline."""

    __slots__ = ("uid", "road_id", "lane_id", "points", "width", "is_junction",
                 "_cum", "length", "successors", "predecessors", "left", "right",
                 "name")

    def __init__(self, uid: int, road_id: int, lane_id: int,
                 points: Sequence[Tuple[float, float]], width: float = 3.5,
                 is_junction: bool = False, name: str = ""):
        if len(points) < 2:
            raise ValueError("a lane needs at least two points")
        self.uid = uid
        self.road_id = road_id
        self.lane_id = lane_id
        self.points: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in points]
        self.width = float(width)
        self.is_junction = bool(is_junction)
        self.name = name
        cum = [0.0]
        for i in range(len(self.points) - 1):
            cum.append(cum[-1] + math.dist(self.points[i], self.points[i + 1]))
        self._cum = cum
        self.length = cum[-1]
        #: (lane uid, manoeuvre) pairs, both ways round
        self.successors: List[Tuple[int, str]] = []
        self.predecessors: List[Tuple[int, str]] = []
        self.left: Optional[int] = None
        self.right: Optional[int] = None

    # -- geometry ---------------------------------------------------------

    def pose_at(self, s: float) -> Tuple[float, float, float]:
        """``(x, y, heading_radians)`` at arc length ``s`` along the lane."""
        s = min(max(s, 0.0), self.length)
        i = self._segment_index(s)
        (x0, y0), (x1, y1) = self.points[i], self.points[i + 1]
        seg_len = self._cum[i + 1] - self._cum[i]
        t = 0.0 if seg_len <= 1e-9 else (s - self._cum[i]) / seg_len
        return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, math.atan2(y1 - y0, x1 - x0))

    def _segment_index(self, s: float) -> int:
        lo, hi = 0, len(self._cum) - 2
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._cum[mid] <= s:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def project(self, x: float, y: float) -> Tuple[float, float]:
        """Closest point on the centreline: ``(s, lateral_distance)``."""
        best_s, best_d2 = 0.0, float("inf")
        for i in range(len(self.points) - 1):
            (x0, y0), (x1, y1) = self.points[i], self.points[i + 1]
            dx, dy = x1 - x0, y1 - y0
            den = dx * dx + dy * dy
            t = 0.0 if den <= 1e-12 else ((x - x0) * dx + (y - y0) * dy) / den
            t = min(1.0, max(0.0, t))
            px, py = x0 + dx * t, y0 + dy * t
            d2 = (x - px) ** 2 + (y - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = self._cum[i] + t * math.sqrt(den)
        return best_s, math.sqrt(best_d2)

    @property
    def start_heading(self) -> float:
        return self.pose_at(0.0)[2]

    @property
    def end_heading(self) -> float:
        return self.pose_at(self.length)[2]

    def __repr__(self) -> str:
        return (f"Lane(uid={self.uid}, road={self.road_id}, lane={self.lane_id}, "
                f"len={self.length:.1f}{', junction' if self.is_junction else ''})")


class Waypoint:
    """A pose on a lane. Same read-only surface as ``carla.Waypoint``."""

    __slots__ = ("_map", "_lane", "s", "transform")

    def __init__(self, road_map: "Map", lane: Lane, s: float):
        self._map = road_map
        self._lane = lane
        self.s = min(max(float(s), 0.0), lane.length)
        x, y, h = lane.pose_at(self.s)
        self.transform = Transform(Location(x, y, 0.0),
                                   Rotation(yaw=math.degrees(h)))

    # -- identity ---------------------------------------------------------

    @property
    def id(self) -> int:
        return self._lane.uid * 100003 + int(self.s * 10)

    @property
    def road_id(self) -> int:
        return self._lane.road_id

    @property
    def lane_id(self) -> int:
        return self._lane.lane_id

    @property
    def lane_width(self) -> float:
        return self._lane.width

    @property
    def is_junction(self) -> bool:
        return self._lane.is_junction

    @property
    def lane(self) -> Lane:
        return self._lane

    # -- traversal --------------------------------------------------------

    def next(self, distance: float) -> List["Waypoint"]:
        """Waypoints ``distance`` metres further along, one per continuation.

        Ordered by :attr:`Map.turn_preference`, so ``next(d)[0]`` is the
        manoeuvre a cruising vehicle takes at the next junction.
        """
        return self._walk(distance, forward=True)

    def previous(self, distance: float) -> List["Waypoint"]:
        return self._walk(distance, forward=False)

    def _walk(self, distance: float, forward: bool, depth: int = 0) -> List["Waypoint"]:
        distance = float(distance)
        if distance < 0:
            return []
        lane = self._lane
        target = self.s + distance if forward else self.s - distance
        if forward and target <= lane.length:
            return [Waypoint(self._map, lane, target)]
        if not forward and target >= 0.0:
            return [Waypoint(self._map, lane, target)]
        if depth >= self._map.max_walk_depth:
            return []
        remaining = target - lane.length if forward else -target
        out: List[Waypoint] = []
        links = [uid for uid, _ in (self._map.ordered_successors(lane) if forward
                                    else self._map.ordered_predecessors(lane))]
        for uid in links:
            nxt = self._map.lanes.get(uid)
            if nxt is None:
                continue
            start_s = 0.0 if forward else nxt.length
            hop = Waypoint(self._map, nxt, start_s)
            out.extend(hop._walk(remaining, forward, depth + 1))
            if len(out) >= self._map.max_branches:
                break
        return out

    # -- neighbours -------------------------------------------------------

    def get_left_lane(self) -> Optional["Waypoint"]:
        return self._neighbour(self._lane.left)

    def get_right_lane(self) -> Optional["Waypoint"]:
        return self._neighbour(self._lane.right)

    def _neighbour(self, uid: Optional[int]) -> Optional["Waypoint"]:
        if uid is None:
            return None
        lane = self._map.lanes.get(uid)
        if lane is None:
            return None
        loc = self.transform.location
        s, _ = lane.project(loc.x, loc.y)
        return Waypoint(self._map, lane, s)

    def __repr__(self) -> str:
        loc = self.transform.location
        return (f"Waypoint(road={self.road_id}, lane={self.lane_id}, "
                f"s={self.s:.1f}, x={loc.x:.1f}, y={loc.y:.1f})")


class Map:
    """A lane graph plus a uniform-grid index for nearest-lane queries."""

    #: how many junction hops a single next()/previous() call may cross
    max_walk_depth = 6
    #: cap on the branches a single next() call returns
    max_branches = 8

    def __init__(self, name: str, lanes: Sequence[Lane],
                 spawn_points: Optional[Sequence[Transform]] = None,
                 turn_preference: str = "straight",
                 index_cell: float = 8.0, index_step: float = 1.0):
        self.name = name
        self.lanes: Dict[int, Lane] = {lane.uid: lane for lane in lanes}
        self.turn_preference = turn_preference
        self._cell = float(index_cell)
        self._index: Dict[Tuple[int, int], List[int]] = {}
        self._build_index(index_step)
        self._spawn_points: List[Transform] = list(spawn_points or [])
        if not self._spawn_points:
            self._spawn_points = self._default_spawn_points()
        self._bounds = self._compute_bounds()

    # -- construction -----------------------------------------------------

    def _build_index(self, step: float) -> None:
        for lane in self.lanes.values():
            s = 0.0
            while True:
                x, y, _ = lane.pose_at(s)
                self._index.setdefault(self._cell_of(x, y), []).append(lane.uid)
                if s >= lane.length:
                    break
                s = min(s + step, lane.length)
        for key, uids in self._index.items():
            self._index[key] = sorted(set(uids))

    def _cell_of(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self._cell)), int(math.floor(y / self._cell)))

    def _compute_bounds(self) -> Tuple[float, float, float, float]:
        xs, ys = [], []
        for lane in self.lanes.values():
            for x, y in lane.points:
                xs.append(x)
                ys.append(y)
        if not xs:
            return (0.0, 0.0, 1.0, 1.0)
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(min_x, min_y, max_x, max_y)`` over every lane point."""
        return self._bounds

    def _default_spawn_points(self, spacing: float = 18.0) -> List[Transform]:
        out: List[Transform] = []
        for uid in sorted(self.lanes):
            lane = self.lanes[uid]
            if lane.is_junction or lane.length < spacing:
                continue
            s = spacing * 0.5
            while s < lane.length - 2.0:
                x, y, h = lane.pose_at(s)
                out.append(Transform(Location(x, y, 0.5),
                                     Rotation(yaw=math.degrees(h))))
                s += spacing
        return out

    # -- CARLA-shaped API -------------------------------------------------

    def get_waypoint(self, location, project_to_road: bool = True,
                     lane_type=None) -> Optional[Waypoint]:
        x, y = location.x, location.y
        best: Optional[Tuple[float, Lane, float]] = None
        for radius in (1, 2, 4, 8, 16):
            candidates = self._candidates(x, y, radius)
            for uid in candidates:
                lane = self.lanes[uid]
                s, d = lane.project(x, y)
                if best is None or d < best[0]:
                    best = (d, lane, s)
            if best is not None:
                break
        if best is None:
            return None
        d, lane, s = best
        if not project_to_road and d > lane.width * 0.5:
            return None
        return Waypoint(self, lane, s)

    def _candidates(self, x: float, y: float, radius: int) -> List[int]:
        cx, cy = self._cell_of(x, y)
        out: List[int] = []
        for i in range(cx - radius, cx + radius + 1):
            for j in range(cy - radius, cy + radius + 1):
                out.extend(self._index.get((i, j), ()))
        return sorted(set(out))

    def get_spawn_points(self) -> List[Transform]:
        return [t.copy() for t in self._spawn_points]

    def generate_waypoints(self, distance: float) -> List[Waypoint]:
        out: List[Waypoint] = []
        for uid in sorted(self.lanes):
            lane = self.lanes[uid]
            s = 0.0
            while s <= lane.length:
                out.append(Waypoint(self, lane, s))
                s += max(distance, 0.5)
        return out

    def get_topology(self) -> List[Tuple[Waypoint, Waypoint]]:
        out = []
        for lane in self.lanes.values():
            out.append((Waypoint(self, lane, 0.0), Waypoint(self, lane, lane.length)))
        return out

    # -- successor ordering ----------------------------------------------

    def ordered_successors(self, lane: Lane) -> List[Tuple[int, str]]:
        return self._ordered(lane.successors)

    def ordered_predecessors(self, lane: Lane) -> List[Tuple[int, str]]:
        """Same preference walking backwards.

        ``previous()`` is what ``position(behind: other)`` resolves against,
        so an ambiguous approach has to prefer the straight one or a vehicle
        placed 16 m behind another ends up part-way round a turn.
        """
        return self._ordered(lane.predecessors)

    def _ordered(self, links: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
        order = _TURN_ORDER.get(self.turn_preference, _TURN_ORDER["straight"])
        rank = {kind: i for i, kind in enumerate(order)}
        return sorted(links, key=lambda item: (rank.get(item[1], 9), item[0]))

    # -- graph wiring, used by the town builders --------------------------

    def link(self, from_uid: int, to_uid: int,
             turn: Optional[str] = None) -> None:
        """Make ``to_uid`` a successor of ``from_uid``.

        Pass ``turn`` when the link is a junction connector: a connector
        starts tangent to the lane feeding it, so its own start heading says
        nothing about the manoeuvre it performs.
        """
        a, b = self.lanes[from_uid], self.lanes[to_uid]
        if turn is None:
            turn = classify_turn(a.end_heading, b.start_heading)
        if (to_uid, turn) not in a.successors:
            a.successors.append((to_uid, turn))
        if (from_uid, turn) not in b.predecessors:
            b.predecessors.append((from_uid, turn))

    def __repr__(self) -> str:
        return f"Map(name={self.name!r}, lanes={len(self.lanes)})"


def classify_turn(heading_in: float, heading_out: float) -> str:
    """``straight`` / ``left`` / ``right`` for a change of heading.

    In CARLA's left-handed frame a left turn *decreases* yaw, so a negative
    delta is a left turn.
    """
    delta = normalise_angle(heading_out - heading_in)
    if abs(delta) < STRAIGHT_TOLERANCE:
        return "straight"
    return "left" if delta < 0 else "right"
