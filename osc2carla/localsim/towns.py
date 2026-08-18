"""Built-in road networks for the local simulator.

CARLA towns are OpenDRIVE assets that ship with the simulator binary; without
CARLA there is nothing to load, so the local backend synthesises its networks
from a compact description instead.  A :class:`GridTown` is a rectangular
lattice of two-way roads joined by junctions, which is enough for the
manoeuvres this compiler emits: cruise, follow, change lane, cross a junction,
ram another actor.

Scenario files name a CARLA town in ``keep(it.map_file == ...)``.  Those names
are aliased onto the default grid here, so an unmodified ``.osc`` file loads.
What it does *not* get is CARLA's geometry: a scenario that hard-codes
Town10HD_Opt coordinates (the files under ``scenarios/benchmark/``) will spawn
its actors wherever those coordinates land in the grid, and the conflict it
stages will not reproduce.  See ``--town`` and the README section on the local
backend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import normalise_angle
from .roadmap import Junction, Lane, Map, classify_turn

#: sampled points per junction connector
CONNECTOR_SAMPLES = 14


def _link(a: Lane, b: Lane, turn: Optional[str] = None) -> None:
    """Make ``b`` a successor of ``a``, tagged with the manoeuvre it is.

    ``turn`` has to be passed for a junction connector: a connector leaves
    tangent to the lane feeding it, so classifying it from its own start
    heading would call every one of them "straight".  The manoeuvre is a
    property of where the connector *ends up*.
    """
    if turn is None:
        turn = classify_turn(a.end_heading, b.start_heading)
    if (b.uid, turn) not in a.successors:
        a.successors.append((b.uid, turn))
    if (a.uid, turn) not in b.predecessors:
        b.predecessors.append((a.uid, turn))


def _bezier(p0: Tuple[float, float], h0: float,
            p1: Tuple[float, float], h1: float,
            samples: int = CONNECTOR_SAMPLES) -> List[Tuple[float, float]]:
    """Cubic Bezier joining two poses, tangent to both headings."""
    d = math.dist(p0, p1) * 0.55
    c0 = (p0[0] + math.cos(h0) * d, p0[1] + math.sin(h0) * d)
    c1 = (p1[0] - math.cos(h1) * d, p1[1] - math.sin(h1) * d)
    pts = []
    for i in range(samples + 1):
        t = i / samples
        u = 1.0 - t
        x = (u ** 3 * p0[0] + 3 * u * u * t * c0[0]
             + 3 * u * t * t * c1[0] + t ** 3 * p1[0])
        y = (u ** 3 * p0[1] + 3 * u * u * t * c0[1]
             + 3 * u * t * t * c1[1] + t ** 3 * p1[1])
        pts.append((x, y))
    return pts


@dataclass
class GridTown:
    """A lattice of two-way roads.

    ``xs`` / ``ys`` are the centrelines of the north-south and east-west
    roads; a junction sits at every crossing.  Road segments exist only
    *between* junctions, so the network is closed: a vehicle that reaches the
    edge turns rather than driving off the map.
    """

    name: str
    xs: Sequence[float]
    ys: Sequence[float]
    lanes_per_dir: int = 2
    lane_width: float = 3.5
    #: Clear area beyond the bare crossing of the two carriageways.
    #:
    #: Without it a junction box is exactly as wide as the roads, which puts
    #: the outermost lane 1.75 m from the box edge and makes a right-turn
    #: connector a ~1.75 m arc -- tighter than the bicycle model's ~4.5 m
    #: minimum turning radius. The controller then cuts the corner into the
    #: neighbouring lane instead of tracking the connector. Real intersections
    #: have the same clear area, for the same reason.
    junction_margin: float = 4.0
    description: str = ""

    #: filled in during build()
    _lanes: List[Lane] = field(default_factory=list, repr=False)
    _next_uid: int = field(default=0, repr=False)
    _next_road: int = field(default=0, repr=False)
    # lane uid -> (junction key it starts at, junction key it ends at)
    _ends: Dict[int, Tuple[Optional[tuple], Optional[tuple]]] = \
        field(default_factory=dict, repr=False)

    @property
    def carriageway_half_width(self) -> float:
        return self.lanes_per_dir * self.lane_width

    @property
    def half_road(self) -> float:
        """Half-size of a junction box: the carriageway plus its clear area."""
        return self.carriageway_half_width + self.junction_margin

    # -- building blocks --------------------------------------------------

    def _add_lane(self, road_id: int, lane_id: int,
                  points: Sequence[Tuple[float, float]], is_junction: bool,
                  name: str, start_j=None, end_j=None) -> Lane:
        lane = Lane(self._next_uid, road_id, lane_id, points,
                    width=self.lane_width, is_junction=is_junction, name=name)
        self._next_uid += 1
        self._lanes.append(lane)
        self._ends[lane.uid] = (start_j, end_j)
        return lane

    def _carriageway(self, road_id: int, axis: str, fixed: float,
                     lo: float, hi: float, start_j, end_j) -> None:
        """Both directions of one road segment.

        The reference direction is +x for an east-west road and +y for a
        north-south one.  CARLA numbers lanes on the right of that reference
        direction negative, innermost first, which is reproduced here.
        """
        w = self.lane_width
        forward: List[Lane] = []
        backward: List[Lane] = []
        for i in range(self.lanes_per_dir):
            off = (i + 0.5) * w
            if axis == "x":
                # +x direction: right-hand side is +y
                fwd = self._add_lane(road_id, -(i + 1),
                                     [(lo, fixed + off), (hi, fixed + off)],
                                     False, f"{road_id}:E{i}", start_j, end_j)
                bwd = self._add_lane(road_id, +(i + 1),
                                     [(hi, fixed - off), (lo, fixed - off)],
                                     False, f"{road_id}:W{i}", end_j, start_j)
            else:
                # +y direction: right-hand side is -x
                fwd = self._add_lane(road_id, -(i + 1),
                                     [(fixed - off, lo), (fixed - off, hi)],
                                     False, f"{road_id}:S{i}", start_j, end_j)
                bwd = self._add_lane(road_id, +(i + 1),
                                     [(fixed + off, hi), (fixed + off, lo)],
                                     False, f"{road_id}:N{i}", end_j, start_j)
            forward.append(fwd)
            backward.append(bwd)
        for group in (forward, backward):
            for i, lane in enumerate(group):
                lane.left = group[i - 1].uid if i > 0 else None
                lane.right = group[i + 1].uid if i + 1 < len(group) else None

    # -- assembly ---------------------------------------------------------

    def build(self, turn_preference: str = "straight") -> Map:
        """Lay out the road segments, join them at every crossing, index."""
        self._lanes, self._ends = [], {}
        self._next_uid = self._next_road = 0
        h = self.half_road
        xs, ys = list(self.xs), list(self.ys)

        for yi, y in enumerate(ys):
            for xi in range(len(xs) - 1):
                self._next_road += 1
                self._carriageway(self._next_road, "x", y,
                                  xs[xi] + h, xs[xi + 1] - h,
                                  ("j", xi, yi), ("j", xi + 1, yi))
        for xi, x in enumerate(xs):
            for yi in range(len(ys) - 1):
                self._next_road += 1
                self._carriageway(self._next_road, "y", x,
                                  ys[yi] + h, ys[yi + 1] - h,
                                  ("j", xi, yi), ("j", xi, yi + 1))

        self._connect_junctions()
        junctions = [
            Junction(len(ys) * xi + yi, (x, y), (h, h))
            for xi, x in enumerate(xs) for yi, y in enumerate(ys)
        ]
        return Map(self.name, self._lanes, turn_preference=turn_preference,
                   junctions=junctions)

    def _connect_junctions(self) -> None:
        """Add one connector lane per legal in-lane/out-lane pair."""
        incoming: Dict[tuple, List[Lane]] = {}
        outgoing: Dict[tuple, List[Lane]] = {}
        for lane in list(self._lanes):
            start_j, end_j = self._ends.get(lane.uid, (None, None))
            if end_j is not None:
                incoming.setdefault(end_j, []).append(lane)
            if start_j is not None:
                outgoing.setdefault(start_j, []).append(lane)

        for key in sorted(set(incoming) | set(outgoing)):
            ins, outs = incoming.get(key, []), outgoing.get(key, [])
            if not ins or not outs:
                continue
            self._next_road += 1
            junction_road = self._next_road
            for a in ins:
                targets = [b for b in outs if self._pairing(a, b)]
                if not targets:
                    # A junction arm this lane index has no manoeuvre into --
                    # the inner lane at a corner of the grid, for instance.
                    # Leaving it unconnected would make next() return [] and
                    # the vehicle would drive straight off the network, so
                    # fall back to the nearest-indexed legal exit.
                    targets = self._fallback_targets(a, outs)
                for b in targets:
                    self._add_connector(junction_road, a, b)

    def _pairing(self, a: Lane, b: Lane) -> bool:
        delta = normalise_angle(b.start_heading - a.end_heading)
        if abs(delta) > math.radians(150.0):
            return False                              # U-turn
        return self._lane_pairing_ok(a, b, delta)

    def _fallback_targets(self, a: Lane, outs: List[Lane]) -> List[Lane]:
        legal = [b for b in outs
                 if abs(normalise_angle(b.start_heading - a.end_heading))
                 <= math.radians(150.0)]
        if not legal:
            return []
        return [min(legal, key=lambda b: (abs(abs(b.lane_id) - abs(a.lane_id)),
                                          b.uid))]

    def _add_connector(self, junction_road: int, a: Lane, b: Lane) -> None:
        ax, ay, ah = a.pose_at(a.length)
        bx, by, bh = b.pose_at(0.0)
        turn = classify_turn(ah, bh)
        conn = self._add_lane(junction_road, a.lane_id,
                              _bezier((ax, ay), ah, (bx, by), bh),
                              True, f"j{junction_road}:{turn}:{a.uid}->{b.uid}")
        _link(a, conn, turn)
        _link(conn, b, "straight")

    def _lane_pairing_ok(self, a: Lane, b: Lane, delta: float) -> bool:
        """Which lane may feed which through a junction.

        Straight-ahead keeps the lane index; a right turn is taken from the
        outermost lane and a left turn from the innermost, as on a real road.
        """
        n = self.lanes_per_dir
        if n == 1:
            return True
        ia, ib = abs(a.lane_id), abs(b.lane_id)
        if abs(delta) < math.radians(20.0):
            return ia == ib
        if delta > 0:                                  # right turn
            return ia == n and ib == n
        return ia == 1 and ib == 1                     # left turn


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

BUILTIN_TOWNS: Dict[str, GridTown] = {
    "grid": GridTown(
        name="grid",
        xs=(0.0, 80.0, 160.0), ys=(0.0, 80.0, 160.0),
        lanes_per_dir=2,
        description="3x3 junctions, 80 m spacing, two lanes each way. "
                    "One four-way junction, at (80, 80).",
    ),
    "loop": GridTown(
        name="loop",
        xs=(0.0, 240.0), ys=(0.0, 140.0),
        lanes_per_dir=2,
        description="Single rectangular circuit with long straights; "
                    "car-following demos never run out of road. No four-way "
                    "junction -- every corner is a two-arm turn.",
    ),
    "wide_grid": GridTown(
        name="wide_grid",
        xs=(-80.0, 0.0, 80.0, 160.0), ys=(-80.0, 0.0, 80.0, 160.0),
        lanes_per_dir=2,
        description="4x4 junctions covering x,y in [-80, 160]. Four four-way "
                    "junctions -- (0,0), (0,80), (80,0), (80,80) -- each with "
                    "66 m of straight approach on every arm. The one to stage "
                    "junction conflicts in.",
    ),
}

#: CARLA town names the scenarios ask for, mapped onto a local stand-in.
TOWN_ALIASES: Dict[str, str] = {
    "town01": "grid",
    "town02": "grid",
    "town03": "wide_grid",
    "town04": "loop",
    "town05": "wide_grid",
    "town06": "loop",
    "town07": "grid",
    "town10hd": "grid",
    "town10hd_opt": "grid",
}

DEFAULT_TOWN = "grid"


def resolve_town_name(map_name: Optional[str]) -> Tuple[str, bool]:
    """``(local town name, was_aliased)`` for a requested map name."""
    if not map_name:
        return DEFAULT_TOWN, False
    key = str(map_name).strip().split("/")[-1].lower()
    if key in BUILTIN_TOWNS:
        return key, False
    if key in TOWN_ALIASES:
        return TOWN_ALIASES[key], True
    return DEFAULT_TOWN, True


def load_town(map_name: Optional[str] = None,
              turn_preference: str = "straight") -> Map:
    """Build the local road network standing in for ``map_name``."""
    town_name, _ = resolve_town_name(map_name)
    return BUILTIN_TOWNS[town_name].build(turn_preference=turn_preference)


def town_names() -> List[str]:
    return sorted(BUILTIN_TOWNS)
