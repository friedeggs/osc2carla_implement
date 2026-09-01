"""One route per vehicle, planned once, taking the junction exit the scenario asked for.

Why a plan and not a per-tick walk
----------------------------------
Every route in this repository used to be re-derived on the tick that needed it,
by projecting the actor onto the map and walking ``waypoint.next(d)[0]``. Two
things go wrong with that, and both were measured on ``red_light`` running on
Town10HD_Opt:

  * ``get_waypoint(location, project_to_road=True)`` answers "the nearest driving
    lane", and around a junction the connecting lanes overlap the lane the
    vehicle is actually on. Junction road 338 -- the left-turn connector -- begins
    at x = -30.3 on the ego's own centreline (y ~ 16.6). At x = -29.51 the
    projection returns 338 rather than the through connector 254, and from that
    tick on the route handed to the ego turns left. The ego was still 19 m short
    of the junction, driving straight, with a straight route recorded in
    ``scene.json`` -- only the route it was *given* had turned.
  * because the projection is redone every tick, the answer is not even stable.
    Under IDM the same junction flipped between roads 382, 467, 338, 255, 256 and
    466 on consecutive 100 ms ticks, each with a different exit, and the ego
    U-turned onto the opposite carriageway.

A route is a property of the scenario, not of where the vehicle happens to be
100 ms from now. So it is planned once, from the spawn, and every tick is served
a slice of that one plan. ``next(d)[0]`` is still how the plan is walked, but
where the lane genuinely branches the exit is *chosen* -- by the preference the
scenario declares -- rather than being whichever branch CARLA enumerated first.

That is the same rule the local backend has always applied: ``localsim``'s
``Map.next()`` orders candidates by ``turn_preference`` so that ``[0]`` *is* the
preferred manoeuvre (``localsim/roadmap.py``). Only the CARLA path inherited the
``[0]`` idiom without the ordering that gives it a meaning.

Handedness
----------
CARLA yaw increases toward the vehicle's right: on a lane whose yaw is 180 deg,
``transform.get_right_vector()`` points at -y. So among the exits of a junction
the left one is the most *negative* change in yaw, the right one the most
positive, and "straight" is the smallest change in absolute value.

Who plans
---------
Nobody plans twice: :func:`plan_for` memoizes per actor, so the runner's own
route sampling (``backend/policy.py``), the object-centric observation served to
a bridged policy (``scenario_orchestration/carla_state_obs.py``) and the
reference path recorded into the trace (``backend/trace.py``) are all slices of
one plan and cannot disagree with each other. They used to: the trace walked
from the spawn and went straight while the policy's own route, walked from the
ego's live position, turned left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

STRAIGHT = "straight"
LEFT = "left"
RIGHT = "right"
PREFERENCES = (STRAIGHT, LEFT, RIGHT)

#: How far ahead a plan is walked. Long enough that no episode outruns its route:
#: the longest benchmark scenario is 26 s and the fastest ego seen is ~11 m/s.
DEFAULT_LENGTH_M = 300.0

#: Plan resolution. Every consumer resamples from this, so it is the finest
#: spacing any of them asks for rather than a per-consumer choice.
DEFAULT_STEP_M = 1.0

#: How far along a candidate exit its manoeuvre is measured. A junction
#: connector barely deviates in its first metre -- the branches only separate
#: once they are into the box -- so comparing the candidates' own yaw decides
#: nothing. This is far enough to clear the junction on the benchmark towns.
EXIT_LOOKAHEAD_M = 40.0

#: Beyond this lateral distance from its plan a vehicle is no longer on the
#: route it was given. It is reported, not repaired: replanning from a position
#: this far off is the per-tick walk again, and the point of a plan is that the
#: scenario decides the exit once.
OFF_PLAN_M = 8.0


def normalize_preference(value: Any, default: str = STRAIGHT) -> str:
    """A declared preference, or ``default`` when it is absent or unrecognized."""
    text = str(value).strip().lower() if value is not None else ""
    return text if text in PREFERENCES else default


#: Set once per process from ``--junction-turn`` (see ``osc2carla/cli.py``), so a
#: consumer that is handed no preference of its own still plans the route the
#: scenario asked for. The bridged-policy observation is the one that needs this:
#: it is loaded by class name and cannot be given command-line arguments.
_default_preference = STRAIGHT


def set_default_preference(value: Any) -> str:
    global _default_preference
    _default_preference = normalize_preference(value)
    return _default_preference


def default_preference() -> str:
    return _default_preference


def _normalize_deg(x: float) -> float:
    """An angle difference folded into (-180, 180]."""
    x = float(x) % 360.0
    if x > 180.0:
        x -= 360.0
    return x


@dataclass
class Decision:
    """One junction the plan chose an exit at, kept for the run report."""
    s_m: float
    chosen_road: int
    chosen_delta_deg: float
    alternatives: List[Tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"s_m": round(self.s_m, 1),
                "took": {"road": self.chosen_road,
                         "delta_deg": round(self.chosen_delta_deg, 1)},
                "over": [{"road": r, "delta_deg": round(d, 1)}
                         for r, d in self.alternatives]}


@dataclass
class RoutePlan:
    """A walked route, and the slices of it a consumer asks for each tick."""

    points: List[Tuple[float, float, float]]      # world x, y, heading (rad)
    step_m: float = DEFAULT_STEP_M
    preference: str = STRAIGHT
    decisions: List[Decision] = field(default_factory=list)
    truncated: bool = False

    #: Progress is monotone: the nearest point is searched near the last one, so
    #: a route that passes close to itself cannot snap the vehicle backwards.
    _cursor: int = 0
    _max_off_plan: float = 0.0
    _off_plan_ticks: int = 0

    def __len__(self) -> int:
        return len(self.points)

    # -- querying ---------------------------------------------------------- #
    def locate(self, x: float, y: float) -> Tuple[int, float]:
        """``(index, distance)`` of the plan point nearest ``(x, y)``."""
        if not self.points:
            return 0, float("inf")
        lo = max(0, self._cursor - 5)
        hi = min(len(self.points), self._cursor + 60)
        best_i, best_d = lo, float("inf")
        for i in range(lo, hi):
            px, py, _ = self.points[i]
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_d:
                best_i, best_d = i, d
        self._cursor = best_i
        dist = math.sqrt(best_d)
        if dist > self._max_off_plan:
            self._max_off_plan = dist
        if dist > OFF_PLAN_M:
            self._off_plan_ticks += 1
        return best_i, dist

    def ahead(self, x: float, y: float, *, first_m: float, step_m: float,
              count: int) -> List[Tuple[float, float, float]]:
        """``count`` world-frame ``(x, y, heading)`` points along the plan.

        Measured by arc length from the vehicle's position on the plan, so the
        spacing a policy was trained on is the spacing it gets whatever the plan
        resolution is. A plan that runs out returns short: a route is not
        invented past its end, and the caller reports the shortfall.
        """
        if not self.points or count <= 0:
            return []
        index, _ = self.locate(x, y)
        out: List[Tuple[float, float, float]] = []
        for k in range(count):
            target = first_m + k * step_m
            position = index + target / self.step_m
            if position > len(self.points) - 1:
                break
            out.append(self._interpolate(position))
        return out

    def _interpolate(self, position: float) -> Tuple[float, float, float]:
        low = int(math.floor(position))
        frac = position - low
        x0, y0, h0 = self.points[low]
        if frac <= 1e-9 or low + 1 >= len(self.points):
            return x0, y0, h0
        x1, y1, h1 = self.points[low + 1]
        turn = math.radians(_normalize_deg(math.degrees(h1 - h0)))
        return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0), h0 + frac * turn

    def world_polyline(self) -> List[Tuple[float, float]]:
        return [(x, y) for x, y, _ in self.points]

    # -- reporting --------------------------------------------------------- #
    def describe(self) -> Dict[str, Any]:
        return {
            "planned_once": True,
            "preference": self.preference,
            "points": len(self.points),
            "step_m": self.step_m,
            "length_m": round(max(0, len(self.points) - 1) * self.step_m, 1),
            "truncated": self.truncated,
            "junctions": [d.to_dict() for d in self.decisions],
            "max_off_plan_m": round(self._max_off_plan, 2),
            "off_plan_ticks": self._off_plan_ticks,
        }


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def _exit_delta_deg(candidate, base_yaw: float, step_m: float) -> float:
    """How far ``candidate`` turns, in degrees, positive to the vehicle's right.

    Followed to the far side of the junction before the angle is taken, because
    one metre into the box every exit still points the same way.
    """
    wp = candidate
    travelled = 0.0
    while travelled < EXIT_LOOKAHEAD_M:
        try:
            nxt = wp.next(step_m)
        except (RuntimeError, AttributeError):        # pragma: no cover
            break
        if not nxt:
            break
        wp = nxt[0]
        travelled += step_m
        if not getattr(wp, "is_junction", False) and travelled > step_m:
            break                                     # out the other side
    return _normalize_deg(wp.transform.rotation.yaw - base_yaw)


def _choose(current, candidates: Sequence[Any], preference: str,
            step_m: float) -> Tuple[Any, Optional[Decision]]:
    """The successor the scenario asked for, and what it was chosen over."""
    if len(candidates) == 1:
        return candidates[0], None
    base = current.transform.rotation.yaw
    scored = [(_exit_delta_deg(c, base, step_m), c) for c in candidates]
    if preference == LEFT:
        pick = min(scored, key=lambda item: item[0])
    elif preference == RIGHT:
        pick = max(scored, key=lambda item: item[0])
    else:
        pick = min(scored, key=lambda item: abs(item[0]))
    decision = Decision(
        s_m=0.0,
        chosen_road=int(getattr(pick[1], "road_id", -1)),
        chosen_delta_deg=pick[0],
        alternatives=[(int(getattr(c, "road_id", -1)), d)
                      for d, c in scored if c is not pick[1]],
    )
    return pick[1], decision


def plan(carla_map, location, *, preference: Any = None,
         length_m: float = DEFAULT_LENGTH_M,
         step_m: float = DEFAULT_STEP_M) -> RoutePlan:
    """Walk one route from ``location``, choosing each junction exit.

    ``location`` is a ``carla.Location`` (or anything with ``x``/``y``/``z``).
    The walk starts at the lane the location projects onto, which at a spawn is
    unambiguous -- planning from a *spawn* is what makes the projection safe,
    and re-projecting later is what this module exists to stop.
    """
    pref = normalize_preference(preference, default=_default_preference)
    if carla_map is None:
        return RoutePlan(points=[], step_m=step_m, preference=pref)
    try:
        wp = carla_map.get_waypoint(location, project_to_road=True)
    except (RuntimeError, AttributeError):            # pragma: no cover
        return RoutePlan(points=[], step_m=step_m, preference=pref)
    if wp is None:
        return RoutePlan(points=[], step_m=step_m, preference=pref)

    tf = wp.transform
    points = [(tf.location.x, tf.location.y, math.radians(tf.rotation.yaw))]
    decisions: List[Decision] = []
    seen = {(round(tf.location.x, 2), round(tf.location.y, 2))}
    travelled = 0.0
    truncated = True
    while travelled < length_m:
        try:
            nxt = wp.next(step_m)
        except (RuntimeError, AttributeError):        # pragma: no cover
            break
        if not nxt:
            break                                     # the road ends here
        wp, decision = _choose(wp, nxt, pref, step_m)
        if decision is not None:
            decision.s_m = travelled
            decisions.append(decision)
        tf = wp.transform
        key = (round(tf.location.x, 2), round(tf.location.y, 2))
        if key in seen:
            break                                     # a loop connector
        seen.add(key)
        travelled += step_m
        points.append((tf.location.x, tf.location.y,
                       math.radians(tf.rotation.yaw)))
    else:
        truncated = False
    return RoutePlan(points=points, step_m=step_m, preference=pref,
                     decisions=decisions, truncated=truncated)


#: actor id -> the one plan that actor drives. Cleared by :func:`reset`.
_plans: Dict[Tuple[int, str], RoutePlan] = {}


def plan_for(carla_map, actor, *, preference: Any = None,
             length_m: float = DEFAULT_LENGTH_M,
             step_m: float = DEFAULT_STEP_M) -> RoutePlan:
    """The plan for ``actor``, walked on first ask and reused after.

    First ask is at the spawn, before anything has moved -- which is the whole
    reason this is memoized rather than recomputed: asking again from mid-lane
    is the projection ambiguity the module docstring describes.
    """
    pref = normalize_preference(preference, default=_default_preference)
    key = (int(getattr(actor, "id", id(actor))), pref)
    cached = _plans.get(key)
    # Length is the one thing a later caller may legitimately want more of, and
    # the plan is shared -- so the longest ask wins rather than whoever asked
    # first. A plan that ran out of road is not re-walked: it is already as long
    # as the road allows, and asking again would only walk the same dead end.
    if cached is not None and (cached.truncated
                               or len(cached) - 1 >= length_m / cached.step_m - 1e-9):
        return cached
    fresh = plan(carla_map, actor.get_location(), preference=pref,
                 length_m=length_m, step_m=step_m)
    _plans[key] = fresh
    return fresh


def reset() -> None:
    """Forget every plan. For tests, and for a process that runs two scenarios."""
    _plans.clear()
