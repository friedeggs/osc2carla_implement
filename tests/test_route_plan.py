"""Simulator-free checks of the planned route and its junction choice.

The fake road below is Town10HD_Opt's red_light junction reduced to what
matters: a westbound lane that branches, 30 m along, into a straight connector
(road 254) and a left one (road 338, -90 deg). That is the branch that used to
be decided by whichever waypoint CARLA enumerated first.
"""
import math

from osc2carla.backend import route as R


class FakeLocation(object):
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class FakeTransform(object):
    def __init__(self, x, y, yaw):
        self.location = FakeLocation(x, y)
        self.rotation = type("R", (), {"yaw": yaw})()


class FakeWaypoint(object):
    """A point on a lane that knows its successors."""

    def __init__(self, road_id, x, y, yaw, is_junction=False, lane_id=1):
        self.road_id = road_id
        self.lane_id = lane_id
        self.is_junction = is_junction
        self.transform = FakeTransform(x, y, yaw)
        self._next = []

    def next(self, distance):
        return list(self._next)


def _lane(road_id, x0, y0, yaw, count, junction=False):
    """`count` waypoints at 1 m spacing, chained."""
    out = []
    for i in range(count):
        x = x0 + i * math.cos(math.radians(yaw))
        y = y0 + i * math.sin(math.radians(yaw))
        out.append(FakeWaypoint(road_id, x, y, yaw, is_junction=junction))
    for a, b in zip(out, out[1:]):
        a._next = [b]
    return out


class FakeMap(object):
    """Approach lane -> {straight connector, left connector}."""

    def __init__(self):
        self.approach = _lane(20, 1.0, 16.6, 180.0, 31)
        self.straight = _lane(254, -30.0, 16.6, 180.0, 60, junction=True)
        # +y is the ego's left when it heads west (yaw 180), so the left
        # connector runs at yaw +90 -- which scores -90 deg. See the module
        # docstring in backend/route.py on CARLA's handedness.
        self.left = _lane(338, -30.0, 16.6, 90.0, 60, junction=True)
        # The branch. Order deliberately puts the LEFT connector first, which is
        # exactly the ordering that produced the bug this module exists to fix.
        self.approach[-1]._next = [self.left[0], self.straight[0]]

    def get_waypoint(self, location, project_to_road=True):
        return self.approach[0]


m = FakeMap()

# 1. straight is chosen over an earlier-enumerated left turn
plan = R.plan(m, FakeLocation(1.0, 16.6), preference="straight", length_m=80.0)
end = plan.world_polyline()[-1]
assert plan.decisions and plan.decisions[0].chosen_road == 254, plan.decisions
assert abs(end[1] - 16.6) < 1e-6, end
print("straight -> road %d, ends (%.1f, %.1f)  [left was offered first]"
      % (plan.decisions[0].chosen_road, end[0], end[1]))

# 2. the same branch, the other way, when the scenario declares a left turn
plan_left = R.plan(m, FakeLocation(1.0, 16.6), preference="left", length_m=80.0)
end_left = plan_left.world_polyline()[-1]
assert plan_left.decisions[0].chosen_road == 338, plan_left.decisions
assert end_left[1] > 40.0, end_left
print("left     -> road %d, ends (%.1f, %.1f)"
      % (plan_left.decisions[0].chosen_road, end_left[0], end_left[1]))

# 3. CARLA yaw increases to the vehicle's right, so a left exit scores negative
delta = plan.decisions[0].alternatives[0][1]
assert delta < -80.0, delta
print("the rejected exit measured %+.0f deg (negative == the ego's left)" % delta)

# 4. a preference for a turn the junction does not offer falls back to the
#    nearest thing on that side rather than inventing one
plan_right = R.plan(m, FakeLocation(1.0, 16.6), preference="right", length_m=80.0)
assert plan_right.decisions[0].chosen_road == 254, plan_right.decisions
print("right    -> road %d (no right exit here; the straight one is nearest)"
      % plan_right.decisions[0].chosen_road)

# 5. the plan is walked once and sliced, so the route a policy sees is the same
#    route wherever it has got to -- the property the per-tick walk did not have
for x in (0.0, -10.0, -20.0, -28.0, -29.5, -31.6, -40.0):
    served = plan.ahead(x, 16.6, first_m=2.5, step_m=1.0, count=20)
    assert served, x
    assert all(abs(y - 16.6) < 1e-6 for _sx, y, _h in served), (x, served[-1])
print("every tick from x=0 to x=-40 is served a straight 20-point route")

# 6. spacing is by arc length, and a route that runs out comes back short
served = plan.ahead(-20.0, 16.6, first_m=2.5, step_m=1.0, count=20)
gaps = [math.dist(served[i][:2], served[i + 1][:2]) for i in range(len(served) - 1)]
assert all(abs(g - 1.0) < 1e-6 for g in gaps), gaps
tail = plan.ahead(-72.0, 16.6, first_m=2.5, step_m=1.0, count=20)
assert 0 < len(tail) < 20, len(tail)
past = plan.ahead(-86.0, 16.6, first_m=2.5, step_m=1.0, count=20)
assert past == [], past
print("1 m spacing held; %d points left near the end of the plan (not padded), "
      "and nothing past it" % len(tail))

# 7. memoization: one plan per actor, so nobody re-derives it mid-run
R.reset()
actor = type("A", (), {"id": 7, "get_location": lambda self: FakeLocation(1.0, 16.6)})()
first = R.plan_for(m, actor, preference="straight", length_m=80.0)
assert R.plan_for(m, actor, preference="straight", length_m=80.0) is first
print("plan_for returns the one plan per actor")

# 8. a later caller wanting more road gets it -- the shared plan is as long as
#    the longest ask, not as long as whoever asked first
R.reset()
short = R.plan_for(m, actor, preference="straight", length_m=40.0)
longer = R.plan_for(m, actor, preference="straight", length_m=80.0)
assert len(longer) > len(short), (len(short), len(longer))
assert R.plan_for(m, actor, preference="straight", length_m=40.0) is longer
print("a 40 m plan (%d pts) is replaced by an 80 m ask (%d pts), and kept"
      % (len(short), len(longer)))
first = longer

# 9. leaving the plan is reported, not repaired
first.locate(-30.0, 30.0)
assert first.describe()["off_plan_ticks"] == 1, first.describe()
print("off-plan ticks are counted: %s" % first.describe()["off_plan_ticks"])

print("\nall route-plan checks passed")
