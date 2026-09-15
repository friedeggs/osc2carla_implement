"""Simulator-free checks of the ego's declared junction phase.

The phase each light in a junction group takes is decided from approach
headings alone, so the decision is testable without a simulator; only the
setting and freezing of real CARLA actors is not.
"""
from osc2carla.backend import signals as S

# The red_light junction on Town10HD_Opt: the ego approaches westbound (yaw
# 180), the violator and the stopper northbound on the crossing road (yaw 90),
# and the opposing approach is eastbound (yaw 0).
EGO, OPPOSING, CROSSING = 180.0, 0.0, 90.0

# 1. the ego crosses on green; the crossing road takes the red it is described
#    as running, and the opposing approach keeps the same green
assert S.phase_for("green", EGO, EGO) == "green"
assert S.phase_for("green", EGO, OPPOSING) == "green"
assert S.phase_for("green", EGO, CROSSING) == "red"
print("red_light  ego green -> opposing green, crossing red")

# 2. left_turn is unprotected precisely because the oncoming approach on the
#    ego's own axis holds the same green
assert S.phase_for("green", EGO, OPPOSING) == "green", "an unprotected left needs oncoming green"
print("left_turn  ego green -> oncoming (opposing) green: the turn is unprotected")

# 3. right_turn is a right turn on red, so the traffic the ego merges into --
#    the crossing road -- is the one with the green
assert S.phase_for("red", EGO, EGO) == "red"
assert S.phase_for("red", EGO, CROSSING) == "green"
assert S.phase_for("red", EGO, OPPOSING) == "red"
print("right_turn ego red   -> crossing green: the traffic it merges into flows")

# 4. junction arms are not square on these towns, so the axis test has slack
for skew in (-40.0, -20.0, 0.0, 20.0, 40.0):
    assert S.phase_for("green", EGO, OPPOSING + skew) == "green", skew
    assert S.phase_for("green", EGO, CROSSING + skew / 4.0) == "red", skew
print("the axis test tolerates +/-%.0f deg of skew" % S.AXIS_TOLERANCE_DEG)

# 5. an approach whose heading cannot be read takes the phase that cannot
#    contradict the ego's
assert S.phase_for("green", EGO, None) == "red"
assert S.phase_for("red", None, CROSSING) == "green"
print("an unreadable approach takes the complement")

# 6. yellow is a stopping phase, so the crossing road still gets the red
assert S.phase_for("yellow", EGO, CROSSING) == "red"
print("yellow is treated as stopping: crossing stays red")

# 7. nothing declared changes nothing, and says so
note = S.apply(None, None, None, None)
assert note["requested"] is None and "declares no phase" in note["note"], note
note = S.apply(None, None, None, "flashing")
assert note["applied"] is None and "unrecognized" in note["note"], note
note = S.apply(None, None, None, "green")
assert note["applied"] is None and "no CARLA world" in note["note"], note
print("an absent, unrecognized, or unsettable phase is reported, not raised")

# 8. the trace records each light of the group with its role for the ego's
#    approach; an unreadable heading is recorded as unknown, not guessed
assert S.role_for(EGO, EGO) == "ego"
assert S.role_for(EGO, OPPOSING) == "opposing"
assert S.role_for(EGO, CROSSING) == "crossing"
assert S.role_for(EGO, None) == "unknown"
print("each light of the group is recorded with its role")


class _Rot(object):
    def __init__(self, yaw):
        self.yaw = yaw


class _Tf(object):
    def __init__(self, yaw):
        self.rotation = _Rot(yaw)


class _Wp(object):
    def __init__(self, yaw, marks=()):
        self.transform = _Tf(yaw)
        self._marks = list(marks)

    def get_landmarks_of_type(self, distance, kind, stop_at_junction):
        return self._marks


class _Mark(object):
    def __init__(self, light_id, distance):
        self.id, self.distance = light_id, distance


class _Light(object):
    def __init__(self, light_id, yaw):
        self.id, self.yaw, self.group = light_id, yaw, []

    def get_stop_waypoints(self):
        return [_Wp(self.yaw)]

    def get_group_traffic_lights(self):
        return self.group


class _World(object):
    def __init__(self, lights):
        self.lights = {l.id: l for l in lights}

    def get_traffic_light(self, mark):
        return self.lights.get(mark.id)


class _Map(object):
    def __init__(self, wp):
        self.wp = wp

    def get_waypoint(self, location, project_to_road=True):
        return self.wp


class _Actor(object):
    def get_location(self):
        return None


# 9. the red_light junction: a light across the road is nearer in the map's
#    records than the ego's own, and the group is read off the ego's light
ego_light, opp, cross_a, cross_b = (_Light(16, EGO), _Light(17, OPPOSING),
                                    _Light(15, CROSSING), _Light(23, CROSSING + 180.0))
for light in (ego_light, opp, cross_a, cross_b):
    light.group = [ego_light, opp, cross_a, cross_b]
world = _World([ego_light, opp, cross_a, cross_b])
cmap = _Map(_Wp(EGO, marks=[_Mark(15, 20.0), _Mark(16, 31.0)]))
roles = {light.id: role for light, role in S.junction_lights(world, cmap, _Actor())}
assert roles == {16: "ego", 17: "opposing", 15: "crossing", 23: "crossing"}, roles
assert S.junction_lights(_World([]), _Map(_Wp(EGO)), _Actor()) == []
print("the recorded group is the ego's junction, with each light's role")

print("\nall signal-phase checks passed")
