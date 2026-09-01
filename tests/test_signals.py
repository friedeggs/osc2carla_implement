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

print("\nall signal-phase checks passed")
