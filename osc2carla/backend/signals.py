"""The signal phase the ego meets at its junction, because the scenario says so.

Why this exists
---------------
This baseline's ``.osc`` dialect has no action that sets or reads a signal
phase, so until now the phase the ego met was whichever one CARLA's own cycle
happened to be showing when the scenario spawned. Three of the benchmark
families are *about* the phase:

  ``red_light``   the ego crosses on green while another vehicle enters against
                  its red. The violation is staged as behaviour and geometry --
                  the violator simply does not brake -- but the ego's own half of
                  that sentence, "on green", was never set.
  ``left_turn``   an unprotected left: the ego turns across oncoming traffic that
                  has the same green, which is what makes it unprotected.
  ``right_turn``  a right turn on red. The ego's light is red *on purpose*; the
                  traffic it merges into is the one with the green.

Left unset, the phase is a coin toss taken at spawn, and it decides the run: a
policy that obeys lights stops at the line and the conflict never develops,
while a policy that ignores them drives through and it does. Measured on
``red_light`` + PlanT 2.0, the ego halted 6.2 m short of traffic light 952 and
sat there for the remaining ten seconds, and the same cell under IDM -- which
has no notion of a signal at all -- was struck by the violator in the junction.
That is not two policies disagreeing about the scenario; it is one of them being
shown a scenario the other never saw.

What is applied
---------------
The light governing the ego's approach is found from the map's own signal
records (the OpenDRIVE landmark on the ego's lane), set to the declared state,
and frozen so the cycle cannot advance underneath the episode. The rest of that
junction's group is set to a phase consistent with it: the approaches parallel
to the ego's -- its own and the one opposing it -- take the same state, and the
crossing approaches take the complement. So ``red_light``'s violator faces the
red it is described as running, and ``left_turn``'s oncoming traffic keeps the
green that makes the turn unprotected.

The crossing lights are, today, cosmetic: scripted actors never yield to CARLA's
signals, and the observation carries only the ego's own light. They are set
anyway, because a recording of this scenario should show what the scenario says
is happening, and because a future actor that does read them should read
something true.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Declared phase -> the ``carla.TrafficLightState`` name to apply.
STATES = {"green": "Green", "yellow": "Yellow", "red": "Red"}

#: The phase the crossing approaches take when the ego's is `key`.
COMPLEMENT = {"green": "red", "yellow": "red", "red": "green"}

#: Two approaches are the same axis when their headings are within this of
#: parallel or anti-parallel. Junction arms are not exactly square on these
#: towns, so the tolerance is generous; it only ever has to separate "along the
#: ego's road" from "across it".
AXIS_TOLERANCE_DEG = 45.0

#: How far ahead of the ego its governing signal is looked for. Its spawn is
#: tens of metres back from the line on every junction family.
SEARCH_M = 150.0

#: OpenDRIVE signal type for a traffic light.
LANDMARK_TRAFFIC_LIGHT = "1000001"


def _normalize_deg(x: float) -> float:
    x = float(x) % 360.0
    return x - 360.0 if x > 180.0 else x


def phase_for(declared: str, ego_yaw: Optional[float],
              member_yaw: Optional[float]) -> str:
    """The phase one light in the group takes, given the ego's declared one.

    Approaches on the ego's own axis -- its own and the one opposing it -- hold
    the ego's phase, so ``left_turn``'s oncoming traffic keeps the green that
    makes the turn unprotected. Everything across it holds the complement, so
    ``red_light``'s violator faces the red it is described as running. An
    approach whose heading cannot be read takes the complement: that is the
    answer that cannot contradict the ego's own phase.
    """
    complement = COMPLEMENT[declared]
    if ego_yaw is None or member_yaw is None:
        return complement
    offset = abs(_normalize_deg(member_yaw - ego_yaw))
    parallel = (offset <= AXIS_TOLERANCE_DEG
                or offset >= 180.0 - AXIS_TOLERANCE_DEG)
    return declared if parallel else complement


def _approach_yaw(light) -> Optional[float]:
    """The heading a vehicle stopped at ``light`` is facing."""
    try:
        stops = list(light.get_stop_waypoints())
    except (AttributeError, RuntimeError):            # pragma: no cover
        return None
    if not stops:
        return None
    return float(stops[0].transform.rotation.yaw)


def _ego_light(world, carla_map, actor):
    """``(light, approach_yaw)`` for the signal governing the ego, or ``(None, None)``.

    Not ``actor.get_traffic_light()``: that answers only once the vehicle is
    inside the light's trigger volume, and at spawn the ego is 30 m or more short
    of it -- measured on ``red_light``, it returns ``None`` everywhere from the
    spawn at x = 1.31 until x = -22.5.

    So the map's own signal records are searched instead, and the candidates are
    filtered by *which way they face*. That filter is not optional: taking the
    nearest landmark that happens to resolve to a light picked one from an
    entirely different junction (actor 9) over the one that actually governs the
    ego's approach (actor 16, approach heading 180.2 deg against the ego's
    180.2). A signal the ego will never meet is worse than none, because setting
    it looks like the phase was controlled.
    """
    try:
        waypoint = carla_map.get_waypoint(actor.get_location(),
                                          project_to_road=True)
    except (AttributeError, RuntimeError):            # pragma: no cover
        return None, None
    if waypoint is None:
        return None, None
    ego_yaw = float(waypoint.transform.rotation.yaw)

    try:
        landmarks = list(waypoint.get_landmarks_of_type(
            SEARCH_M, LANDMARK_TRAFFIC_LIGHT, False))
    except (AttributeError, RuntimeError):            # pragma: no cover
        landmarks = []
    landmarks.sort(key=lambda mark: getattr(mark, "distance", 0.0))

    for landmark in landmarks:
        try:
            light = world.get_traffic_light(landmark)
        except (AttributeError, RuntimeError):        # pragma: no cover
            continue
        if light is None:
            continue
        yaw = _approach_yaw(light)
        # Same direction, not merely the same axis: the opposing approach has a
        # light too, and it governs the other carriageway.
        if yaw is not None and abs(_normalize_deg(yaw - ego_yaw)) <= AXIS_TOLERANCE_DEG:
            return light, ego_yaw
    return None, ego_yaw


def apply(world, carla_map, actor, phase: Optional[str]) -> Dict[str, Any]:
    """Set and freeze the ego's junction phase. Returns what was done.

    Never raises: a scenario that declares no phase, a map with no signal on the
    ego's approach, or a backend with no traffic lights at all are all reported
    rather than treated as failures. Every one of them is a run that is still
    worth having -- it just is not one that can claim the phase was controlled.
    """
    declared = str(phase or "").strip().lower()
    if not declared:
        return {"requested": None,
                "note": "the scenario declares no phase for the ego; whatever "
                        "the simulator's own cycle was showing applied"}
    if declared not in STATES:
        return {"requested": declared, "applied": None,
                "note": "unrecognized phase %r; nothing was set" % declared}
    if world is None or carla_map is None or actor is None:
        return {"requested": declared, "applied": None,
                "note": "no CARLA world on this backend; no signal to set"}

    try:
        import carla                                   # type: ignore
    except ImportError:                                # pragma: no cover
        return {"requested": declared, "applied": None,
                "note": "the carla module is not importable here"}

    light, ego_yaw = _ego_light(world, carla_map, actor)
    if light is None:
        return {"requested": declared, "applied": None,
                "note": "no traffic light faces the ego's approach within "
                        "%.0f m; the scenario asked for a phase at a junction "
                        "that has none" % SEARCH_M}
    try:
        group = list(light.get_group_traffic_lights())
    except (AttributeError, RuntimeError):             # pragma: no cover
        group = [light]
    if not any(member.id == light.id for member in group):
        group.append(light)

    complement = COMPLEMENT[declared]
    applied: List[Dict[str, Any]] = []
    # Every state first, and the freeze only once all of them are set: freezing
    # holds the lights at their current state, so a freeze taken mid-loop would
    # be a freeze of a half-applied phase.
    for member in group:
        if member.id == light.id:
            phase_for_member = declared
        else:
            phase_for_member = phase_for(declared, ego_yaw,
                                         _approach_yaw(member))
        try:
            member.set_state(getattr(carla.TrafficLightState,
                                     STATES[phase_for_member]))
        except (AttributeError, RuntimeError) as exc:  # pragma: no cover
            applied.append({"id": int(member.id), "error": str(exc)})
            continue
        applied.append({"id": int(member.id), "phase": phase_for_member,
                        "ego": member.id == light.id})

    # `TrafficLight.freeze` stops every light in the scene at its current state,
    # not just this one -- so it is taken last, once the whole group holds the
    # phase the scenario asked for. Taken any earlier it pins the lights it has
    # not reached yet at whatever the cycle was showing, which is how the ego
    # came to face a permanently frozen Yellow while a light at another junction
    # held the green that was meant for it.
    frozen = True
    try:
        light.freeze(True)
    except (AttributeError, RuntimeError) as exc:      # pragma: no cover
        frozen = False
        applied.append({"freeze_error": str(exc)})

    return {
        "requested": declared,
        "applied": declared,
        "ego_light_id": int(light.id),
        "group": applied,
        "frozen": frozen,
        "note": "the ego's approach and the one opposing it hold %s; the "
                "crossing approaches hold %s, and the group is frozen for the "
                "episode" % (declared, complement),
    }
