"""Atomic behaviours: the py_trees leaves an OSC2 action compiles to.

Written against CARLA's actor API and reached through :mod:`.simapi`, so the
same leaves drive either the CARLA server or the bundled local simulator --
which is the point of the indirection: an action's semantics are a property of
the language, not of the simulator underneath.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

import py_trees

from .context import ExecutionContext, Quantity, _ActorHandle
from .method_registry import register
from .simapi import sim as carla


def _value(q) -> float:
    if isinstance(q, Quantity):
        return q.value
    return float(q)


def _carla_actor(actor_or_handle):
    if isinstance(actor_or_handle, _ActorHandle):
        return actor_or_handle.carla_actor
    return actor_or_handle


def _wrap_speed(v_setpoint, ctx: ExecutionContext) -> float:
    if callable(v_setpoint):
        return _value(v_setpoint())
    if isinstance(v_setpoint, str):
        return _value(ctx.get_variable(v_setpoint))
    return _value(v_setpoint)


def _speed_modifier(modifiers, ctx: ExecutionContext):
    """The setpoint of a ``speed(...)`` modifier, or None if there is none.

    Shared by ``drive`` and ``change_lane``: the modifier means the same thing
    on both -- hold this speed while the action runs.
    """
    for m in modifiers:
        if m.name != "speed":
            continue
        if m.args.positional:
            return ctx.eval(m.args.positional[0])
        if "target" in m.args.named:
            return ctx.eval(m.args.named["target"])
    return None


class _SpeedPID:
    """PID on speed error, producing a throttle/brake pair.

    Factored out of :class:`WaypointFollowerLite` so ``change_lane`` can hold
    a commanded speed with the same law ``drive`` uses. A manoeuvre whose
    speed is a fixed throttle takes a distance that depends on the road's
    slope, which is exactly what a choreographed conflict cannot tolerate.
    """

    def __init__(self, kp=0.6, ki=0.05, kd=0.1, dt=0.05,
                 max_throttle=1.0, max_brake=1.0):
        self._kp, self._ki, self._kd, self._dt = kp, ki, kd, dt
        self._max_throttle, self._max_brake = max_throttle, max_brake
        self.reset()

    def reset(self) -> None:
        self._err_sum = 0.0
        self._prev_err = 0.0

    def apply(self, control, target_v: float, cur_v: float) -> None:
        err = target_v - cur_v
        self._err_sum += err * self._dt
        d = err - self._prev_err
        self._prev_err = err
        out = self._kp * err + self._ki * self._err_sum + self._kd * d
        if out >= 0:
            control.throttle = min(self._max_throttle, out)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(self._max_brake, -out)


def _speed_of(actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def _yaw_error(from_yaw: float, to_x: float, to_y: float,
               loc) -> float:
    """Signed heading error, in radians, from ``loc`` toward ``(to_x, to_y)``."""
    err = math.atan2(to_y - loc.y, to_x - loc.x) - from_yaw
    return (err + math.pi) % (2 * math.pi) - math.pi


def _is_driving(wp) -> bool:
    """Is this waypoint on a lane a vehicle may change into?

    CARLA hands back shoulders and parking bays from
    ``get_left_lane()``/``get_right_lane()`` -- on the Town04 highway the lane
    left of the innermost one is the median shoulder -- so a lane change has
    to filter by type. The local simulator synthesises driving lanes only and
    reports so, which keeps this the same test on both backends.
    """
    lane_type = getattr(wp, "lane_type", None)
    if lane_type is None:
        return True
    return str(lane_type).split(".")[-1].lower() == "driving"


class WaypointFollowerLite(py_trees.behaviour.Behaviour):
    """PID longitudinal control + pure-pursuit steering along the lane graph.

    ``keep_lane`` implements the spatial modifier of the same name (Table 2,
    "Spatial Modifiers: ... position, lane, keep_lane, and change_lane").
    Without it, steering re-projects the actor onto whichever lane is nearest
    on every tick, so a vehicle that leaves a junction carrying lateral error
    can be captured by the neighbouring lane and stay there. With it, the leaf
    latches the first non-junction lane it sees and steers toward that lane's
    centreline for as long as it runs, crossing junctions unchanged.
    """

    #: how many lateral steps to take when pulling back to the locked lane
    MAX_LANE_CORRECTION = 3

    def __init__(self, actor_handle, v_setpoint, ctx,
                 name="DriveLite", lookahead=5.0,
                 kp=0.6, ki=0.05, kd=0.1,
                 max_throttle=1.0, max_brake=1.0,
                 keep_lane=False):
        super().__init__(name=name)
        self._actor = actor_handle
        self._v_set = v_setpoint
        self._ctx = ctx
        self._lookahead = lookahead
        self._pid = _SpeedPID(kp=kp, ki=ki, kd=kd,
                              max_throttle=max_throttle, max_brake=max_brake)
        self._keep_lane = keep_lane
        self._locked_lane = None
        self._left_junction = False

    def initialise(self):
        self._locked_lane = None
        self._left_junction = False

    def update(self):
        actor = _carla_actor(self._actor)
        if actor is None or not carla:
            return py_trees.common.Status.RUNNING
        target_v = _wrap_speed(self._v_set, self._ctx)
        control = carla.VehicleControl()
        self._pid.apply(control, target_v, _speed_of(actor))
        control.steer = self._compute_steer(actor)
        actor.apply_control(control)
        return py_trees.common.Status.RUNNING

    def _compute_steer(self, actor) -> float:
        if not carla:
            return 0.0
        world = self._ctx.world
        if world is None:
            return 0.0
        carla_map = self._ctx.carla_map
        loc = actor.get_location()
        wp = carla_map.get_waypoint(loc, project_to_road=True)
        if wp is None:
            return 0.0
        wp = self._hold_lane(wp)
        next_wps = wp.next(self._lookahead)
        if not next_wps:
            return 0.0
        target = next_wps[0].transform.location
        yaw = math.radians(actor.get_transform().rotation.yaw)
        err = _yaw_error(yaw, target.x, target.y, loc)
        return max(-1.0, min(1.0, err))


    def _hold_lane(self, wp):
        """Pull the steering reference back onto the locked lane.

        Latching is lazy: a leaf that starts while the actor is inside a
        junction has no lane to hold yet, so it takes the first ordinary lane
        it reaches. Correction only ever moves sideways within one
        carriageway -- lane ids carry the side of the road in their sign, so a
        mismatched sign means the actor is somewhere this modifier has no
        opinion about.
        """
        if not self._keep_lane:
            return wp
        if wp.is_junction:
            # Inside a junction there is no lane to hold; note that the next
            # ordinary lane is a new one to commit to.
            self._left_junction = True
            return wp
        if self._locked_lane is None or self._left_junction:
            # Latch lazily, and re-latch on the way out of a junction: the
            # connector decides which lane the vehicle emerges in, and that
            # is the lane it should then hold.
            self._locked_lane = wp.lane_id
            self._left_junction = False
            return wp
        target = self._locked_lane
        if wp.lane_id == target or (wp.lane_id > 0) != (target > 0):
            return wp
        for _ in range(self.MAX_LANE_CORRECTION):
            nxt = wp.get_left_lane() if abs(wp.lane_id) > abs(target) \
                else wp.get_right_lane()
            if nxt is None or nxt.is_junction:
                break
            wp = nxt
            if wp.lane_id == target:
                break
        return wp


@register("vehicle.drive")
def _build_drive(actor_handle, args, modifiers, ctx):
    v_setpoint: Any = _speed_modifier(modifiers, ctx)
    if v_setpoint is None:
        v_setpoint = 0.0
    keep_lane = any(m.name == "keep_lane" for m in modifiers)
    suffix = "+keep_lane" if keep_lane else ""
    name = f"Drive[{actor_handle._binding}{suffix}]"
    return WaypointFollowerLite(actor_handle, v_setpoint, ctx, name=name,
                                keep_lane=keep_lane)


class ChangeTargetSpeed(py_trees.behaviour.Behaviour):
    def __init__(self, actor_handle, target_speed, ctx,
                 profile="smooth", name="ChangeTargetSpeed"):
        super().__init__(name=name)
        self._actor = actor_handle
        self._target = target_speed
        self._ctx = ctx
        self._profile = profile

    def update(self):
        actor = _carla_actor(self._actor)
        if actor is None or not carla:
            return py_trees.common.Status.SUCCESS
        v = actor.get_velocity()
        cur = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        err = self._target - cur
        control = carla.VehicleControl()
        if self._profile == "asap":
            if err >= 0:
                control.throttle = 1.0
            else:
                control.brake = 1.0
        else:
            gain = 0.4
            out = gain * err
            if out >= 0:
                control.throttle = min(0.7, out)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = min(0.7, -out)
        actor.apply_control(control)
        if abs(err) < 0.3:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


@register("vehicle.change_speed")
def _build_change_speed(actor_handle, args, modifiers, ctx):
    target = ctx.eval(args.named["target"]) if "target" in args.named else 0.0
    profile_arg = args.named.get("rate_profile")
    profile = ctx.eval(profile_arg) if profile_arg is not None else "smooth"
    if isinstance(profile, str) and profile.lower() == "asap":
        profile = "asap"
    elif isinstance(profile, str):
        profile = "smooth"
    name = f"ChangeSpeed[{actor_handle._binding}->{_value(target):.1f}]"
    return ChangeTargetSpeed(actor_handle, _value(target), ctx, profile=str(profile), name=name)


class LaneChangeLite(py_trees.behaviour.Behaviour):
    """Steer into a neighbouring lane, then hold it for a short run-out.

    The target lane is resolved once, in ``initialise``, by stepping
    ``num_of_lanes`` neighbours toward ``side``; from then on the leaf steers
    at a lookahead point on that lane and reports SUCCESS after
    :attr:`COMPLETION_DISTANCE` metres of travel on it.

    Three properties of the manoeuvre drive that shape:

    * On an undivided road the lane to the left of the innermost one carries
      *oncoming* traffic -- which is the whole content of an overtake. Walking
      toward the target by lane-id arithmetic cannot express that, because the
      ids change sign across the centre line; the walk follows the requested
      side instead. For the same reason the lookahead is taken along the
      **actor's** direction of travel, which on that lane is its
      ``previous()``, not its ``next()``, and ``side`` is read in the actor's
      frame rather than the lane's (see :meth:`_neighbour`).
    * Steering at a lookahead point rather than at the abeam projection makes
      the path a blend rather than a right-angle sidestep, and leaves nothing
      to jitter once the actor is on the new lane and the two coincide.
    * The manoeuvre is not over until the actor is *aligned* with the new
      lane, not merely on it. A leaf that hands over mid-yaw leaves the car
      crossing the lane it was supposed to have taken, and whatever action
      runs next inherits the drift -- ``change_speed`` in particular, which
      commands the wheels straight.

    With a ``speed(...)`` modifier the leaf holds that speed with the same PID
    ``drive()`` uses; without one it falls back to a fixed throttle, whose
    distance-to-complete then depends on the slope of the road.
    """

    #: metres of travel on the new lane before the manoeuvre counts as done
    COMPLETION_DISTANCE = 8.0
    #: and how nearly parallel to it the actor has to be by then
    COMPLETION_YAW = math.radians(3.0)
    #: hard bound on waiting for that: succeed anyway after this much travel
    MAX_SETTLE_DISTANCE = 30.0
    #: how far ahead on the target lane to aim
    LOOKAHEAD = 6.0
    #: bound on the neighbour walk, so a bad target cannot spin forever
    MAX_STEPS = 4

    def __init__(self, actor_handle, num_of_lanes, side, ctx, name="LaneChange",
                 v_setpoint=None, throttle=0.5, steer_gain=0.8):
        super().__init__(name=name)
        self._actor = actor_handle
        self._num = max(1, int(num_of_lanes))
        self._side = "left" if str(side).lower() == "left" else "right"
        self._ctx = ctx
        self._v_set = v_setpoint
        self._throttle = throttle
        self._steer_gain = steer_gain
        self._pid = _SpeedPID()
        self._target_lane_id: Optional[int] = None
        self._distance_on_new_lane = 0.0
        self._last_loc = None

    def initialise(self):
        self._target_lane_id = None
        self._distance_on_new_lane = 0.0
        self._pid.reset()
        actor = _carla_actor(self._actor)
        if actor is None or self._ctx.carla_map is None:
            return
        wp = self._ctx.carla_map.get_waypoint(actor.get_location(),
                                              project_to_road=True)
        if wp is None:
            return
        target = self._step_aside(wp, actor)
        if target is not None:
            self._target_lane_id = target.lane_id
        self._last_loc = actor.get_location()

    def _neighbour(self, wp, actor):
        """The lane next to ``wp`` on the requested side, in the actor's frame.

        ``get_left_lane()`` / ``get_right_lane()`` are relative to the lane's
        own direction of travel. For a vehicle running *against* its lane --
        the second half of an overtake, where the car is in the oncoming lane
        and wants to pull back in to its right -- the two are swapped. The
        ``side:`` argument of a manoeuvre is the driver's side, so the sense is
        taken from the actor's heading rather than from the lane's.
        """
        yaw = math.radians(actor.get_transform().rotation.yaw)
        lane_yaw = math.radians(wp.transform.rotation.yaw)
        delta = abs((lane_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
        against = delta > math.pi / 2
        want_left = (self._side == "left") != against
        return wp.get_left_lane() if want_left else wp.get_right_lane()

    def _step_aside(self, wp, actor):
        """``num_of_lanes`` neighbours toward ``side``, or None if there is none.

        Stops early at the edge of the drivable carriageway and returns the
        furthest lane actually reached, so a two-lane request on a road with
        one lane to spare still changes lane once.
        """
        out = None
        cur = wp
        for _ in range(self._num):
            nxt = self._neighbour(cur, actor)
            if nxt is None or not _is_driving(nxt):
                break
            out = cur = nxt
        return out

    def _target_waypoint(self, wp_cur, actor):
        """The waypoint on the target lane abeam the actor, or None."""
        cur = wp_cur
        for _ in range(self.MAX_STEPS):
            if cur.lane_id == self._target_lane_id:
                return cur
            nxt = self._neighbour(cur, actor)
            if nxt is None:
                return None
            cur = nxt
        return None

    @staticmethod
    def _lane_yaw_error(actor, wp) -> float:
        """How far off ``wp``'s lane line the actor is pointing, in radians.

        Parallel and anti-parallel both count as aligned: a vehicle passing in
        the oncoming lane is lined up with that lane while running against it.
        """
        yaw = math.radians(actor.get_transform().rotation.yaw)
        lane_yaw = math.radians(wp.transform.rotation.yaw)
        delta = abs((lane_yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
        return min(delta, math.pi - delta)

    def _aim_location(self, actor, wp_cur):
        """Where to steer: a point LOOKAHEAD metres along the target lane."""
        target_wp = self._target_waypoint(wp_cur, actor) or wp_cur
        yaw = math.radians(actor.get_transform().rotation.yaw)
        lane_yaw = math.radians(target_wp.transform.rotation.yaw)
        delta = (lane_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        # A change into oncoming traffic runs against the target lane's own
        # direction, so "ahead" on it is previous(), not next().
        steps = target_wp.previous(self.LOOKAHEAD) if abs(delta) > math.pi / 2 \
            else target_wp.next(self.LOOKAHEAD)
        return (steps[0] if steps else target_wp).transform.location

    def update(self):
        actor = _carla_actor(self._actor)
        if actor is None or not carla or self._ctx.carla_map is None:
            return py_trees.common.Status.SUCCESS
        loc = actor.get_location()
        wp_cur = self._ctx.carla_map.get_waypoint(loc, project_to_road=True)
        if wp_cur is None or self._target_lane_id is None:
            # No lane to move into: the manoeuvre is not available here, and a
            # leaf that never succeeds would stall the serial block it is in.
            return py_trees.common.Status.SUCCESS
        if self._last_loc is not None and wp_cur.lane_id == self._target_lane_id:
            self._distance_on_new_lane += math.sqrt(
                (loc.x - self._last_loc.x) ** 2 + (loc.y - self._last_loc.y) ** 2
            )
        self._last_loc = loc
        # Done when the actor is on the new lane *and* pointing along it. Only
        # the second half makes the manoeuvre composable: a leaf that hands
        # over while the car is still yawed leaves it crossing the lane it was
        # supposed to have taken, and whatever runs next inherits the drift.
        if self._distance_on_new_lane > self.COMPLETION_DISTANCE and (
                self._lane_yaw_error(actor, wp_cur) < self.COMPLETION_YAW
                or self._distance_on_new_lane > self.MAX_SETTLE_DISTANCE):
            return py_trees.common.Status.SUCCESS
        aim = self._aim_location(actor, wp_cur)
        yaw = math.radians(actor.get_transform().rotation.yaw)
        control = carla.VehicleControl()
        if self._v_set is None:
            control.throttle = self._throttle
        else:
            self._pid.apply(control, _wrap_speed(self._v_set, self._ctx),
                            _speed_of(actor))
        control.steer = max(-1.0, min(
            1.0, self._steer_gain * _yaw_error(yaw, aim.x, aim.y, loc)))
        actor.apply_control(control)
        return py_trees.common.Status.RUNNING


@register("vehicle.change_lane")
def _build_change_lane(actor_handle, args, modifiers, ctx):
    n = ctx.eval(args.named.get("num_of_lanes", None)) if "num_of_lanes" in args.named else 1
    side = ctx.eval(args.named.get("side", None)) if "side" in args.named else "right"
    if isinstance(n, Quantity):
        n = n.value
    return LaneChangeLite(actor_handle, int(n or 1), str(side or "right"), ctx,
                          v_setpoint=_speed_modifier(modifiers, ctx),
                          name=f"LaneChange[{actor_handle._binding}->{side}]")


class _Noop(py_trees.behaviour.Behaviour):
    def __init__(self, name="Noop"):
        super().__init__(name=name)

    def update(self):
        return py_trees.common.Status.SUCCESS


@register("vehicle.assign_position")
def _build_assign_position(actor_handle, args, modifiers, ctx):
    return _Noop(name=f"AssignPosition[{actor_handle._binding}]")


@register("stationary_object.assign_position")
def _build_assign_position_static(actor_handle, args, modifiers, ctx):
    return _Noop(name=f"AssignPositionStatic[{actor_handle._binding}]")


class SetLights(py_trees.behaviour.Behaviour):
    def __init__(self, actor_handle, mode, name="SetLights"):
        super().__init__(name=name)
        self._actor = actor_handle
        self._mode = mode

    def update(self):
        actor = _carla_actor(self._actor)
        if actor is None or not carla:
            return py_trees.common.Status.SUCCESS
        states = carla.VehicleLightState
        flag = states.NONE
        m = self._mode.lower()
        if m == "low_beam":
            flag = states.LowBeam | states.Position
        elif m == "high_beam":
            flag = states.HighBeam | states.LowBeam | states.Position
        elif m == "fog":
            flag = states.Fog | states.LowBeam | states.Position
        elif m == "brake":
            flag = states.Brake
        elif m == "auto":
            flag = states.LowBeam | states.Position
        try:
            actor.set_light_state(carla.VehicleLightState(int(flag)))
        except Exception:
            pass
        return py_trees.common.Status.SUCCESS


@register("vehicle.set_lights")
def _build_set_lights(actor_handle, args, modifiers, ctx):
    mode = ctx.eval(args.named["mode"]) if "mode" in args.named else "auto"
    return SetLights(actor_handle, str(mode), name=f"SetLights[{actor_handle._binding}->{mode}]")


class AssignCelestial(py_trees.behaviour.Behaviour):
    def __init__(self, ctx, azimuth, elevation):
        super().__init__(name=f"Celestial[az={azimuth},el={elevation}]")
        self._ctx = ctx
        self._azimuth = azimuth
        self._elevation = elevation

    def update(self):
        if self._ctx.world is None or not carla:
            return py_trees.common.Status.SUCCESS
        weather = self._ctx.world.get_weather()
        weather.sun_azimuth_angle = float(self._azimuth)
        weather.sun_altitude_angle = float(self._elevation)
        self._ctx.world.set_weather(weather)
        return py_trees.common.Status.SUCCESS


class RamTarget(py_trees.behaviour.Behaviour):
    """Drive full-throttle toward another actor's current location.

    Mirrors ``drive_npc_at_ego`` from scripts/record_scenario_collision.py:
    each tick, steer toward the target and slam the throttle.
    """

    def __init__(self, actor_handle, target_handle, ctx,
                 name="Ram", throttle=1.0, steer_gain=1.5):
        super().__init__(name=name)
        self._actor = actor_handle
        self._target = target_handle
        self._ctx = ctx
        self._throttle = throttle
        self._steer_gain = steer_gain

    def update(self):
        me = _carla_actor(self._actor)
        them = _carla_actor(self._target)
        if me is None or them is None or not carla:
            return py_trees.common.Status.RUNNING
        my_tf = me.get_transform()
        their_loc = them.get_location()
        to_x = their_loc.x - my_tf.location.x
        to_y = their_loc.y - my_tf.location.y
        yaw_rad = math.radians(my_tf.rotation.yaw)
        fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
        cross_z = fx * to_y - fy * to_x
        dot = fx * to_x + fy * to_y
        angle = math.atan2(cross_z, dot)
        steer = max(-1.0, min(1.0, angle * self._steer_gain))
        me.apply_control(carla.VehicleControl(throttle=self._throttle,
                                              steer=steer, brake=0.0))
        return py_trees.common.Status.RUNNING


@register("vehicle.ram")
def _build_ram(actor_handle, args, modifiers, ctx):
    target_expr = args.named.get("target")
    target_handle = None
    if target_expr is not None:
        target_val = ctx.eval(target_expr)
        if isinstance(target_val, _ActorHandle):
            target_handle = target_val
        elif isinstance(target_val, str):
            target_handle = _ActorHandle(ctx, target_val)
    return RamTarget(actor_handle, target_handle, ctx,
                     name=f"Ram[{actor_handle._binding}->{getattr(target_handle, '_binding', '?')}]")


@register("environment.assign_celestial_position")
def _build_celestial(actor_handle, args, modifiers, ctx):
    az = _value(ctx.eval(args.named["azimuth"]))
    el = _value(ctx.eval(args.named["elevation"]))
    az_deg = math.degrees(az)
    el_deg = math.degrees(el)
    return AssignCelestial(ctx, az_deg, el_deg)
