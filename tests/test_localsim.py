"""Checks for the local simulator backend. Needs no CARLA and no display.

Run directly:  python tests/test_localsim.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osc2carla.localsim import Client, load_town, town_names
from osc2carla.localsim.api import (Location, Rotation, Transform,
                                    VehicleControl)
from osc2carla.localsim.collision import obb_overlap
from osc2carla.localsim.geometry import corners_2d

failures = []


def check(label, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {label}" + (f"   {detail}" if detail else ""))
    if not condition:
        failures.append(label)


# --------------------------------------------------------------------------
# 1. road network
# --------------------------------------------------------------------------
print("--- road network ---")
for name in town_names():
    town = load_town(name)
    dead_ends = [l for l in town.lanes.values() if not l.successors]
    check(f"{name}: every lane continues somewhere", not dead_ends,
          f"{len(town.lanes)} lanes, {len(dead_ends)} dead ends")
    check(f"{name}: has spawn points", len(town.get_spawn_points()) > 0,
          f"{len(town.get_spawn_points())} points")

town = load_town("grid")

# A point 40 m along the y = 0 road projects onto the eastbound inner lane.
wp = town.get_waypoint(Location(40.0, 1.75, 0.0))
check("get_waypoint lands on the nearest lane", wp is not None and not wp.is_junction,
      repr(wp))
check("waypoint heading matches the lane direction",
      abs(wp.transform.rotation.yaw) < 1e-6, f"yaw={wp.transform.rotation.yaw}")

# A point well off the network still projects, because project_to_road=True.
off = town.get_waypoint(Location(40.0, 24.0, 0.0), project_to_road=True)
check("off-network points project onto the road", off is not None, repr(off))

# next()/previous() are inverses along a straight lane.
ahead = wp.next(20.0)
check("next() returns a continuation", len(ahead) == 1, repr(ahead))
back = ahead[0].previous(20.0)
check("previous() undoes next() on a straight lane",
      abs(back[0].transform.location.x - wp.transform.location.x) < 1e-6,
      f"{back[0].transform.location.x:.3f} vs {wp.transform.location.x:.3f}")

# Crossing a junction: 200 m ahead of x = 40 is past two junctions.
far = wp
for _ in range(40):
    step = far.next(5.0)
    assert step, "network dead-ended"
    far = step[0]
check("a vehicle can drive 200 m without leaving the network", True,
      repr(far))

# Lane neighbours exist inside a carriageway and stop at its edge.
right = wp.get_right_lane()
check("inner lane has an outer neighbour", right is not None
      and abs(right.lane_id) == abs(wp.lane_id) + 1, repr(right))
check("outermost lane has no further neighbour",
      right is not None and right.get_right_lane() is None)

# --------------------------------------------------------------------------
# 2. turn classification
# --------------------------------------------------------------------------
print("\n--- junction manoeuvres ---")
# CARLA is left-handed with +y south, so a left turn *decreases* yaw.
straight_town = load_town("grid", turn_preference="straight")
left_town = load_town("grid", turn_preference="left")
# Approach junction (80, 80) eastbound: it is the one with all four arms,
# so a left exit (north, toward -y) actually exists there.
approach = straight_town.get_waypoint(Location(70.0, 81.75, 0.0))
approach_l = left_town.get_waypoint(Location(70.0, 81.75, 0.0))


def heading_after(start, distance=26.0):
    w = start
    travelled = 0.0
    while travelled < distance:
        nxt = w.next(2.0)
        if not nxt:
            break
        w = nxt[0]
        travelled += 2.0
    return w.transform.rotation.yaw


yaw_straight = heading_after(approach)
yaw_left = heading_after(approach_l)
check("--junction-turn straight keeps the heading", abs(yaw_straight) < 20.0,
      f"yaw={yaw_straight:.1f} deg")
check("--junction-turn left turns toward -y (CARLA's left)", yaw_left < -45.0,
      f"yaw={yaw_left:.1f} deg")

# --------------------------------------------------------------------------
# 3. vehicle dynamics
# --------------------------------------------------------------------------
print("\n--- vehicle dynamics ---")
world = Client().load_world("grid")
bp = world.get_blueprint_library().find("vehicle.tesla.model3")
bp.set_attribute("role_name", "hero")
car = world.spawn_actor(bp, Transform(Location(40.0, 1.75, 0.5), Rotation(yaw=0.0)))

car.apply_control(VehicleControl(throttle=1.0))
for _ in range(40):                       # 2 s
    world.tick()
check("full throttle accelerates", 4.0 < car.speed < 9.0,
      f"v={car.speed:.2f} m/s after 2 s")

car.apply_control(VehicleControl(brake=1.0))
for _ in range(60):                       # 3 s
    world.tick()
check("full brake stops the car", car.speed < 0.05, f"v={car.speed:.3f} m/s")

# Steering sign: positive steer must turn toward +y, as it does in CARLA.
before = car.get_transform().rotation.yaw
car.apply_control(VehicleControl(throttle=0.6, steer=0.5))
for _ in range(40):
    world.tick()
after = car.get_transform().rotation.yaw
check("positive steer increases yaw (turns toward +y)", after > before + 5.0,
      f"{before:.1f} -> {after:.1f} deg")

# --------------------------------------------------------------------------
# 4. collision detection and reporting
# --------------------------------------------------------------------------
print("\n--- collisions ---")
a = corners_2d(0.0, 0.0, 0.0, 2.4, 1.0)
check("boxes 3 m apart do not overlap",
      obb_overlap(a, corners_2d(6.0, 0.0, 0.0, 2.4, 1.0)) is None)
hit = obb_overlap(a, corners_2d(4.0, 0.0, 0.0, 2.4, 1.0))
check("overlapping boxes report a normal and a depth",
      hit is not None and hit[1] > 0.0, repr(hit))

world = Client().load_world("grid")
bl = world.get_blueprint_library()
struck = world.spawn_actor(bl.find("vehicle.tesla.model3"),
                           Transform(Location(40.0, 1.75, 0.5), Rotation(yaw=0.0)))
runner = world.spawn_actor(bl.find("vehicle.dodge.charger_2020"),
                           Transform(Location(20.0, 1.75, 0.5), Rotation(yaw=0.0)))
events = []
sensor = world.spawn_actor(bl.find("sensor.other.collision"), Transform(),
                           attach_to=struck)
sensor.listen(events.append)
runner.apply_control(VehicleControl(throttle=1.0))
for _ in range(200):
    world.tick()
check("a rear-end contact reaches the collision sensor", len(events) > 0,
      f"{len(events)} events")
check("the event names the other actor",
      bool(events) and events[0].other_actor.type_id == "vehicle.dodge.charger_2020")
check("the event carries a non-zero impulse",
      bool(events) and events[0].normal_impulse.length() > 0.0,
      f"|J|={events[0].normal_impulse.length():.0f}" if events else "")

# CARLA refuses a spawn into an occupied box; the initializer relies on it.
blocked = world.try_spawn_actor(bl.find("vehicle.audi.tt"),
                                struck.get_transform())
check("spawning into an occupied box fails", blocked is None)

# --------------------------------------------------------------------------
# 5. the backend indirection
# --------------------------------------------------------------------------
print("\n--- backend binding ---")
from osc2carla.backend import simapi

simapi.bind("pygame")
check("bind('pygame') makes the proxy live", bool(simapi.sim))
check("the proxy forwards module attributes",
      simapi.sim.VehicleControl(throttle=0.5).throttle == 0.5)
check("the proxy knows it is the local one", simapi.sim.is_local)

from osc2carla.backend import atomic_behaviors
check("behaviour modules see the same binding",
      bool(atomic_behaviors.carla) and atomic_behaviors.carla.is_local)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all local simulator checks passed")
