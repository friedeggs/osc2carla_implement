"""A run must not spawn its cast on top of the previous run's.

`load_world()` is skipped when the requested town is already loaded, and with it
the actor cleanup a map change gives for free. A run that dies between spawning
and teardown leaves its whole cast standing; the next run then puts a second ego
inside the first, physics pins both, and every metric is produced describing a
scenario that never happened. That is the dangerous kind of failure -- nothing
raises -- so it gets a test rather than a comment.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from osc2carla.backend.initializer import (LEFTOVER_PREFIXES,
                                           clear_leftover_actors)

ok = lambda label: print(f"[ok  ] {label}")


class FakeActor:
    def __init__(self, type_id, world=None):
        self.type_id = type_id
        self.destroyed = False
        self.stopped = False
        self._world = world

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True
        if self._world is not None:
            self._world.destroy_order.append(self.type_id)


class FakeWorld:
    def __init__(self, type_ids):
        self.destroy_order = []
        self.ticks = 0
        self.actors = [FakeActor(t, self) for t in type_ids]

    def get_actors(self):
        return list(self.actors)

    def tick(self):
        self.ticks += 1


# The town's own furniture, and one run's leftover cast.
MAP_FURNITURE = ["traffic.traffic_light", "traffic.speed_limit.30",
                 "static.trigger.friction", "spectator"]
LEFTOVERS = ["vehicle.tesla.model3", "vehicle.audi.a2", "walker.pedestrian.0001",
             "sensor.camera.rgb", "sensor.other.collision",
             "static.prop.dirtdebris01"]

world = FakeWorld(MAP_FURNITURE + LEFTOVERS)
cleared = clear_leftover_actors(world)
assert cleared == len(LEFTOVERS), cleared
destroyed = {a.type_id for a in world.actors if a.destroyed}
assert destroyed == set(LEFTOVERS), destroyed
ok("every vehicle, walker, sensor and spawned prop is destroyed")

survivors = {a.type_id for a in world.actors if not a.destroyed}
assert survivors == set(MAP_FURNITURE), survivors
ok("the map's own traffic lights, signs and spectator are left alone")

# A sensor outlives its parent's destruction and keeps streaming into a callback
# with nowhere to put the data, so sensors go first.
sensors = [i for i, t in enumerate(world.destroy_order) if t.startswith("sensor.")]
others = [i for i, t in enumerate(world.destroy_order) if not t.startswith("sensor.")]
assert max(sensors) < min(others), world.destroy_order
ok("sensors are destroyed before the actors they are attached to")

assert world.ticks == 1
ok("the world is ticked once, so the destruction has taken effect")

assert clear_leftover_actors(FakeWorld(MAP_FURNITURE)) == 0
ok("a clean world is left completely alone, and reports nothing cleared")


class BrokenWorld:
    def get_actors(self):
        raise RuntimeError("no connection")


assert clear_leftover_actors(BrokenWorld()) == 0
ok("a world that cannot be listed is not a crash: clearing is best-effort")


class StubbornActor(FakeActor):
    def destroy(self):
        raise RuntimeError("already destroyed")


world = FakeWorld(["vehicle.audi.a2"])
world.actors.append(StubbornActor("vehicle.tesla.model3"))
assert clear_leftover_actors(world) == 1
ok("an actor that is already gone is the outcome wanted, not an error")

assert LEFTOVER_PREFIXES == ("vehicle.", "walker.", "sensor.", "static.prop.")
ok("the prefixes are exactly the categories no CARLA map ships")

print("\nall leftover-actor checks passed")
