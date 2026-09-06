"""Simulator-free checks of the ego sensor rig and the harness policy bridge.

No CARLA server and no checkpoint. What is exercised here is everything between
"a policy declares a rig" and "the policy is handed that rig's measurements for
the tick it is reasoning about", because that is the part where a mistake is
invisible: a mis-shaped image, a stale frame or a route on the wrong grid all
produce a policy that drives badly rather than a run that fails.

The simulator double binds the LOCAL simulator's datatypes -- real
``Transform``/``Location``/``Rotation``/``VehicleControl`` -- and fakes only the
sensor blueprints the local backend does not have. So the geometry under test is
the geometry the runtime uses.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scenario_orchestration"))

import numpy as np

from osc2carla.backend import simapi

simapi.bind("pygame")                       # real Transform/Location/Rotation

from osc2carla.backend.policy import (Command, EgoPolicy, ExternalEgoController,
                                      Leader, Observation, RoutePoint)
from osc2carla.backend.sensors import (CameraSpec, LidarSpec, RadarSpec,
                                       SensorRig, rig, specs_from)

ok = lambda label: print(f"[ok  ] {label}")


# ---------------------------------------------------------------------------
# The double
# ---------------------------------------------------------------------------

class FakeMeasurement:
    def __init__(self, frame, raw, height=0, width=0):
        self.frame = frame
        self.raw_data = raw
        self.height = height
        self.width = width


class FakeSensor:
    def __init__(self, spec):
        self.spec = spec
        self.listener = None
        self.stopped = False
        self.destroyed = False

    def listen(self, callback):
        self.listener = callback

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True


class FakeBlueprint:
    def __init__(self, bp_id):
        self.id = bp_id
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class FakeLibrary:
    #: What the double refuses to create, to exercise the partial-rig path.
    def __init__(self, missing=()):
        self.missing = set(missing)

    def find(self, bp_id):
        if bp_id in self.missing:
            raise IndexError(f"blueprint {bp_id} not found")
        return FakeBlueprint(bp_id)


class FakeSettings:
    fixed_delta_seconds = 0.05


class FakeSnapshot:
    def __init__(self, frame):
        self.frame = frame


class FakeWorld:
    def __init__(self, missing=()):
        self.library = FakeLibrary(missing)
        self.sensors = []
        self.frame = 100

    def get_blueprint_library(self):
        return self.library

    def get_settings(self):
        return FakeSettings()

    def get_snapshot(self):
        return FakeSnapshot(self.frame)

    def spawn_actor(self, blueprint, transform, attach_to=None):
        sensor = FakeSensor(blueprint.id)
        sensor.blueprint = blueprint
        self.sensors.append(sensor)
        return sensor


# ---------------------------------------------------------------------------
# 1. spec normalisation: exactly the dicts the two installed adapters return
# ---------------------------------------------------------------------------

simlingo_declared = [
    {"name": "rgb_front", "width": 1024, "height": 512, "fov": 110.0,
     "x": -1.5, "y": 0.0, "z": 2.0},
]
tfv6_declared = [
    {"name": "PCAM_L0", "width": 384, "height": 384, "fov": 60.0, "yaw": -57.5,
     "x": 0.0, "y": -0.3, "z": 2.25},
    {"name": "lidar", "kind": "sensor.lidar.ray_cast", "channels": 64,
     "range_m": 100.0, "rotation_frequency": 20.0, "x": 1.0, "y": 0.0, "z": 2.5},
    {"name": "radar1", "kind": "sensor.other.radar", "x": 2.6, "z": 0.60,
     "yaw": -45.0, "horizontal_fov": 90.0, "vertical_fov": 0.1,
     "range_m": 100.0, "points_per_second": 1500},
]

specs = specs_from(simlingo_declared)
assert len(specs) == 1 and isinstance(specs[0], CameraSpec), specs
assert (specs[0].width, specs[0].height, specs[0].fov) == (1024, 512, 110.0)
ok("a camera dict with no 'kind' normalises to a CameraSpec")

specs = specs_from(tfv6_declared)
assert isinstance(specs[0], CameraSpec) and specs[0].yaw == -57.5
assert isinstance(specs[1], LidarSpec) and specs[1].channels == 64
assert isinstance(specs[2], RadarSpec) and specs[2].horizontal_fov == 90.0
ok("'kind' selects the LiDAR and radar specs, and their fields survive")

# A key the spec does not define must not stop a rig from being attached.
specs = specs_from([{"name": "cam", "annotation_the_runner_never_heard_of": 1}])
assert specs[0].name == "cam"
ok("an unknown key in a declared sensor is ignored, not rejected")

assert [s.name for s in rig("tfv6")][:3] == ["PCAM_L0", "PCAM_F0", "PCAM_R0"]
assert [s.name for s in rig("tfv6")][4:] == ["radar1", "radar2", "radar3", "radar4"]
ok("the named tfv6 rig keeps the camera stitch order and the radar order")


# ---------------------------------------------------------------------------
# 2. spawn: retiming, partial attachment, teardown
# ---------------------------------------------------------------------------

world = FakeWorld()
r = SensorRig(world, ego_actor=object(), specs=specs_from(tfv6_declared)).spawn()
assert r.active and r.names == ["PCAM_L0", "lidar", "radar1"], r.names
assert r.sensor_hz == 20.0, r.sensor_hz          # 1 / 0.05
lidar = [s for s in r.specs if isinstance(s, LidarSpec)][0]
assert lidar.rotation_frequency == 20.0
assert lidar.points_per_second == 30000 * 20, lidar.points_per_second
ok("sweeping sensors are retimed to one revolution per simulation tick")

radar = [s for s in r.specs if isinstance(s, RadarSpec)][0]
assert radar.points_per_second == 150 * 20, radar.points_per_second
ok("the radar is retimed to deliver its per-tick budget every tick")

camera = world.sensors[0]
assert camera.blueprint.attributes["image_size_x"] == "384"
assert camera.blueprint.attributes["fov"] == "60.0"
ok("camera blueprint attributes come from the declared spec")

partial = SensorRig(FakeWorld(missing=["sensor.lidar.ray_cast"]), object(),
                    specs_from(tfv6_declared)).spawn()
assert partial.active and partial.names == ["PCAM_L0", "radar1"], partial.names
assert len(partial.failed) == 1 and partial.failed[0].startswith("lidar:")
ok("a sensor that cannot be created is recorded, and the rest still attach")

none_at_all = SensorRig(FakeWorld(missing=["sensor.camera.rgb",
                                           "sensor.lidar.ray_cast",
                                           "sensor.other.radar"]),
                        object(), specs_from(tfv6_declared)).spawn()
assert not none_at_all.active and len(none_at_all.failed) == 3
ok("a rig that attaches nothing reports it rather than raising")

r.destroy()
assert all(s.stopped and s.destroyed for s in world.sensors)
assert not r.active
ok("destroy() stops and destroys every attached sensor")


# ---------------------------------------------------------------------------
# 3. capture: frame synchronisation and the array conversions
# ---------------------------------------------------------------------------

def bgra(height, width, rgb):
    """A CARLA-shaped BGRA buffer of one flat colour."""
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[:, :, 0] = rgb[2]
    frame[:, :, 1] = rgb[1]
    frame[:, :, 2] = rgb[0]
    return frame.tobytes()


world = FakeWorld()
r = SensorRig(world, object(), specs_from(simlingo_declared)).spawn()
_spec, sensor, _q = r.attached[0]
# Two stale frames ahead of the one being asked for: a rig that kept the newest
# would be right by luck here, one that kept the OLDEST would silently hand the
# policy a frame from before the tick it is reasoning about.
sensor.listener(FakeMeasurement(98, bgra(512, 1024, (0, 0, 0)), 512, 1024))
sensor.listener(FakeMeasurement(99, bgra(512, 1024, (0, 0, 0)), 512, 1024))
sensor.listener(FakeMeasurement(100, bgra(512, 1024, (10, 20, 30)), 512, 1024))
captured = r.capture(frame=100)
image = captured["rgb_front"]
assert image.shape == (512, 1024, 3), image.shape
assert image.dtype == np.uint8
assert tuple(image[0, 0]) == (10, 20, 30), tuple(image[0, 0])
ok("capture returns the frame it asked for, as HxWx3 RGB, dropping older ones")

r2 = SensorRig(FakeWorld(), object(), specs_from(tfv6_declared)).spawn()
for spec, sensor, _q in r2.attached:
    if spec.name == "lidar":
        cloud = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.25]],
                         dtype=np.float32)
        sensor.listener(FakeMeasurement(7, cloud.tobytes()))
    elif spec.name == "radar1":
        # (velocity, azimuth, altitude, depth): one detection dead ahead of a
        # sensor yawed -45 deg and mounted 2.6 m forward.
        det = np.array([[-3.0, 0.0, 0.0, 10.0]], dtype=np.float32)
        sensor.listener(FakeMeasurement(7, det.tobytes()))
    else:
        sensor.listener(FakeMeasurement(7, bgra(384, 384, (1, 2, 3)), 384, 384))
captured = r2.capture(frame=7)

points = captured["lidar"]
assert points.shape == (2, 4) and points[1, 0] == 4.0, points
ok("LiDAR returns Nx4 (x, y, z, intensity)")

detections = captured["radar1"]
assert detections.shape == (1, 4), detections.shape
x, y, z, v = detections[0]
# 10 m along a boresight yawed -45 deg, from a mount at x=2.6, z=0.60.
assert abs(x - (2.6 + 10.0 * math.cos(math.radians(-45.0)))) < 1e-3, x
assert abs(y - (10.0 * math.sin(math.radians(-45.0)))) < 1e-3, y
assert abs(z - 0.60) < 1e-3, z
assert v == -3.0, v      # CARLA's own sign, untouched: negative is closing
ok("radar returns ego-frame (x, y, z, v), with CARLA's velocity sign kept")

panel = r2.panel(captured)
assert panel.shape == (384, 384, 3), panel.shape       # one camera in this rig
ok("the video panel stitches the rig's cameras and skips the range sensors")

# A sensor that delivered nothing must be ABSENT, not zero-filled: a model
# cannot tell an all-zero raster from a clear road.
empty = SensorRig(FakeWorld(), object(), specs_from(simlingo_declared)).spawn()
import osc2carla.backend.sensors as sensors_module
sensors_module.CAPTURE_TIMEOUT_S = 0.01
assert empty.capture(frame=1) == {}
assert empty.dropped == {"rgb_front": 1}
ok("a sensor that delivers nothing is absent from the capture, and counted")


# ---------------------------------------------------------------------------
# 4. the ego-frame route
# ---------------------------------------------------------------------------

# Ego at (10, 5) heading +90 deg (CARLA's +y). A point 3 m further along +y is
# 3 m AHEAD; a point 2 m along +x is 2 m to the ego's LEFT, i.e. y = -2.
obs = Observation(t=0.0, speed=0.0, x=10.0, y=5.0, heading=math.radians(90.0),
                  route=[RoutePoint(10.0, 8.0, 0.0, 3.0),
                         RoutePoint(12.0, 5.0, 0.0, 2.0)])
route = obs.route_ego()
assert abs(route[0][0] - 3.0) < 1e-9 and abs(route[0][1]) < 1e-9, route[0]
assert abs(route[1][0]) < 1e-9 and abs(route[1][1] + 2.0) < 1e-9, route[1]
ok("route_ego() puts +x forward and +y right, in CARLA's handedness")


# ---------------------------------------------------------------------------
# 5. the harness bridge
# ---------------------------------------------------------------------------

import osc2carla_policy_bridge as bridge

# Both installed sensorimotor adapters declare `control` and return a nested
# control block beside their waypoints, so the block has to be read for either
# declared action space -- reading it only for `waypoints` rejected them as
# carrying no control at all.
cmd = bridge._to_command({"control": {"throttle": 0.4, "steer": -0.2, "brake": 0.0},
                          "waypoints": [[1, 2]], "meta": {"policy": "x"}})
assert (cmd.throttle, cmd.steer, cmd.brake) == (0.4, -0.2, 0.0), cmd
cmd = bridge._to_command({"control": {"throttle": 0.4, "steer": -0.2, "brake": 0.0}},
                         action_space="waypoints")
assert cmd.throttle == 0.4, cmd
ok("a nested 'control' block is read whichever action space is declared")

cmd = bridge._to_command({"throttle": 1.0, "steer": 0.0, "brake": 0.0})
assert cmd.throttle == 1.0
cmd = bridge._to_command({"acceleration_mps2": -5.0})
assert cmd.brake == 1.0 and cmd.throttle == 0.0, cmd
ok("the flat form and a bare acceleration are still accepted")

try:
    bridge._to_command({"waypoints": [[1, 2]]}, action_space="waypoints")
except bridge.PolicyBridgeError as exc:
    assert "no 'control' block" in str(exc), exc
else:
    raise AssertionError("a waypoint policy with no control must be refused")
ok("a waypoint policy that returns no control is refused, not given a follower")

payload = bridge._observation_payload(
    Observation(t=1.5, speed=7.0, x=0.0, y=0.0, heading=0.0,
                route=[RoutePoint(float(s), 0.0, 0.0, float(s))
                       for s in range(2, 62, 2)],
                leader=Leader(gap=12.0, speed=6.0, actor_id=3, type_id="vehicle.a"),
                sensors={"rgb_front": np.zeros((4, 4, 3), np.uint8)},
                frame=42))
assert payload["speed_mps"] == 7.0
assert payload["sensor"]["cameras"]["rgb_front"].shape == (4, 4, 3)
assert payload["sensor"]["frame"] == 42
assert payload["leader"]["gap_m"] == 12.0
# `route` is the harness contract's ego-frame list and belongs to the
# object-centric builder; this repository's own world-frame samples keep their
# own key rather than colliding with it.
assert "route" not in payload and len(payload["route_world"]) == 30
ok("the observation payload carries the car-following view and every sensor")

state_only = bridge._observation_payload(
    Observation(t=0.0, speed=0.0, x=0.0, y=0.0, heading=0.0))
assert "sensor" not in state_only
ok("a state-only observation carries no sensor block at all")


# ---------------------------------------------------------------------------
# 6. the controller: rig hand-off and the decision rate
# ---------------------------------------------------------------------------

class RecordingPolicy(EgoPolicy):
    """Declares the simlingo rig and records what it is handed."""
    name = "recording"

    def __init__(self):
        self.seen = []

    def sensors(self):
        return list(simlingo_declared)

    def act(self, obs):
        self.seen.append(obs)
        return Command(throttle=0.5, steer=0.1)


class FakeActor:
    id = 1
    type_id = "vehicle.tesla.model3"

    def __init__(self):
        self.controls = []

    def get_transform(self):
        return simapi.sim.Transform(simapi.sim.Location(0.0, 0.0, 0.0),
                                    simapi.sim.Rotation(0.0, 0.0, 0.0))

    def get_velocity(self):
        return simapi.sim.Vector3D(3.0, 0.0, 0.0)

    def get_location(self):
        return simapi.sim.Location(0.0, 0.0, 0.0)

    def get_speed_limit(self):
        return 50.0

    def apply_control(self, control):
        self.controls.append(control)


class FakeContext:
    def __init__(self, actor):
        self._actor = actor

    def actor(self, binding):
        return self._actor


actor = FakeActor()
world = FakeWorld()
policy = RecordingPolicy()
controller = ExternalEgoController(world, carla_map=None, ctx=FakeContext(actor),
                                   binding="ego", policy=policy, decision_hz=2.0)
assert controller.rig is not None and controller.rig.active
assert controller.rig.names == ["rgb_front"]
ok("the controller attaches the rig the policy declared, before setup")

sensors_module.CAPTURE_TIMEOUT_S = 20.0


def feed(frame):
    world.frame = frame
    for _spec, sensor, _q in controller.rig.attached:
        sensor.listener(FakeMeasurement(frame, bgra(512, 1024, (7, 7, 7)),
                                        512, 1024))


feed(101)
controller.tick(0.0)
assert len(policy.seen) == 1
assert policy.seen[0].sensors["rgb_front"].shape == (512, 1024, 3)
assert policy.seen[0].frame == 101
ok("the observation carries this tick's frames, stamped with this tick")

# 2 Hz against 0.1 s steps: decide, hold four, decide again.
held = [controller.tick(0.1 * i) for i in range(1, 5)]
assert controller.decisions == 1, controller.decisions
assert len(actor.controls) == 5      # the held command is still applied
feed(102)
fresh = controller.tick(0.5)
assert controller.decisions == 2, controller.decisions
ok("decision_hz holds the last command between decisions and still actuates")

# A held tick is still a measured tick. The caller reads leader.gap off what
# tick() returns for the run summary, and sampling that at the policy's
# decision rate would report a closest approach nothing looked for.
assert [round(o.t, 3) for o in held] == [0.1, 0.2, 0.3, 0.4]
assert all(o.sensors == {} for o in held), "a held tick must not render"
assert controller.last_observation.t == fresh.t == 0.5
ok("a held tick reports this tick's state and renders nothing for it")

detail = controller.describe()
assert detail["observation_space"] == "state+sensor"
assert detail["sensor_rig"]["attached"] == ["rgb_front"]
assert detail["decisions"] == 2 and detail["control_steps"] == 6
ok("describe() reports the rig that was attached and what it delivered")


class BlindPolicy(EgoPolicy):
    name = "blind"

    def act(self, obs):
        assert obs.sensors == {}
        return Command()


blind = ExternalEgoController(FakeWorld(), None, FakeContext(FakeActor()),
                              "ego", BlindPolicy())
assert blind.rig is None
blind.tick(0.0)
assert blind.describe()["observation_space"] == "state"
ok("a policy that declares no rig is unchanged: no sensors, no rig, no cost")

print("\nall sensor rig and policy bridge checks passed")
