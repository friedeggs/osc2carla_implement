"""The ego's sensor rig: what a vision-based policy actually sees.

``capabilities.json`` used to declare ``observation_spaces: ["state"]``, and the
reason it gave was exact: *"No sensor stream is rendered for the policy, so
sensorimotor policies (transfuser, tfv6, simlingo) and waypoint/trajectory
action spaces are not supported."* This module is the missing half. Everything
else on the external-policy seam -- :class:`~.policy.ExternalEgoController`, the
``--ego-policy`` entry point, the harness bridge -- already existed and was
already exercised by the IDM arm.

Who decides what to attach
--------------------------
The policy does, and that is not a stylistic preference. A sensorimotor policy
is trained behind one specific rig: TFv6 reads three 384x384 pinhole cameras
stitched into a 1152x384 strip plus a LiDAR raster and four radars, SimLingo
reads a single wide forward camera it tiles itself. A rig chosen *here* would be
a rig neither model was trained on, and the resulting numbers would be this
file's behaviour published under the model's name.

So an ``EgoPolicy`` may expose ``sensors()`` returning a list of specs -- or of
plain dicts, so a policy repository never has to import this module to describe
its own rig -- and :class:`SensorRig` attaches exactly that. A policy that
exposes neither ``sensors()`` nor ``camera_rig`` gets no rig and the ``state``
observation it always got, so every arm that worked before this module existed
is untouched.

Synchronous capture
-------------------
CARLA delivers sensor data asynchronously even in synchronous mode: the callback
fires somewhere between ``world.tick()`` returning and the next tick. A rig that
simply kept the newest frame would hand the policy an image from an arbitrary
earlier tick under load, which is the classic way an agent's behaviour becomes
irreproducible without anything appearing to go wrong. Each sensor therefore has
its own queue and :meth:`SensorRig.capture` blocks until the measurement
*stamped with the frame it asked for* arrives, discarding anything older. A
rendered tick then costs real wall-clock time, which is the same trade the CARLA
Leaderboard makes.

The local backend
-----------------
``osc2carla/localsim`` provides ``sensor.other.collision`` and nothing else, so
every spec fails to spawn there and lands in :attr:`SensorRig.failed`. That is
deliberate rather than tolerated: the rig must not raise in a process that is
only being asked to run the analytic arm. Refusing a *sensor* policy on that
backend is a decision about the experiment, so it is made in
``scenario_orchestration/run.py`` where the request is read, not here.
"""
from __future__ import annotations

import os
import queue
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .simapi import sim as carla

try:                                    # numpy is optional at import time so
    import numpy as np                  # this module can be imported (and its
except Exception:                       # spec tables read) without it
    np = None  # type: ignore


#: How long :meth:`SensorRig.capture` waits for one sensor's measurement before
#: giving up on it for this tick. Generous on purpose: a 1152x384 render on a
#: busy GPU is not instant, and a timeout here surfaces as a policy driving
#: blind, which is a worse outcome than a slow tick.
CAPTURE_TIMEOUT_S = 20.0

#: Beyond this many stale measurements on one queue, the sensor is stuck rather
#: than lagging and the newest frame is better than none.
STALE_LIMIT = 1000


# ---------------------------------------------------------------------------
# Specs: one dataclass per sensor kind, in CARLA blueprint units
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraSpec:
    """One RGB camera.

    ``x``/``y``/``z`` are metres in the ego's own frame (+x forward, +y right,
    +z up) and ``roll``/``pitch``/``yaw`` are degrees, i.e. exactly a
    ``carla.Transform`` relative to the vehicle. The defaults are the CARLA
    Leaderboard's forward camera mounting, which is what this family of models
    is trained behind.
    """
    name: str
    width: int = 1024
    height: int = 512
    fov: float = 110.0
    x: float = -1.5
    y: float = 0.0
    z: float = 2.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    kind: str = "sensor.camera.rgb"

    def to_transform(self):
        return carla.Transform(
            carla.Location(x=float(self.x), y=float(self.y), z=float(self.z)),
            carla.Rotation(roll=float(self.roll), pitch=float(self.pitch),
                           yaw=float(self.yaw)))

    def attributes(self) -> Dict[str, str]:
        return {"image_size_x": str(int(self.width)),
                "image_size_y": str(int(self.height)),
                "fov": str(float(self.fov))}


@dataclass(frozen=True)
class LidarSpec:
    """One ray-cast LiDAR.

    A camera-only rig would starve TFv6: it is a camera+LiDAR fusion model whose
    BEV branch reads ``rasterized_lidar``. Handing it an all-zero raster is not
    a malformed input -- it is a well-formed one meaning "the sensor works and
    nothing is out there", which is worse, because nothing downstream can tell
    it apart from a clear road.
    """
    name: str = "lidar"
    channels: int = 64
    range_m: float = 100.0
    #: Points per REVOLUTION is what matters to a model; CARLA is configured in
    #: points per second, so :meth:`for_tick_rate` scales this by the rotation
    #: rate it ends up setting.
    points_per_revolution: int = 30000
    points_per_second: int = 600000
    #: Revolutions per second. This MUST match the simulation tick rate, not the
    #: policy's decision rate. CARLA accumulates returns over simulated time and
    #: emits whatever swept past on each tick: at 20 rev/s in a world ticking at
    #: 60 Hz one frame carries a third of a revolution -- a fixed 120 degree
    #: wedge, and not the one in front. :meth:`SensorRig.spawn` overwrites this
    #: from the world's own ``fixed_delta_seconds``.
    rotation_frequency: float = 20.0
    upper_fov: float = 10.0
    lower_fov: float = -30.0
    x: float = -0.5
    y: float = 0.0
    z: float = 1.85
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    kind: str = "sensor.lidar.ray_cast"

    def to_transform(self):
        return carla.Transform(
            carla.Location(x=float(self.x), y=float(self.y), z=float(self.z)),
            carla.Rotation(roll=float(self.roll), pitch=float(self.pitch),
                           yaw=float(self.yaw)))

    def attributes(self) -> Dict[str, str]:
        return {"channels": str(int(self.channels)),
                "range": str(float(self.range_m)),
                "points_per_second": str(int(self.points_per_second)),
                "rotation_frequency": str(float(self.rotation_frequency)),
                "upper_fov": str(float(self.upper_fov)),
                "lower_fov": str(float(self.lower_fov))}

    def for_tick_rate(self, hz: float) -> "LidarSpec":
        """This spec, rotating exactly once per tick at ``hz``.

        One full revolution per tick is the only setting that gives the model a
        complete sweep, and ``points_per_second`` has to rise with the rotation
        rate to keep the same number of points inside each one.
        """
        if hz <= 0:
            return self
        return replace(self, rotation_frequency=float(hz),
                       points_per_second=int(round(self.points_per_revolution * hz)))


@dataclass(frozen=True)
class RadarSpec:
    """One radar.

    Detections are returned in the EGO frame -- ``(x, y, z, velocity)``, x
    forward, metres -- which is the frame the LiDAR path already uses and the
    frame TFv6's own ``preprocess_radar_input`` bounds-checks against.
    """
    name: str = "radar"
    horizontal_fov: float = 90.0
    vertical_fov: float = 0.1
    range_m: float = 100.0
    #: Returns per TICK the consumer wants. TFv6 pads or truncates each sensor's
    #: block to ``num_radar_points_per_sensor``, so fewer than this per tick is
    #: zero padding pretending to be clear road.
    points_per_tick: int = 150
    points_per_second: int = 1500
    x: float = 2.6
    y: float = 0.0
    z: float = 0.6
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    kind: str = "sensor.other.radar"

    def to_transform(self):
        return carla.Transform(
            carla.Location(x=float(self.x), y=float(self.y), z=float(self.z)),
            carla.Rotation(roll=float(self.roll), pitch=float(self.pitch),
                           yaw=float(self.yaw)))

    def attributes(self) -> Dict[str, str]:
        return {"horizontal_fov": str(float(self.horizontal_fov)),
                "vertical_fov": str(float(self.vertical_fov)),
                "range": str(float(self.range_m)),
                "points_per_second": str(int(self.points_per_second))}

    def for_tick_rate(self, hz: float) -> "RadarSpec":
        """This spec, delivering ``points_per_tick`` returns on every tick."""
        if hz <= 0:
            return self
        return replace(self, points_per_second=int(round(self.points_per_tick * hz)))


SensorSpec = Any  # CameraSpec | LidarSpec | RadarSpec, spelled for Python 3.8


# ---------------------------------------------------------------------------
# Named rigs
# ---------------------------------------------------------------------------

#: The rigs the two integrated sensorimotor policies are trained behind. A
#: policy's own ``sensors()`` is always preferred -- these exist so that a
#: policy which only NAMES a rig (``camera_rig = "tfv6"``) still gets the right
#: one, and so that this repository can render a rig without a checkpoint
#: present, which is what makes the sensor path testable on its own.
RIGS: Dict[str, List[SensorSpec]] = {
    # TFv6 / LEAD: three pinhole cameras stitched left-to-right into the
    # 1152x384 strip its `final_image_*` config describes, one LiDAR, and four
    # radars. The radar ORDER is load-bearing: `preprocess_radar_input` writes
    # each sensor's index into the fifth column, so reordering this list
    # silently relabels every detection.
    "tfv6": [
        CameraSpec("PCAM_L0", width=384, height=384, fov=60.0, yaw=-57.5,
                   x=0.0, y=-0.3, z=2.25),
        CameraSpec("PCAM_F0", width=384, height=384, fov=60.0, yaw=0.0,
                   x=0.25, y=0.0, z=2.25),
        CameraSpec("PCAM_R0", width=384, height=384, fov=60.0, yaw=57.5,
                   x=0.0, y=0.3, z=2.25),
        LidarSpec("lidar", channels=64, range_m=100.0, x=1.0, y=0.0, z=2.5),
        RadarSpec("radar1", x=2.6, z=0.60, yaw=-45.0),
        RadarSpec("radar2", x=2.6, z=0.60, yaw=45.0),
        RadarSpec("radar3", x=-2.6, z=0.60, yaw=135.0),
        RadarSpec("radar4", x=-2.6, z=0.60, yaw=225.0),
    ],
    # SimLingo: one wide forward camera, which the model tiles itself
    # (`dynamic_preprocess`), so the rig only has to deliver the full frame.
    "simlingo": [
        CameraSpec("rgb_front", width=1024, height=512, fov=110.0,
                   x=-1.5, y=0.0, z=2.0),
    ],
}


def rig(name: str) -> List[SensorSpec]:
    """A named rig, by copy so a caller cannot edit the table."""
    if name not in RIGS:
        raise KeyError(f"unknown sensor rig {name!r}; have {sorted(RIGS)}")
    return list(RIGS[name])


def specs_from(declared: Sequence[Any]) -> List[SensorSpec]:
    """Normalize whatever a policy's ``sensors()`` returned into specs.

    A policy may return specs from this module or plain dicts. Dicts are the
    documented form, because a policy repository describing its own rig must not
    have to import an execution method to do it -- that would be exactly the
    ``M x P`` coupling the standardized interface exists to avoid.

    The dict's ``kind`` chooses the dataclass, defaulting to a camera, and keys
    the chosen dataclass does not define are ignored rather than rejected: a
    policy may carry extra annotation for its own use, and a rig it declares
    should not stop working because this repository has not heard of a field.
    """
    out: List[SensorSpec] = []
    for item in declared:
        if isinstance(item, (CameraSpec, LidarSpec, RadarSpec)):
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(
                f"a policy's sensors() must yield CameraSpec, LidarSpec, "
                f"RadarSpec or dict, got {type(item).__name__}")
        kind = str(item.get("kind", ""))
        cls = (RadarSpec if "radar" in kind
               else LidarSpec if "lidar" in kind else CameraSpec)
        known = set(cls.__dataclass_fields__)
        out.append(cls(**{k: v for k, v in item.items() if k in known}))
    return out


# ---------------------------------------------------------------------------
# The rig itself
# ---------------------------------------------------------------------------

class SensorRig:
    """The ego's sensors, spawned, queued and drained in lockstep with ticks."""

    def __init__(self, world, ego_actor, specs: Sequence[SensorSpec]):
        self.world = world
        self.ego_actor = ego_actor
        self.specs: List[SensorSpec] = list(specs)
        #: ``(spec, sensor, queue)`` only for sensors that actually attached, so
        #: a failed spawn cannot misalign a spec with another sensor's queue.
        self.attached: List[Tuple[SensorSpec, Any, "queue.Queue"]] = []
        self.failed: List[str] = []
        #: Tick rate the sweeping sensors were retimed to, for the run report.
        self.sensor_hz: float = 0.0
        #: Ticks on which a sensor delivered nothing within the timeout.
        self.dropped: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def _tick_hz(self) -> float:
        """Simulation ticks per second, from the world's own settings.

        Read rather than assumed: a sweeping sensor's rate has to match the tick
        rate, and the tick rate is ``--fixed-dt``, not a constant.
        """
        try:
            delta = float(self.world.get_settings().fixed_delta_seconds or 0.0)
        except (RuntimeError, AttributeError, TypeError):
            return 0.0
        return 1.0 / delta if delta > 0 else 0.0

    def spawn(self) -> "SensorRig":
        """Attach every sensor.

        A sensor that cannot be created is recorded in :attr:`failed` rather
        than raised: the local backend has no camera blueprints, and a rig that
        raised there would break the analytic arm this repository ships with.
        Whether an unattached rig is fatal is the caller's decision, and
        :attr:`active` is what it decides on.
        """
        try:
            library = self.world.get_blueprint_library()
        except (RuntimeError, AttributeError) as exc:
            self.failed.append(f"blueprint library unavailable: {exc}")
            return self
        hz = 0.0 if os.environ.get("OSC2CARLA_SENSOR_RETIME") == "off" \
            else self._tick_hz()
        if hz > 0:
            self.specs = [spec.for_tick_rate(hz)
                          if hasattr(spec, "for_tick_rate") else spec
                          for spec in self.specs]
            self.sensor_hz = hz
        for spec in self.specs:
            try:
                bp = library.find(spec.kind)
                for key, value in spec.attributes().items():
                    bp.set_attribute(key, value)
                sensor = self.world.spawn_actor(bp, spec.to_transform(),
                                                attach_to=self.ego_actor)
            except Exception as exc:  # noqa: BLE001 - any spawn failure is data
                self.failed.append(f"{spec.name}: {exc}")
                continue
            q: "queue.Queue" = queue.Queue()
            sensor.listen(q.put)
            self.attached.append((spec, sensor, q))
        return self

    @property
    def active(self) -> bool:
        return bool(self.attached)

    @property
    def names(self) -> List[str]:
        return [spec.name for spec, _sensor, _q in self.attached]

    # ------------------------------------------------------------------ #
    def capture(self, frame: Optional[int] = None) -> Dict[str, Any]:
        """This rig's measurements for ``frame``, keyed by the declared name.

        Cameras come back as ``HxWx3`` uint8 RGB, LiDAR as ``Nx4``
        ``(x, y, z, intensity)`` in the sensor frame, radar as ``Nx4``
        ``(x, y, z, radial velocity)`` in the ego frame. A sensor that delivered
        nothing is simply absent from the mapping, which is what lets a policy
        say for itself that it is driving blind rather than be handed zeros it
        cannot distinguish from an empty road.
        """
        out: Dict[str, Any] = {}
        for spec, _sensor, q in self.attached:
            measurement = self._await(q, frame)
            if measurement is None:
                self.dropped[spec.name] = self.dropped.get(spec.name, 0) + 1
                continue
            if isinstance(spec, RadarSpec):
                array = self._to_radar(measurement, spec)
            elif isinstance(spec, LidarSpec):
                array = self._to_points(measurement)
            else:
                array = self._to_rgb(measurement)
            if array is not None:
                out[spec.name] = array
        return out

    @staticmethod
    def _await(q: "queue.Queue", frame: Optional[int]):
        """The measurement stamped with ``frame``, dropping everything older."""
        stale = 0
        while True:
            try:
                measurement = q.get(timeout=CAPTURE_TIMEOUT_S)
            except queue.Empty:
                return None
            if frame is None or getattr(measurement, "frame", frame) >= frame:
                return measurement
            stale += 1
            if stale > STALE_LIMIT:     # a stuck sensor, not a lagging one
                return measurement

    @staticmethod
    def _to_rgb(image):
        """CARLA hands over BGRA bytes; models in this family want RGB."""
        if np is None:
            return None
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        raw = raw.reshape((image.height, image.width, 4))
        return raw[:, :, :3][:, :, ::-1].copy()             # BGRA -> RGB

    @staticmethod
    def _to_points(measurement):
        """CARLA hands over flat float32 x,y,z,intensity; models want Nx4."""
        if np is None:
            return None
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
        return np.reshape(raw, (-1, 4)).copy()

    @staticmethod
    def _to_radar(measurement, spec: RadarSpec):
        """Radar detections in the ego frame.

        CARLA reports ``(velocity, azimuth, altitude, depth)`` per detection in
        the SENSOR frame. The consumer wants cartesian ego-frame points, so the
        spherical coordinates are resolved and the mounting transform applied.
        ``velocity`` is radial and keeps CARLA's own sign (negative is closing):
        a sign convention invented here would be a different quantity wearing
        the same name.

        Roll and pitch are zero on every radar in the rigs this repository
        knows, so only the yaw is rotated; a non-zero roll or pitch would need a
        full rotation and is refused rather than silently ignored.
        """
        if np is None:
            return None
        if float(spec.roll) or float(spec.pitch):
            raise ValueError(
                f"radar {spec.name!r} declares roll/pitch "
                f"({spec.roll}, {spec.pitch}); this conversion resolves yaw "
                "only, and quietly dropping the rest would mislabel every "
                "detection's position")
        raw = np.frombuffer(measurement.raw_data, dtype=np.float32)
        det = np.reshape(raw, (-1, 4))
        if det.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        vel, azimuth, altitude, depth = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
        horizontal = depth * np.cos(altitude)
        xs = horizontal * np.cos(azimuth)
        ys = horizontal * np.sin(azimuth)
        zs = depth * np.sin(altitude)
        yaw = np.radians(float(spec.yaw))
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        xe = xs * cos_y - ys * sin_y + float(spec.x)
        ye = xs * sin_y + ys * cos_y + float(spec.y)
        ze = zs + float(spec.z)
        return np.stack([xe, ye, ze, vel], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    def panel(self, captured: Dict[str, Any], height: int = 0):
        """The rig's camera frames as one strip, for a video overlay.

        Cameras only, in declared order, which for a multi-camera rig is the
        stitch order the model itself uses -- so the panel shows what the policy
        was looking at rather than a view chosen for the viewer. Returns ``None``
        when nothing was captured, which the recorder reads as "no panel this
        tick" rather than as an error.
        """
        if np is None:
            return None
        views = []
        for spec, _sensor, _q in self.attached:
            if not isinstance(spec, CameraSpec):
                continue
            frame = captured.get(spec.name)
            if frame is None:
                continue
            views.append(np.asarray(frame)[..., :3])
        if not views:
            return None
        target = height or max(v.shape[0] for v in views)
        scaled = []
        for view in views:
            if view.shape[0] != target:
                view = _resize(view, target)
            scaled.append(view)
        return np.concatenate(scaled, axis=1)

    # ------------------------------------------------------------------ #
    def destroy(self) -> None:
        for _spec, sensor, _q in self.attached:
            for method in ("stop", "destroy"):
                try:
                    getattr(sensor, method)()
                except Exception:  # noqa: BLE001 - teardown must not fail a run
                    pass
        self.attached = []

    def describe(self) -> Dict[str, Any]:
        """What was asked for and what was actually attached, for the report."""
        return {
            "cameras": [{"name": s.name, "width": s.width, "height": s.height,
                         "fov": s.fov, "yaw": s.yaw,
                         "mount": [s.x, s.y, s.z]}
                        for s in self.specs if isinstance(s, CameraSpec)],
            "lidars": [{"name": s.name, "channels": s.channels,
                        "range_m": s.range_m,
                        "rotation_frequency": s.rotation_frequency,
                        "points_per_second": s.points_per_second}
                       for s in self.specs if isinstance(s, LidarSpec)],
            "radars": [{"name": s.name, "yaw": s.yaw,
                        "horizontal_fov": s.horizontal_fov,
                        "range_m": s.range_m,
                        "points_per_second": s.points_per_second}
                       for s in self.specs if isinstance(s, RadarSpec)],
            "attached": self.names,
            "sensor_hz": self.sensor_hz,
            "dropped_frames": dict(self.dropped),
            "failed": list(self.failed),
        }


def _resize(view, height: int):
    """Nearest-neighbour row/column sampling, so the panel needs no OpenCV.

    The panel is a visual check, not model input; nothing reads it back. Using
    numpy indexing rather than cv2 keeps the rig importable in an interpreter
    that has the CARLA API but no OpenCV, which is the common case inside a
    simulator image.
    """
    scale = height / float(view.shape[0])
    width = max(1, int(round(view.shape[1] * scale)))
    rows = (np.arange(height) / scale).astype(int).clip(0, view.shape[0] - 1)
    cols = (np.arange(width) / scale).astype(int).clip(0, view.shape[1] - 1)
    return view[rows][:, cols]
