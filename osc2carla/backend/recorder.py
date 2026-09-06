"""Camera + collision sensor recorder, used to produce annotated MP4 output.

Designed to mirror ``scripts/record_scenario_collision.py``: a chase-cam is
attached to the target actor, every synchronous tick a frame is grabbed and
overlaid with the current collision count + last impulse magnitude, and at
``finalize`` time the frames are encoded into an MP4 via ffmpeg.

The vision panel
----------------
A chase cam shows what the *scenario* did.  For a sensorimotor ego it does not
show what the *policy* did, because the policy never saw the chase cam -- it saw
its own rig, at its own mounting, cropped and tiled its own way.  So an optional
``vision`` hook lets the caller supply, per tick, the rig's own frames and a
couple of lines of policy state; they are drawn under the chase cam in the same
MP4.  Without it the recording is exactly what it always was.

That panel is the cheapest check there is on a sensor integration.  A rig that
is mounted wrong, pointed backwards, or delivering the previous tick's frame all
produce plausible-looking numbers and an obviously wrong video.
"""
from __future__ import annotations

import glob
import math
import os
import queue
import subprocess
from typing import Any, List, Optional

from .simapi import sim as carla

try:
    import numpy as np
    import cv2
except Exception:  # noqa: BLE001
    np = None  # type: ignore
    cv2 = None  # type: ignore

#: Height of the vision band when a caller supplies a `vision` hook without
#: choosing one. Tall enough to read a 384-row rig strip scaled to 1280 wide.
DEFAULT_VISION_HEIGHT = 260


class Recorder:
    def __init__(self, world, target_actor, output_video: str,
                 frames_dir: Optional[str] = None,
                 width: int = 1280, height: int = 720, fps: int = 20,
                 record_collisions: bool = True,
                 cam_x: float = -9.0, cam_z: float = 5.0, cam_pitch: float = -22.0,
                 fov: float = 95.0,
                 vision: Optional[Any] = None, vision_height: int = 0):
        self.world = world
        self.target = target_actor
        self.output_video = output_video
        self.frames_dir = frames_dir or (output_video + "_frames")
        self.width = width
        self.height = height
        self.fps = fps
        self.record_collisions = record_collisions
        #: Callable returning ``{"panel": HxWx3 RGB array or None,
        #: "lines": [str, ...]}`` for the current tick, or None for no panel.
        self.vision = vision
        #: Fixed height of the panel band. Fixed, not derived per frame, because
        #: every frame in an MP4 must be the same size: a tick where the rig
        #: delivered nothing has to produce a black band of the same height
        #: rather than a shorter frame ffmpeg would refuse.
        self.vision_height = int(vision_height or (DEFAULT_VISION_HEIGHT
                                                   if vision else 0))
        self._cam = None
        self._col = None
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._collisions: List[dict] = []
        self._frame_idx = 0
        self._sim_time = 0.0
        os.makedirs(self.frames_dir, exist_ok=True)
        # Clear stale frames from a previous run so ffmpeg does not append an
        # outdated tail (frame_%05d.png is overwritten from index 0 each run).
        for stale in glob.glob(os.path.join(self.frames_dir, "frame_*.png")):
            try:
                os.remove(stale)
            except OSError:
                pass

        if not carla or world is None or target_actor is None:
            return

        bps = world.get_blueprint_library()
        cam_bp = bps.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(width))
        cam_bp.set_attribute("image_size_y", str(height))
        cam_bp.set_attribute("fov", str(fov))
        cam_tf = carla.Transform(
            carla.Location(x=cam_x, z=cam_z),
            carla.Rotation(pitch=cam_pitch),
        )
        self._cam = world.spawn_actor(cam_bp, cam_tf, attach_to=target_actor)
        self._cam.listen(self._queue.put)

        if record_collisions:
            col_bp = bps.find("sensor.other.collision")
            self._col = world.spawn_actor(col_bp, carla.Transform(), attach_to=target_actor)
            self._col.listen(self._on_collision)

    def _on_collision(self, event):
        imp = event.normal_impulse
        mag = math.sqrt(imp.x * imp.x + imp.y * imp.y + imp.z * imp.z)
        self._collisions.append({
            "frame": event.frame,
            # Stamped from the last tick() so metrics can report when the first
            # contact happened without attaching a second collision sensor.
            "sim_time": self._sim_time,
            "other": event.other_actor.type_id,
            "other_role": (getattr(event.other_actor, "attributes", {}) or {}
                           ).get("role_name", ""),
            "impulse_mag": mag,
        })

    @property
    def collisions(self) -> List[dict]:
        return list(self._collisions)

    def tick(self, sim_time: float) -> None:
        self._sim_time = sim_time
        if self._cam is None or np is None or cv2 is None:
            return
        try:
            image = self._queue.get(timeout=2.0)
        except queue.Empty:
            return
        buf = np.frombuffer(image.raw_data, dtype=np.uint8)
        img = buf.reshape((self.height, self.width, 4))[:, :, :3].copy()
        n_hits = len(self._collisions)
        last_imp = self._collisions[-1]["impulse_mag"] if n_hits else 0.0
        cv2.putText(img, f"t={sim_time:4.1f}s  hits={n_hits}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if n_hits > 0:
            cv2.putText(img, f"COLLISION! impulse={last_imp:7.0f}", (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        if self.vision_height > 0:
            img = np.vstack([img, self._vision_band()])
        cv2.imwrite(os.path.join(self.frames_dir, f"frame_{self._frame_idx:05d}.png"), img)
        self._frame_idx += 1

    def _vision_band(self):
        """The band under the chase cam: what the ego policy saw, and was told.

        Always ``vision_height`` rows of ``width`` columns, whatever the rig
        delivered, so the frame size never changes mid-recording. A tick with no
        panel is a black band, which is itself readable: it says the rig
        delivered nothing on that tick.
        """
        band = np.zeros((self.vision_height, self.width, 3), dtype=np.uint8)
        payload = {}
        try:
            payload = self.vision() or {}
        except Exception as exc:  # noqa: BLE001 - a recording must not fail a run
            cv2.putText(band, f"vision panel unavailable: {exc}"[:110], (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            return band
        panel = payload.get("panel")
        if panel is not None:
            panel = np.asarray(panel)[..., :3]
            # The rig hands over RGB; cv2 writes BGR, and the chase-cam frame
            # above is already BGR because it comes straight off CARLA's BGRA
            # buffer. Swapping here rather than at capture keeps the array the
            # policy saw untouched.
            panel = panel[:, :, ::-1]
            scale = min(self.vision_height / panel.shape[0],
                        self.width / panel.shape[1])
            rows = (np.arange(int(panel.shape[0] * scale))
                    / scale).astype(int).clip(0, panel.shape[0] - 1)
            cols = (np.arange(int(panel.shape[1] * scale))
                    / scale).astype(int).clip(0, panel.shape[1] - 1)
            fitted = panel[rows][:, cols]
            band[:fitted.shape[0], :fitted.shape[1]] = fitted
        for i, line in enumerate(payload.get("lines") or []):
            cv2.putText(band, str(line)[:120], (12, 24 + 26 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        return band

    def finalize(self) -> Optional[str]:
        try:
            if self._cam is not None:
                self._cam.stop()
            if self._col is not None:
                self._col.stop()
        except Exception:
            pass
        try:
            if self._cam is not None:
                self._cam.destroy()
            if self._col is not None:
                self._col.destroy()
        except Exception:
            pass
        if self._frame_idx == 0:
            return None
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(self.fps),
            "-i", os.path.join(self.frames_dir, "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            self.output_video,
        ]
        try:
            subprocess.run(cmd, check=False)
            return self.output_video
        except FileNotFoundError:
            return None
