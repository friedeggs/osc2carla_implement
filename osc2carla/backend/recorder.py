"""Camera + collision sensor recorder, used to produce annotated MP4 output.

Two RGB cameras on the target actor, composited side by side:

    top     a bird's-eye view straight down on the ego, North up. This is the
            view that shows the other actors -- who is approaching, from which
            arm, and how close.
    chase   a following camera behind the ego, which shows whether the ego is
            driving the route and what it hit.

North up rather than ego-heading up on purpose: a rotating frame makes it hard to
see that a vehicle is closing from a fixed direction, which is the thing these
scenarios are about. It also matches the convention the orchestration method's own
recorder uses, so the two methods' videos can be read side by side.

The top view follows the ego rather than centring on a junction, because three of
the six families are highway scenarios with no junction to centre on.

Every synchronous tick both frames are grabbed and the pair is overlaid with the
collision count and last impulse magnitude; at ``finalize`` the frames are encoded
into an MP4 via ffmpeg. Set ``$OSC2CARLA_RECORD_VIEW=chase`` for the single-camera
behaviour this had before.
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


class Recorder:
    def __init__(self, world, target_actor, output_video: str,
                 frames_dir: Optional[str] = None,
                 width: int = 1280, height: int = 720, fps: int = 20,
                 record_collisions: bool = True,
                 cam_x: float = -9.0, cam_z: float = 5.0, cam_pitch: float = -22.0,
                 fov: float = 95.0):
        self.world = world
        self.target = target_actor
        self.output_video = output_video
        self.frames_dir = frames_dir or (output_video + "_frames")
        self.width = width
        self.height = height
        self.fps = fps
        self.record_collisions = record_collisions
        self._cam = None
        self._top = None
        self._col = None
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._top_queue: "queue.Queue[Any]" = queue.Queue()
        # "both" composites top+chase; "chase" is the older single-camera output.
        self._view = (os.environ.get("OSC2CARLA_RECORD_VIEW") or "both").strip().lower()
        #: Metres across the short axis of the top view. 60 m at this fov keeps a
        #: junction and its approaches in frame without shrinking the vehicles to
        #: specks.
        self._top_span = float(os.environ.get("OSC2CARLA_TOP_SPAN") or 60.0)
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

        if self._view == "both":
            # Height from the span and the fov, so the framing is a stated number
            # of metres rather than a magic altitude. yaw=-90 puts world +x to the
            # image right and world -y up, which is North up in CARLA's
            # left-handed frame.
            z = 0.5 * self._top_span / math.tan(math.radians(0.5 * fov))
            top_bp = bps.find("sensor.camera.rgb")
            top_bp.set_attribute("image_size_x", str(width))
            top_bp.set_attribute("image_size_y", str(height))
            top_bp.set_attribute("fov", str(fov))
            self._top = world.spawn_actor(
                top_bp,
                carla.Transform(carla.Location(z=z),
                                carla.Rotation(pitch=-90.0, yaw=-90.0)),
                attach_to=target_actor)
            self._top.listen(self._top_queue.put)

        if record_collisions:
            col_bp = bps.find("sensor.other.collision")
            self._col = world.spawn_actor(col_bp, carla.Transform(), attach_to=target_actor)
            self._col.listen(self._on_collision)

    def _to_bgr(self, image):
        """One CARLA image as an H x W x 3 BGR array."""
        buf = np.frombuffer(image.raw_data, dtype=np.uint8)
        return buf.reshape((self.height, self.width, 4))[:, :, :3].copy()

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
        img = self._to_bgr(image)

        if self._top is not None:
            try:
                top = self._to_bgr(self._top_queue.get(timeout=2.0))
            except queue.Empty:
                # A dropped top frame must not shift the chase timeline, so the
                # pair is padded rather than skipped.
                top = np.zeros_like(img)
            cv2.putText(top, "top (North up)", (20, self.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "chase", (20, self.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            img = np.hstack((top, img))

        n_hits = len(self._collisions)
        last_imp = self._collisions[-1]["impulse_mag"] if n_hits else 0.0
        cv2.putText(img, f"t={sim_time:4.1f}s  hits={n_hits}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if n_hits > 0:
            cv2.putText(img, f"COLLISION! impulse={last_imp:7.0f}", (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.frames_dir, f"frame_{self._frame_idx:05d}.png"), img)
        self._frame_idx += 1

    def finalize(self) -> Optional[str]:
        try:
            if self._cam is not None:
                self._cam.stop()
            if self._top is not None:
                self._top.stop()
            if self._col is not None:
                self._col.stop()
        except Exception:
            pass
        try:
            if self._cam is not None:
                self._cam.destroy()
            if self._top is not None:
                self._top.destroy()
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
