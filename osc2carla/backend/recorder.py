"""Camera + collision sensor recorder, used to produce annotated MP4 output.

Designed to mirror ``scripts/record_scenario_collision.py``: a chase-cam is
attached to the target actor, every synchronous tick a frame is grabbed and
overlaid with the current collision count + last impulse magnitude, and at
``finalize`` time the frames are encoded into an MP4 via ffmpeg.
"""
from __future__ import annotations

import glob
import math
import os
import queue
import subprocess
from typing import Any, List, Optional

try:
    import carla  # type: ignore
except Exception:  # noqa: BLE001
    carla = None  # type: ignore

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
        self._col = None
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._collisions: List[dict] = []
        self._frame_idx = 0
        os.makedirs(self.frames_dir, exist_ok=True)
        # Clear stale frames from a previous run so ffmpeg does not append an
        # outdated tail (frame_%05d.png is overwritten from index 0 each run).
        for stale in glob.glob(os.path.join(self.frames_dir, "frame_*.png")):
            try:
                os.remove(stale)
            except OSError:
                pass

        if carla is None or world is None or target_actor is None:
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
            "other": event.other_actor.type_id,
            "impulse_mag": mag,
        })

    @property
    def collisions(self) -> List[dict]:
        return list(self._collisions)

    def tick(self, sim_time: float) -> None:
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
        cv2.imwrite(os.path.join(self.frames_dir, f"frame_{self._frame_idx:05d}.png"), img)
        self._frame_idx += 1

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
