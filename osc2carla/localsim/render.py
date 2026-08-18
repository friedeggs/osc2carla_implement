"""Bird's-eye renderer and recorder for the local simulator.

Interface-compatible with ``osc2carla.backend.recorder.Recorder`` -- same
``tick(sim_time)`` / ``finalize()`` / ``collisions`` surface -- so the CLI
wires either one into the same run loop.  What differs is the picture: CARLA
renders a chase camera on a GPU, this draws the lane graph and the actors'
collision boxes top-down with pygame.

The view uses CARLA's convention (+x east, +y *south*), so the drawing has +x
right and +y down.  A left turn therefore looks like a left turn, and the
coordinates printed in a scenario file are the coordinates on screen.

Headless use is the same code path with SDL's dummy video driver: the frames
are drawn to an off-screen surface, written as PNGs, and encoded with ffmpeg
at ``finalize()``.
"""
from __future__ import annotations

import glob
import math
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pygame
except Exception:  # noqa: BLE001 - pygame is optional until a run needs it
    pygame = None  # type: ignore


# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------

GRASS = (28, 36, 30)
ROAD = (56, 58, 62)
JUNCTION = (66, 68, 73)
EDGE_LINE = (196, 198, 200)
CENTRE_LINE = (206, 176, 60)
DASH = (150, 152, 156)
PROP = (176, 132, 60)
HUD_BG = (12, 14, 18)
HUD_FG = (226, 230, 234)
HUD_DIM = (140, 148, 158)
ACCENT = (120, 190, 255)
ALERT = (255, 90, 70)
OK = (120, 220, 140)


@dataclass
class HudState:
    """Everything the overlay shows that the simulator itself does not know."""

    scenario: str = ""
    backend: str = "pygame"
    town: str = ""
    events: Dict[str, bool] = field(default_factory=dict)
    active_leaves: List[str] = field(default_factory=list)
    tree_status: str = ""
    ego_policy: Optional[str] = None
    ego_binding: Optional[str] = None
    bindings: Dict[int, str] = field(default_factory=dict)   # actor id -> binding
    paused: bool = False
    note: str = ""


class RendererUnavailable(RuntimeError):
    pass


class BevRenderer:
    """Draws the world top-down; optionally to a window, to PNGs, or both."""

    #: metres of map padding around the network when fitting the whole town
    FIT_MARGIN = 12.0
    #: pixels per metre used when the camera follows an actor
    FOLLOW_SCALE = 7.0

    def __init__(self, world, target_actor=None, output_video: Optional[str] = None,
                 frames_dir: Optional[str] = None,
                 width: int = 1280, height: int = 720, fps: int = 20,
                 display: bool = True, scale: Optional[float] = None,
                 follow: bool = True, record_collisions: bool = True,
                 caption: str = "osc2carla — local simulator"):
        if pygame is None:
            raise RendererUnavailable(
                "pygame is not installed. `pip install pygame`, or run with "
                "--render-mode off to simulate without a picture.")
        self.world = world
        self.map = world.get_map()
        self.target = target_actor
        self.output_video = output_video
        self.frames_dir = frames_dir or (output_video + "_frames" if output_video else None)
        self.width, self.height = int(width), int(height)
        self.fps = int(fps)
        self.display = bool(display)
        self.follow = bool(follow)
        self._quit = False
        self._paused = False
        self._frame_idx = 0
        self._sim_time = 0.0
        self._collisions: List[dict] = []
        self._col_sensor = None
        self._flash: List[Tuple[float, float, float]] = []   # x, y, ttl

        if not self.display:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.font.init()
        if self.display:
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(caption)
        else:
            self._screen = pygame.Surface((self.width, self.height))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("dejavusansmono,monospace", 15)
        self._font_small = pygame.font.SysFont("dejavusansmono,monospace", 13)
        self._font_big = pygame.font.SysFont("dejavusansmono,monospace", 20, bold=True)

        # Following one actor wants a close view; showing the whole town
        # wants whatever fits.  --render-scale overrides both.
        self._scale = float(scale) if scale else (
            max(self._fit_scale(), self.FOLLOW_SCALE) if self.follow
            else self._fit_scale())
        self._background: Optional[Any] = None
        self._bg_origin = (0.0, 0.0)
        self._render_background()

        if self.frames_dir:
            os.makedirs(self.frames_dir, exist_ok=True)
            for stale in glob.glob(os.path.join(self.frames_dir, "frame_*.png")):
                try:
                    os.remove(stale)
                except OSError:
                    pass

        if record_collisions and target_actor is not None:
            bp = world.get_blueprint_library().find("sensor.other.collision")
            from .geometry import Transform
            self._col_sensor = world.spawn_actor(bp, Transform(),
                                                 attach_to=target_actor)
            self._col_sensor.listen(self._on_collision)

    # -- collision bookkeeping (mirrors backend.recorder.Recorder) ---------

    def _on_collision(self, event) -> None:
        imp = event.normal_impulse
        self._collisions.append({
            "frame": event.frame,
            "sim_time": self._sim_time,
            "other": event.other_actor.type_id,
            "other_role": (getattr(event.other_actor, "attributes", {}) or {}
                           ).get("role_name", ""),
            "impulse_mag": imp.length(),
        })
        loc = event.other_actor.get_location()
        self._flash.append((loc.x, loc.y, 0.6))

    @property
    def collisions(self) -> List[dict]:
        return list(self._collisions)

    @property
    def quit_requested(self) -> bool:
        return self._quit

    @property
    def paused(self) -> bool:
        return self._paused

    # -- camera -----------------------------------------------------------

    def _fit_scale(self) -> float:
        min_x, min_y, max_x, max_y = self.map.bounds
        span_x = (max_x - min_x) + 2 * self.FIT_MARGIN
        span_y = (max_y - min_y) + 2 * self.FIT_MARGIN
        return max(1.0, min(self.width / max(span_x, 1.0),
                            self.height / max(span_y, 1.0)))

    def _camera_centre(self) -> Tuple[float, float]:
        if self.follow and self.target is not None and self.target.is_alive:
            loc = self.target.get_location()
            return loc.x, loc.y
        min_x, min_y, max_x, max_y = self.map.bounds
        return (min_x + max_x) * 0.5, (min_y + max_y) * 0.5

    def _to_screen(self, x: float, y: float) -> Tuple[int, int]:
        cx, cy = self._camera_centre()
        return (int(round((x - cx) * self._scale + self.width * 0.5)),
                int(round((y - cy) * self._scale + self.height * 0.5)))

    # -- static background ------------------------------------------------

    def _render_background(self) -> None:
        """Draw the road network once into an off-screen surface."""
        min_x, min_y, max_x, max_y = self.map.bounds
        pad = self.FIT_MARGIN
        self._bg_origin = (min_x - pad, min_y - pad)
        w = int((max_x - min_x + 2 * pad) * self._scale) + 2
        h = int((max_y - min_y + 2 * pad) * self._scale) + 2
        surf = pygame.Surface((max(w, 1), max(h, 1)))
        surf.fill(GRASS)

        def to_bg(px: float, py: float) -> Tuple[float, float]:
            return ((px - self._bg_origin[0]) * self._scale,
                    (py - self._bg_origin[1]) * self._scale)

        lanes = list(self.map.lanes.values())
        # Junction boxes first, as filled areas: drawing each connector as a
        # thick polyline instead leaves a scalloped edge where the arcs fan out.
        for rect in _junction_boxes(lanes):
            x0, y0 = to_bg(rect[0], rect[1])
            x1, y1 = to_bg(rect[2], rect[3])
            pygame.draw.rect(surf, JUNCTION,
                             pygame.Rect(int(x0), int(y0),
                                         max(1, int(x1 - x0)), max(1, int(y1 - y0))))
        for lane in lanes:
            if lane.is_junction:
                continue
            pts = [to_bg(x, y) for x, y in lane.points]
            _thick_polyline(surf, pts, ROAD, lane.width * self._scale)
        for lane in lanes:
            if not lane.is_junction:
                self._lane_markings(surf, lane, to_bg)
        self._background = surf

    def _lane_markings(self, surf, lane, to_bg) -> None:
        half = lane.width * 0.5
        inner = _offset_polyline(lane.points, -half)
        outer = _offset_polyline(lane.points, +half)
        if lane.left is None:
            # innermost lane of a carriageway: the road's centre line
            _line(surf, [to_bg(*p) for p in inner], CENTRE_LINE,
                  max(1, int(0.16 * self._scale)))
        if lane.right is None:
            _line(surf, [to_bg(*p) for p in outer], EDGE_LINE,
                  max(1, int(0.14 * self._scale)))
        else:
            _dashed(surf, [to_bg(*p) for p in outer], DASH,
                    max(1, int(0.12 * self._scale)), self._scale)

    # -- the frame --------------------------------------------------------

    def tick(self, sim_time: float, hud: Optional[HudState] = None) -> bool:
        """Draw one frame. Returns False once the user has closed the window."""
        self._sim_time = sim_time
        self._pump_events()
        if self._quit:
            return False
        self._draw(sim_time, hud or HudState())
        if self.display:
            pygame.display.flip()
        if self.frames_dir:
            pygame.image.save(
                self._screen,
                os.path.join(self.frames_dir, f"frame_{self._frame_idx:05d}.png"))
        self._frame_idx += 1
        return True

    def pump(self) -> bool:
        """Process input without drawing (used while paused)."""
        self._pump_events()
        return not self._quit

    def throttle(self, fps: Optional[int] = None) -> None:
        """Sleep so the run plays back at wall-clock speed."""
        self._clock.tick(fps or self.fps)

    def _pump_events(self) -> None:
        if not self.display:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit = True
            elif event.type == pygame.KEYDOWN:
                self._on_key(event.key)

    def _on_key(self, key) -> None:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self._quit = True
        elif key == pygame.K_SPACE:
            self._paused = not self._paused
        elif key == pygame.K_f:
            self.follow = not self.follow
        elif key == pygame.K_TAB:
            self._cycle_target()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._set_scale(self._scale * 1.25)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._set_scale(self._scale / 1.25)

    def _set_scale(self, scale: float) -> None:
        self._scale = max(0.8, min(40.0, scale))
        self._render_background()

    def _cycle_target(self) -> None:
        vehicles = self.world.vehicles()
        if not vehicles:
            return
        try:
            idx = vehicles.index(self.target)
        except ValueError:
            idx = -1
        self.target = vehicles[(idx + 1) % len(vehicles)]

    def _draw(self, sim_time: float, hud: HudState) -> None:
        screen = self._screen
        screen.fill(GRASS)
        cx, cy = self._camera_centre()
        ox = int(round((self._bg_origin[0] - cx) * self._scale + self.width * 0.5))
        oy = int(round((self._bg_origin[1] - cy) * self._scale + self.height * 0.5))
        screen.blit(self._background, (ox, oy))

        for prop in self.world.props():
            self._draw_prop(prop, hud)
        for vehicle in self.world.vehicles():
            self._draw_vehicle(vehicle, hud)
        self._draw_contacts()
        self._draw_hud(sim_time, hud)

    # -- actors -----------------------------------------------------------

    def _draw_prop(self, prop, hud: HudState) -> None:
        e = prop.bounding_box.extent
        pts = [self._to_screen(*p) for p in _box_corners(prop, e.x, e.y)]
        pygame.draw.polygon(self._screen, PROP, pts)
        label = hud.bindings.get(prop.id) or prop.attributes.get("role_name", "")
        if label:
            self._label(pts[0][0], pts[0][1] - 14, label, HUD_DIM, self._font_small)

    def _draw_vehicle(self, vehicle, hud: HudState) -> None:
        e = vehicle.bounding_box.extent
        corners = [self._to_screen(*p) for p in _box_corners(vehicle, e.x, e.y)]
        colour = getattr(vehicle, "color", (170, 170, 175))
        pygame.draw.polygon(self._screen, colour, corners)
        pygame.draw.polygon(self._screen, _darken(colour, 0.45), corners, 2)

        # nose wedge, so heading is readable at any zoom
        yaw = vehicle.yaw
        nose = self._to_screen(vehicle._x + math.cos(yaw) * e.x * 0.98,
                               vehicle._y + math.sin(yaw) * e.x * 0.98)
        pygame.draw.circle(self._screen, _lighten(colour, 0.5), nose,
                           max(2, int(0.35 * self._scale)))
        if getattr(vehicle, "braking", False):
            rear = self._to_screen(vehicle._x - math.cos(yaw) * e.x,
                                   vehicle._y - math.sin(yaw) * e.x)
            pygame.draw.circle(self._screen, ALERT, rear,
                               max(2, int(0.4 * self._scale)))

        role = hud.bindings.get(vehicle.id) or vehicle.attributes.get("role_name", "")
        speed_kph = vehicle.speed * 3.6
        tag = f"{role} {speed_kph:.0f}" if role else f"{speed_kph:.0f}"
        top = min(c[1] for c in corners)
        mid = sum(c[0] for c in corners) // 4
        is_ego = hud.ego_binding is not None and role == hud.ego_binding
        self._label(mid - 4 * len(tag), top - 16, tag,
                    ACCENT if is_ego else HUD_FG, self._font_small)

    def _draw_contacts(self) -> None:
        alive = []
        for x, y, ttl in self._flash:
            r = max(6, int((0.8 - ttl) * 3 * self._scale))
            pygame.draw.circle(self._screen, ALERT, self._to_screen(x, y), r, 2)
            ttl -= 1.0 / max(self.fps, 1)
            if ttl > 0:
                alive.append((x, y, ttl))
        self._flash = alive[-24:]

    # -- overlay ----------------------------------------------------------

    def _label(self, x: int, y: int, text: str, colour, font) -> None:
        self._screen.blit(font.render(text, True, colour), (x, y))

    def _draw_hud(self, sim_time: float, hud: HudState) -> None:
        lines: List[Tuple[str, Any]] = []
        n_hits = len(self._collisions)
        lines.append((f"t = {sim_time:6.2f} s", HUD_FG))
        if hud.scenario:
            lines.append((f"scenario  {hud.scenario}", HUD_DIM))
        lines.append((f"backend   {hud.backend}"
                      + (f" / {hud.town}" if hud.town else ""), HUD_DIM))
        if hud.ego_policy:
            lines.append((f"ego       {hud.ego_binding} <- {hud.ego_policy}", ACCENT))
        lines.append(("", HUD_DIM))
        lines.append((f"collisions {n_hits}", ALERT if n_hits else OK))
        if n_hits:
            last = self._collisions[-1]
            lines.append((f"  last {last['other_role'] or last['other']} "
                          f"|J|={last['impulse_mag']:.0f}", ALERT))
        fired = [k for k, v in hud.events.items() if v]
        if fired:
            lines.append(("", HUD_DIM))
            lines.append(("events", HUD_DIM))
            for name in fired[-8:]:
                lines.append((f"  @{name}", OK))
        if hud.active_leaves:
            lines.append(("", HUD_DIM))
            lines.append(("running", HUD_DIM))
            for leaf in hud.active_leaves[:12]:
                lines.append((f"  {leaf}", HUD_FG))
        if hud.note:
            lines.append(("", HUD_DIM))
            lines.append((hud.note, HUD_DIM))

        pad = 10
        box_w = 330
        box_h = pad * 2 + len(lines) * 18
        panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        panel.fill((*HUD_BG, 205))
        self._screen.blit(panel, (0, 0))
        y = pad
        for text, colour in lines:
            if text:
                self._label(pad, y, text, colour, self._font)
            y += 18

        if self.display:
            help_text = "SPACE pause  TAB next actor  F follow  +/- zoom  Q quit"
            self._label(pad, self.height - 22, help_text, HUD_DIM, self._font_small)
        if hud.paused or self._paused:
            self._label(self.width // 2 - 40, 12, "PAUSED", ALERT, self._font_big)

    # -- teardown ---------------------------------------------------------

    def finalize(self) -> Optional[str]:
        if self._col_sensor is not None:
            try:
                self._col_sensor.stop()
                self._col_sensor.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._col_sensor = None
        try:
            pygame.display.quit()
        except Exception:  # noqa: BLE001
            pass
        pygame.quit()
        if not self.output_video or not self.frames_dir or self._frame_idx == 0:
            return None
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.fps),
               "-i", os.path.join(self.frames_dir, "frame_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", self.output_video]
        try:
            subprocess.run(cmd, check=False)
        except FileNotFoundError:
            return None
        return self.output_video


# --------------------------------------------------------------------------
# drawing helpers
# --------------------------------------------------------------------------

def _junction_boxes(lanes) -> List[Tuple[float, float, float, float]]:
    """One ``(min_x, min_y, max_x, max_y)`` per junction, from its connectors."""
    groups: Dict[int, List[Tuple[float, float]]] = {}
    widths: Dict[int, float] = {}
    for lane in lanes:
        if not lane.is_junction:
            continue
        groups.setdefault(lane.road_id, []).extend(lane.points)
        widths[lane.road_id] = max(widths.get(lane.road_id, 0.0), lane.width)
    out = []
    for road_id, pts in groups.items():
        half = widths[road_id] * 0.5
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append((min(xs) - half, min(ys) - half, max(xs) + half, max(ys) + half))
    return out


def _box_corners(actor, half_len: float, half_wid: float):
    from .geometry import corners_2d
    return corners_2d(actor._x, actor._y, math.radians(actor._yaw_deg),
                      half_len, half_wid)


def _thick_polyline(surf, points: Sequence[Tuple[float, float]], colour,
                    width: float) -> None:
    w = max(1, int(round(width)))
    ipts = [(int(round(x)), int(round(y))) for x, y in points]
    if len(ipts) >= 2:
        pygame.draw.lines(surf, colour, False, ipts, w)
    r = w // 2
    if r >= 1:
        for p in ipts:                    # round off the joins
            pygame.draw.circle(surf, colour, p, r)


def _line(surf, points, colour, width: int) -> None:
    ipts = [(int(round(x)), int(round(y))) for x, y in points]
    if len(ipts) >= 2:
        pygame.draw.lines(surf, colour, False, ipts, max(1, width))


def _dashed(surf, points, colour, width: int, scale: float,
            dash_m: float = 3.0, gap_m: float = 3.0) -> None:
    dash = max(3.0, dash_m * scale)
    gap = max(3.0, gap_m * scale)
    carry = 0.0
    drawing = True
    for i in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[i], points[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-6:
            continue
        pos = 0.0
        while pos < seg:
            span = (dash if drawing else gap) - carry
            end = min(seg, pos + span)
            if drawing:
                t0, t1 = pos / seg, end / seg
                pygame.draw.line(
                    surf, colour,
                    (int(x0 + (x1 - x0) * t0), int(y0 + (y1 - y0) * t0)),
                    (int(x0 + (x1 - x0) * t1), int(y0 + (y1 - y0) * t1)),
                    max(1, width))
            if end >= pos + span:
                drawing = not drawing
                carry = 0.0
            else:
                carry += end - pos
            pos = end


def _offset_polyline(points: Sequence[Tuple[float, float]],
                     offset: float) -> List[Tuple[float, float]]:
    """Shift a polyline sideways; positive is to the right of travel."""
    out = []
    n = len(points)
    for i, (x, y) in enumerate(points):
        j = min(i + 1, n - 1)
        k = max(i - 1, 0)
        dx, dy = points[j][0] - points[k][0], points[j][1] - points[k][1]
        d = math.hypot(dx, dy)
        if d < 1e-9:
            out.append((x, y))
            continue
        out.append((x - dy / d * offset, y + dx / d * offset))
    return out


def _darken(colour, k: float):
    return tuple(max(0, int(c * (1.0 - k))) for c in colour[:3])


def _lighten(colour, k: float):
    return tuple(min(255, int(c + (255 - c) * k)) for c in colour[:3])
