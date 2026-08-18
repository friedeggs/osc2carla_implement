"""Render a local town as a coordinate reference sheet for scenario authoring.

Writing a scenario against a CARLA town means opening the map in the
simulator and reading coordinates off it.  The local towns are synthesised,
so there is nothing to open -- this module is the substitute.  It draws the
network to scale with a metric grid, labels every junction centre and every
carriageway with the numbers you paste into ``position(x:, y:, h:)``, and
writes a companion Markdown table of the same figures.

    python -m osc2carla.localsim.mapview                  # all towns -> maps/
    python -m osc2carla.localsim.mapview --town grid --scale 12

The picture uses CARLA's convention: +x east (right), +y *south* (down).  A
left turn therefore appears to go left, and headings run clockwise on screen,
which is why ``h`` for a southbound lane is +1.5708 rad and not -1.5708.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from .render import (ACCENT, ALERT, GRASS, HUD_DIM, HUD_FG, OK,
                     draw_network, junction_boxes)
from .roadmap import Map
from .towns import BUILTIN_TOWNS, load_town, town_names

try:
    import pygame
except Exception:  # noqa: BLE001
    pygame = None  # type: ignore

#: metric grid spacing, in metres
GRID_STEP = 20.0
GRID_LINE = (48, 58, 50)
GRID_TEXT = (128, 140, 132)
MARGIN_LEFT = 74     # pixels reserved for the y-axis labels
MARGIN_TOP = 30      # ... the x-axis labels
MARGIN_RIGHT = 190   # ... junction labels near the eastern edge
LEGEND_H = 132       # a band under the map, so nothing overlays the geometry
TARGET_PX = 1700     # longest image side we aim for when picking a scale
LABEL_LINE = 16      # pixel spacing between stacked carriageway labels


class Carriageway:
    """One direction of one road line: the thing a scenario places actors on.

    Grid towns are axis-aligned, so every travel lane is a constant ``x`` or
    a constant ``y``.  That constant, plus the heading, is exactly what a
    ``position(x:, y:, h:)`` modifier needs.
    """

    def __init__(self, axis: str, constant: float, heading: float,
                 lane_id: int, width: float):
        self.axis = axis                 # "x": runs along x; "y": runs along y
        self.constant = constant         # the coordinate held fixed
        self.heading = heading           # radians
        self.lane_id = lane_id
        self.width = width
        self.spans: List[Tuple[float, float]] = []

    @property
    def direction(self) -> str:
        deg = round(math.degrees(self.heading)) % 360
        return {0: "east (+x)", 90: "south (+y)",
                180: "west (-x)", 270: "north (-y)"}.get(deg, f"{deg} deg")

    @property
    def travel_range(self) -> Tuple[float, float]:
        lo = min(s[0] for s in self.spans)
        hi = max(s[1] for s in self.spans)
        return lo, hi

    def midpoint(self) -> Tuple[float, float]:
        """A point on the longest segment, where the on-map label goes."""
        lo, hi = max(self.spans, key=lambda s: s[1] - s[0])
        along = (lo + hi) * 0.5
        return (along, self.constant) if self.axis == "x" else (self.constant, along)

    def label(self) -> str:
        arrow = {"east (+x)": "->", "west (-x)": "<-",
                 "south (+y)": "v", "north (-y)": "^"}.get(self.direction, "")
        coord = "y" if self.axis == "x" else "x"
        return (f"{arrow} {coord}={self.constant:g}  lane {self.lane_id:+d}  "
                f"h={self.heading:+.4f}")


def carriageways(road_map: Map) -> List[Carriageway]:
    """Group the travel lanes of a map into axis-aligned carriageways."""
    found: Dict[Tuple[str, float, int, int], Carriageway] = {}
    for lane in road_map.lanes.values():
        if lane.is_junction:
            continue
        xs = [p[0] for p in lane.points]
        ys = [p[1] for p in lane.points]
        if max(ys) - min(ys) < 1e-6:
            axis, constant, span = "x", ys[0], (min(xs), max(xs))
        elif max(xs) - min(xs) < 1e-6:
            axis, constant, span = "y", xs[0], (min(ys), max(ys))
        else:
            continue                     # not axis-aligned; nothing to tabulate
        heading = lane.start_heading
        key = (axis, round(constant, 3), lane.lane_id,
               int(round(math.degrees(heading))) % 360)
        cw = found.get(key)
        if cw is None:
            cw = Carriageway(axis, constant, heading, lane.lane_id, lane.width)
            found[key] = cw
        cw.spans.append(span)
    return sorted(found.values(), key=lambda c: (c.axis, c.constant, c.lane_id))


def junction_centres(road_map: Map) -> List[Tuple[float, float]]:
    """Declared junction centres, sorted west-to-east then north-to-south."""
    declared = getattr(road_map, "junctions", None)
    if declared:
        return sorted(j.centre for j in declared)
    return sorted(((x0 + x1) * 0.5, (y0 + y1) * 0.5)
                  for x0, y0, x1, y1 in junction_boxes(road_map))


#: heading in degrees -> compass arm, in CARLA's +y-south frame
_COMPASS = {0: "E", 90: "S", 180: "W", 270: "N"}


def junction_arms(road_map: Map) -> Dict[Tuple[float, float], str]:
    """Which compass arms each junction has a road on.

    The corners and edges of a grid are not four-way: a scenario that needs
    an opposing approach or a left exit has to be staged at a junction that
    has one.
    """
    boxes = [(j.centre, j.bounding_box) for j in getattr(road_map, "junctions", [])]
    out: Dict[Tuple[float, float], set] = {c: set() for c, _ in boxes}
    for lane in road_map.lanes.values():
        if lane.is_junction:
            continue
        for s_at, incoming in ((0.0, False), (lane.length, True)):
            x, y, heading = lane.pose_at(s_at)
            deg = int(round(math.degrees(heading))) % 360
            if incoming:                     # arrives from the opposite side
                deg = (deg + 180) % 360
            arm = _COMPASS.get(deg)
            if arm is None:
                continue
            for centre, (x0, y0, x1, y1) in boxes:
                tol = lane.width
                if x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol:
                    out[centre].add(arm)
    return {c: "".join(a for a in "NESW" if a in arms) for c, arms in out.items()}


def _cluster(values: Sequence[float], tolerance: float) -> List[List[float]]:
    """Group nearby coordinates -- the carriageways of one road line."""
    out: List[List[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tolerance:
            out[-1].append(v)
        else:
            out.append([v])
    return out


# --------------------------------------------------------------------------
# picture
# --------------------------------------------------------------------------

def render_map(road_map: Map, path: str, scale: Optional[float] = None,
               show_spawn_points: bool = False, title: str = "") -> str:
    """Draw the network with a metric grid and coordinate labels."""
    if pygame is None:
        raise RuntimeError("pygame is required to render a map reference sheet")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.font.init()

    min_x, min_y, max_x, max_y = road_map.bounds
    pad = 8.0
    min_x, min_y, max_x, max_y = min_x - pad, min_y - pad, max_x + pad, max_y + pad
    if scale is None:
        scale = TARGET_PX / max(max_x - min_x, max_y - min_y)
        scale = max(4.0, min(16.0, scale))

    map_w = int((max_x - min_x) * scale)
    map_h = int((max_y - min_y) * scale)
    surf = pygame.Surface((map_w + MARGIN_LEFT + MARGIN_RIGHT,
                           map_h + MARGIN_TOP + LEGEND_H))
    surf.fill(GRASS)

    def to_px(px: float, py: float) -> Tuple[float, float]:
        return ((px - min_x) * scale + MARGIN_LEFT,
                (py - min_y) * scale + MARGIN_TOP)

    font = pygame.font.SysFont("dejavusansmono,monospace", 14)
    font_small = pygame.font.SysFont("dejavusansmono,monospace", 12)
    font_big = pygame.font.SysFont("dejavusansmono,monospace", 22, bold=True)

    _draw_grid(surf, to_px, font_small, min_x, min_y, max_x, max_y,
               map_h + MARGIN_TOP)
    draw_network(surf, road_map, to_px, scale)

    if show_spawn_points:
        for tf in road_map.get_spawn_points():
            _spawn_arrow(surf, to_px, tf, scale)

    for cx, cy in junction_centres(road_map):
        x, y = to_px(cx, cy)
        pygame.draw.circle(surf, ALERT, (int(x), int(y)), 5)
        pygame.draw.circle(surf, (10, 12, 16), (int(x), int(y)), 5, 1)
        arms = junction_arms(road_map).get((cx, cy), "")
        _text(surf, font, f"({cx:g}, {cy:g}) {arms}", int(x) + 9, int(y) - 8,
              ALERT if len(arms) == 4 else HUD_DIM)

    _label_carriageways(surf, road_map, to_px, font_small, scale)
    _legend(surf, road_map, font, font_big, title, scale,
            map_h + MARGIN_TOP)

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    pygame.image.save(surf, path)
    return path


def _draw_grid(surf, to_px, font, min_x, min_y, max_x, max_y,
               map_bottom: int) -> None:
    v = math.ceil(min_x / GRID_STEP) * GRID_STEP
    while v <= max_x:
        x, _ = to_px(v, 0.0)
        pygame.draw.line(surf, GRID_LINE, (x, MARGIN_TOP), (x, map_bottom))
        _text(surf, font, f"x={v:g}", int(x) + 3, MARGIN_TOP - 16, GRID_TEXT)
        v += GRID_STEP
    v = math.ceil(min_y / GRID_STEP) * GRID_STEP
    while v <= max_y:
        _, y = to_px(0.0, v)
        pygame.draw.line(surf, GRID_LINE, (MARGIN_LEFT, y),
                         (surf.get_width() - MARGIN_RIGHT, y))
        _text(surf, font, f"y={v:g}", 6, int(y) - 6, GRID_TEXT)
        v += GRID_STEP


def _label_carriageways(surf, road_map, to_px, font, scale: float) -> None:
    """Annotate every carriageway once, stacked clear of its own road.

    Labels for a north-south road cannot sit on the road: four of them share
    one along-axis midpoint and would print on top of each other.  They are
    stacked beside the carriageway group instead.
    """
    ways = carriageways(road_map)
    for cw in [c for c in ways if c.axis == "x"]:
        mx, my = cw.midpoint()
        x, y = to_px(mx, my)
        _text(surf, font, cw.label(), int(x) - 110, int(y) - 15, HUD_FG)

    vertical = [c for c in ways if c.axis == "y"]
    if not vertical:
        return
    lane_w = max(c.width for c in vertical)
    for group in _cluster([c.constant for c in vertical], lane_w * 1.6):
        members = sorted((c for c in vertical if c.constant in group),
                         key=lambda c: c.constant)
        anchor_x = max(group) + lane_w
        lo, hi = members[0].travel_range
        along = lo + (hi - lo) * 0.35
        base_x, base_y = to_px(anchor_x, along)
        for i, cw in enumerate(members):
            _text(surf, font, cw.label(), int(base_x) + 6,
                  int(base_y) + i * LABEL_LINE, HUD_FG)


def _spawn_arrow(surf, to_px, tf, scale: float) -> None:
    x, y = to_px(tf.location.x, tf.location.y)
    yaw = math.radians(tf.rotation.yaw)
    tip = (x + math.cos(yaw) * 1.8 * scale, y + math.sin(yaw) * 1.8 * scale)
    pygame.draw.line(surf, OK, (x, y), tip, 2)
    pygame.draw.circle(surf, OK, (int(x), int(y)), 3)


def _legend(surf, road_map, font, font_big, title: str, scale: float,
            top: int) -> None:
    """A band under the map, so the legend never covers the geometry."""
    pygame.draw.rect(surf, (10, 12, 16),
                     pygame.Rect(0, top, surf.get_width(), LEGEND_H))
    surf.blit(font_big.render(title or road_map.name, True, ACCENT),
              (MARGIN_LEFT, top + 12))
    lines = [
        f"{len(road_map.lanes)} lanes (junction connectors included)   "
        f"{len(junction_centres(road_map))} junctions   "
        f"{len(road_map.get_spawn_points())} default spawn points   "
        f"{scale:.1f} px/m, grid every {GRID_STEP:g} m",
        "+x east (right), +y SOUTH (down): CARLA's frame. A left turn goes "
        "toward -y, and h grows clockwise on this page -- southbound is "
        "h=+1.5708, northbound h=-1.5708.",
        "red dot = junction centre (paste as a distance reference)   "
        "yellow = road centre line   white dashes = lane divider   "
        "green arrow = default spawn point",
    ]
    y = top + 48
    for text in lines:
        surf.blit(font.render(text, True, HUD_DIM), (MARGIN_LEFT, y))
        y += font.get_height() + 6


def _text(surf, font, text: str, x: int, y: int, colour) -> None:
    surf.blit(font.render(text, True, colour), (x, y))


# --------------------------------------------------------------------------
# companion table
# --------------------------------------------------------------------------

def describe_map(road_map: Map, town_key: str = "") -> str:
    """Markdown reference: the same numbers, in copy-pasteable form."""
    town = BUILTIN_TOWNS.get(town_key)
    min_x, min_y, max_x, max_y = road_map.bounds
    out: List[str] = []
    out.append(f"# Town `{road_map.name}`\n")
    if town is not None:
        out.append(f"{town.description}\n")
    out.append(f"- extent: x in [{min_x:.1f}, {max_x:.1f}], "
               f"y in [{min_y:.1f}, {max_y:.1f}]")
    out.append(f"- lanes: {len(road_map.lanes)} "
               f"(including junction connectors)")
    out.append(f"- spawn points: {len(road_map.get_spawn_points())}")
    if town is not None:
        out.append(f"- lane width: {town.lane_width:g} m, "
                   f"{town.lanes_per_dir} lanes each way")
    out.append("")
    out.append("Coordinates are CARLA's: +x east, +y **south**, `h` in radians "
               "growing clockwise when the map is drawn with +y down. A left "
               "turn goes toward -y.\n")

    out.append("## Junctions\n")
    out.append("`arms` lists the compass directions that have a road. Only a "
               "junction with all four (`NESW`) can stage a conflict that "
               "needs an opposing approach, such as an unprotected left.\n")
    arms = junction_arms(road_map)
    out.append("| centre (x, y) | arms | marker to use as a distance reference |")
    out.append("|---|---|---|")
    for cx, cy in junction_centres(road_map):
        out.append(f"| ({cx:g}, {cy:g}) | `{arms.get((cx, cy), '?')}` | "
                   f"`position(x: {cx:.2f}, y: {cy:.2f}, z: -0.40, at: start)` |")
    out.append("")

    out.append("## Carriageways\n")
    out.append("A vehicle placed on one of these, with the matching `h`, "
               "drives along it.\n")
    out.append("| runs along | fixed coord | direction | lane | `h` (rad) | "
               "travel range |")
    out.append("|---|---|---|---|---|---|")
    for cw in carriageways(road_map):
        lo, hi = cw.travel_range
        fixed = f"y = {cw.constant:g}" if cw.axis == "x" else f"x = {cw.constant:g}"
        along = "x" if cw.axis == "x" else "y"
        out.append(f"| {along} | {fixed} | {cw.direction} | {cw.lane_id:+d} | "
                   f"{cw.heading:+.4f} | {along} in [{lo:g}, {hi:g}] |")
    out.append("")
    out.append("`lane -1` is the lane nearest the road's centre line and "
               "`lane -2` the one outside it, following CARLA's sign "
               "convention; the positive ids are the opposing carriageway.")
    out.append("")
    out.append("Travel ranges stop at the junction boxes: the segment between "
               "two junctions is where a vehicle has straight road. Placing an "
               "actor outside every range puts it off the network, where "
               "`get_waypoint(project_to_road=True)` will snap it to whatever "
               "lane happens to be nearest.")
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render local-simulator towns as coordinate reference "
                    "sheets for scenario authoring.")
    parser.add_argument("--town", action="append", default=None,
                        help="Town to render, repeatable (default: all).")
    parser.add_argument("--out-dir", default="maps",
                        help="Directory for the PNG and Markdown output "
                             "(default: maps).")
    parser.add_argument("--scale", type=float, default=None,
                        help="Pixels per metre (default: fit ~1700 px).")
    parser.add_argument("--spawn-points", action="store_true",
                        help="Mark every default spawn point with a heading "
                             "arrow.")
    args = parser.parse_args(argv)

    towns = args.town or town_names()
    unknown = [t for t in towns if t not in BUILTIN_TOWNS]
    if unknown:
        parser.error(f"unknown town(s) {unknown}; have {town_names()}")

    for name in towns:
        road_map = load_town(name)
        png = os.path.join(args.out_dir, f"{name}.png")
        render_map(road_map, png, scale=args.scale,
                   show_spawn_points=args.spawn_points, title=f"town: {name}")
        md = os.path.join(args.out_dir, f"{name}.md")
        with open(md, "w") as fh:
            fh.write(describe_map(road_map, name))
        print(f"[mapview] {name}: {png}, {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
