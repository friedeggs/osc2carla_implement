"""A standalone simulator and bird's-eye renderer for compiled scenarios.

This package is the second execution backend for ``osc2carla``.  The compiler
front end, the semantic analysis and the py_trees behaviour tree are shared
verbatim with the CARLA backend; what changes is only *where* the tree's
atomic behaviours actuate.  Instead of an RPC call into a UE4 server, they
drive a few hundred lines of planar vehicle dynamics, and instead of an RGB
camera the run is drawn top-down with pygame.

Nothing in here imports ``carla``.  The point is a laptop-sized loop:

    python -m osc2carla scenarios/closed_loop_demo.osc --backend pygame

What it is not: a replacement for CARLA measurements.  There is no
photorealistic sensor model, no tyre model, no real town geometry -- see
``towns.py`` -- so it previews and debugs scenario logic rather than
reproducing CARLA numbers.
"""
from __future__ import annotations

from . import api  # noqa: F401
from .roadmap import Lane, Map, Waypoint  # noqa: F401
from .towns import BUILTIN_TOWNS, load_town, town_names  # noqa: F401
from .world import Client, World  # noqa: F401

__all__ = ["api", "BUILTIN_TOWNS", "Client", "Lane", "Map", "Waypoint",
           "World", "load_town", "town_names"]
