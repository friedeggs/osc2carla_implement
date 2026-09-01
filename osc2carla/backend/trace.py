"""Per-tick trace recording for the harness's metrics package.

The centralized harness (`scenario_orchestration`) now evaluates scenario
success on the realized trajectory rather than accepting each method's
self-report -- for this baseline, the self-report was
`collision_occurred == expect_collision`, which asks "did the scripted crash
happen" and therefore makes a competent ego policy look like a scenario
failure. Evaluating on the trajectory instead needs a per-tick state series,
and this run writes none.

This module is the whole of this repository's side of that contract. It finds
`metrics.recording` in the harness that issued the run and imports the recorder
from there rather than reimplementing the format, so the two cannot drift.

Three scenario-specific notes.

*Reference paths* are sampled once, before the run starts, so an actor that
stops short still has a route to be measured against. The ego under an external
policy takes the *plan* that policy is conditioned on (`backend/route.py`), so
the recorded path and the route the policy is handed are one object and cannot
disagree -- they did before that plan existed: this file walked straight through
Town10HD_Opt's junction while the policy's own per-tick walk had turned left.
Every other actor keeps the `waypoint.next()[0]` walk from its spawn transform,
which remains the rule the compiled `drive()` behaviour steers by.

*The conflict point* is not inferred: the benchmark `.osc` files declare a
`conflict_point` stationary object at the exact lane crossing, and its
placement is the authoritative answer. It is recorded as a zone.

*States come from one world snapshot*, not from per-actor `get_transform()`
calls: `world.get_snapshot()` is served from the frame the client already holds
and returns the whole cast, so recording six actors costs one call instead of
eighteen. It is also the reading that is guaranteed self-consistent -- every
actor as of the same frame -- which per-actor calls are not.

It is *not* a determinism measure, and an earlier version of this comment said
it was, on four samples that happened to agree. This scenario is not
reproducible run to run with or without any recorder: five repeats with tracing
disabled gave five different travelled distances (26.5 to 45.1 m) and two
different first-collision impulses, and the three seeds recorded before this
module existed show the same spread. The collision-level metrics
(`collision_occurred`, `first_collision_time`) are stable across all of them;
what varies is the post-impact motion of a wrecked vehicle. See
`experiments/001-near-collision-scenario-success.md` section 8.

*Nothing here may fail the run.* Every entry point swallows its own errors.
"""
from __future__ import annotations

import math
import os
import sys

TRACE_RATE_HZ = 10.0

#: How far ahead a reference path is projected, and at what spacing. 120 m is
#: past the junction on every benchmark scenario at the declared speeds.
ROUTE_LENGTH_M = 150.0
ROUTE_STEP_M = 1.0


def find_harness_root(start=None):
    """Where the harness that issued this run lives. Only `metrics.recording`
    is imported from it, and that subpackage is stdlib-only by contract."""
    override = os.environ.get("OSC2CARLA_HARNESS_ROOT")
    if override and os.path.isdir(override):
        return override
    path = os.path.abspath(start or os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))
    while True:
        if os.path.isdir(os.path.join(path, "metrics", "recording")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


#: The name the harness's recording package is loaded under. Deliberately not
#: `metrics`: a method repository can already have a top-level module by that
#: name (the orchestration port does), and then `import metrics.recording`
#: resolves to it. Loading by path under a private name also makes it
#: structural, rather than promised, that nothing else in the harness is
#: imported.
_RECORDING_MODULE = "harness_trace_recording"


def _load_recorder_class(root):
    import importlib.util
    if _RECORDING_MODULE in sys.modules:
        return sys.modules[_RECORDING_MODULE].TraceRecorder
    pkg_dir = os.path.join(root, "metrics", "recording")
    spec = importlib.util.spec_from_file_location(
        _RECORDING_MODULE, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    module = importlib.util.module_from_spec(spec)
    sys.modules[_RECORDING_MODULE] = module
    spec.loader.exec_module(module)
    return module.TraceRecorder


def make_recorder(output_dir, rate_hz=TRACE_RATE_HZ, context=None):
    """`(recorder, note)`; `recorder` is None when the harness is not found."""
    root = find_harness_root()
    if root is None:
        return None, ("no metrics/recording found above this repository; "
                      "no per-tick trace was written")
    try:
        TraceRecorder = _load_recorder_class(root)
    except Exception as exc:                       # pragma: no cover
        return None, "metrics/recording could not be loaded from %s: %s" % (
            root, exc)
    try:
        return TraceRecorder(output_dir, rate_hz=rate_hz,
                             context=dict(context or {})), None
    except Exception as exc:                       # pragma: no cover
        return None, "TraceRecorder could not be opened: %s" % (exc,)


class SceneTracer(object):
    """Binds the recorder to one `ExecutionContext` and its CARLA actors."""

    def __init__(self, recorder, ctx, carla_map, ego_binding=None):
        self.rec = recorder
        self.ctx = ctx
        self.map = carla_map
        self.ego_binding = ego_binding
        self.actors = {}                # binding name -> CARLA actor
        self.errors = []
        self._warned_no_snapshot = False

    # -- static ----------------------------------------------------------- #

    def declare_scene(self):
        if self.rec is None:
            return
        try:
            self._bind()
            self._declare_actors()
            self._declare_paths()
            self._declare_conflict_point()
        except Exception as exc:                    # pragma: no cover
            self.errors.append("scene: %s" % (exc,))

    def _bind(self):
        for binding in self.ctx.annotated.scenario.actors:
            if binding.type_name not in ("vehicle", "stationary_object"):
                continue
            actor = self.ctx.actor(binding.name)
            if actor is not None:
                self.actors[binding.name] = actor

    def _declare_actors(self):
        for name, actor in self.actors.items():
            extent = None
            bb = getattr(actor, "bounding_box", None)
            if bb is not None and getattr(bb, "extent", None) is not None:
                extent = (2.0 * bb.extent.x, 2.0 * bb.extent.y)
            self.rec.declare(name, extent=extent,
                             type_id=getattr(actor, "type_id", None),
                             role=name,
                             is_ego=(name == self.ego_binding),
                             carla_id=getattr(actor, "id", None))

    def _declare_paths(self):
        """Each vehicle's reference path, recorded once before the run, so an
        actor that stops short still has the route it was on.

        The ego under an external policy is declared from the *plan* that policy
        is actually being conditioned on (`backend/route.py`), so the recorded
        path and the route the policy sees cannot disagree -- they used to: the
        path here walked straight through Town10HD_Opt's junction while the
        policy's own per-tick walk had turned left.

        Every other vehicle keeps the `waypoint.next()[0]` walk, because that is
        what still steers them: `drive()` recomputes its steering reference from
        the actor's live position every tick (`atomic_behaviors.py`), so a path
        declared by any other rule would describe something they do not do.
        """
        if self.map is None:
            return
        for name, actor in self.actors.items():
            if not str(getattr(actor, "type_id", "")).startswith("vehicle."):
                continue
            if self.ego_binding is not None and name == self.ego_binding:
                poly = self._planned_route(actor)
            else:
                poly = self._route_from(actor)
            if len(poly) >= 2:
                self.rec.declare_path(name, poly, source="route")

    def _planned_route(self, actor):
        """The one plan this ego drives, as a world-frame polyline."""
        try:
            from . import route as route_plan
            # No length of its own: the plan's reach is the plan's property,
            # and the policy conditioned on it needs more road than a 150 m
            # reference path does.
            plan = route_plan.plan_for(self.map, actor)
        except Exception as exc:                     # pragma: no cover
            self.errors.append("planned route: %s" % (exc,))
            return self._route_from(actor)
        return plan.world_polyline()

    def _route_from(self, actor):
        try:
            wp = self.map.get_waypoint(actor.get_location(), project_to_road=True)
        except Exception:                            # pragma: no cover
            return []
        if wp is None:
            return []
        loc = actor.get_location()
        out = [(loc.x, loc.y)]
        s = 0.0
        seen = set()
        while s < ROUTE_LENGTH_M:
            try:
                nxt = wp.next(ROUTE_STEP_M)
            except Exception:                        # pragma: no cover
                break
            if not nxt:
                break
            wp = nxt[0]
            key = (round(wp.transform.location.x, 2),
                   round(wp.transform.location.y, 2))
            if key in seen:
                # A loop connector would otherwise spin here forever.
                break
            seen.add(key)
            out.append((wp.transform.location.x, wp.transform.location.y))
            s += ROUTE_STEP_M
        return out

    def _declare_conflict_point(self):
        """The benchmark scenarios place a ground decal at the exact lane
        crossing and write their own junction-occupancy conditions against it,
        so it is the authoritative conflict point rather than an inference."""
        actor = self.actors.get("conflict_point")
        if actor is None:
            return
        loc = actor.get_location()
        self.rec.declare_zone("conflict_point", "conflict_point",
                              center=(loc.x, loc.y), radius=8.0,
                              source="declared by the scenario")

    # -- per tick ---------------------------------------------------------- #

    def tick(self, sim_time):
        """One tick, read from the frame the client already holds.

        `world.get_snapshot()` is one call for the whole cast, and every actor
        in it is read as of the same frame. Where a snapshot is unavailable the
        per-actor path is used and the fallback is recorded, so a trace can
        never look snapshot-sourced when it is not.
        """
        if self.rec is None:
            return
        snap = None
        world = getattr(self.ctx, "world", None)
        if world is not None:
            try:
                snap = world.get_snapshot()
            except Exception:                        # pragma: no cover
                snap = None
        if snap is None and not self._warned_no_snapshot:
            self._warned_no_snapshot = True
            self.errors.append("no world snapshot; fell back to per-actor reads")
        states = {}
        for name, actor in self.actors.items():
            if not str(getattr(actor, "type_id", "")).startswith("vehicle."):
                continue
            row = self._from_snapshot(snap, actor) if snap is not None \
                else self._from_actor(actor)
            if row is not None:
                states[name] = row
        self.rec.tick(sim_time, states)

    @staticmethod
    def _from_snapshot(snap, actor):
        try:
            st = snap.find(actor.id)
        except Exception:                            # pragma: no cover
            st = None
        if st is None:
            return None                              # not alive this frame
        tf, v, a = st.get_transform(), st.get_velocity(), st.get_acceleration()
        return (tf.location.x, tf.location.y, tf.rotation.yaw,
                v.x, v.y, a.x, a.y)

    @staticmethod
    def _from_actor(actor):
        try:
            tf = actor.get_transform()
            v = actor.get_velocity()
            a = actor.get_acceleration()
        except Exception:                            # destroyed mid-run
            return None
        return (tf.location.x, tf.location.y, tf.rotation.yaw,
                v.x, v.y, getattr(a, "x", None), getattr(a, "y", None))

    def note_collisions(self, hits, ego_binding):
        """Collision events, read off the metrics collector's own sensor list
        so the trace and `osc2carla_metrics.json` cannot disagree."""
        if self.rec is None:
            return
        by_type = {}
        for name, actor in self.actors.items():
            by_type.setdefault(str(getattr(actor, "type_id", "")), name)
        for h in hits:
            other = h.get("other_role") or by_type.get(str(h.get("other")))
            actors = [a for a in (ego_binding, other) if a]
            self.rec.event(float(h.get("sim_time") or 0.0), "collision",
                           actors=actors, other_type=h.get("other"),
                           impulse=h.get("impulse_mag"))

    def close(self):
        if self.rec is None:
            return None
        return self.rec.close()
