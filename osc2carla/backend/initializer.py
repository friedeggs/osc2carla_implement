"""Spawn CARLA actors with the placement rules from at:start modifiers."""
from __future__ import annotations

import random
import sys
from typing import Any, Dict, List, Optional

try:
    import carla  # type: ignore
except Exception:  # noqa: BLE001
    carla = None  # type: ignore

from ..frontend import nodes
from ..middle import AnnotatedScenario
from .context import ExecutionContext


def _log(msg: str) -> None:
    print(f"[initializer] {msg}", file=sys.stderr)


class ScenarioInitializer:
    def __init__(self, world: Any, carla_map: Any, annotated: AnnotatedScenario,
                 ctx: ExecutionContext):
        self.world = world
        self.map = carla_map
        self.annotated = annotated
        self.ctx = ctx
        self._spawned: List[Any] = []
        # binding name -> max forward distance another actor is placed ahead of it.
        # Used to pick a spawn point for the reference actor that has enough
        # straight road ahead so the dependent actor can be placed reliably.
        self._referenced_ahead: Dict[str, float] = {}
        # binding name -> carla.Transform it was spawned at. In synchronous mode
        # actor.get_transform() returns the origin until the first world.tick(),
        # so dependent actors must resolve positions against this instead.
        self._spawn_transforms: Dict[str, Any] = {}

    def initialise(self) -> List[Any]:
        self._referenced_ahead = self._scan_referenced_distances()
        for binding in self.annotated.scenario.actors:
            self._spawn_binding(binding)
        return self._spawned

    def _scan_referenced_distances(self) -> Dict[str, float]:
        """Find, for each binding, how far ahead another actor is anchored to it.

        Mirrors the collision demo, where the ego spawn point is
        chosen so that ~25 m of road exists ahead for the opposing actor.
        """
        needs: Dict[str, float] = {}
        for binding in self.annotated.scenario.actors:
            for mod in self._collect_init_modifiers(binding):
                if mod.name != "position":
                    continue
                named = mod.args.named
                ref = None
                if "ahead_of" in named:
                    ref = self._eval_ref_name(named["ahead_of"])
                elif "behind" in named:
                    ref = self._eval_ref_name(named["behind"])
                if ref is None or "distance" not in named:
                    continue
                try:
                    dist = float(self.ctx.eval(named["distance"]))
                except Exception:  # noqa: BLE001
                    continue
                needs[ref] = max(needs.get(ref, 0.0), dist)
        return needs

    def _eval_ref_name(self, expr) -> Optional[str]:
        try:
            val = self.ctx.eval(expr)
        except Exception:  # noqa: BLE001
            return None
        if isinstance(val, str):
            return val
        return getattr(val, "_binding", None)

    def _spawn_binding(self, binding: nodes.ScenarioActorBinding) -> None:
        if carla is None:
            return
        attrs: Dict[str, Any] = getattr(binding, "attributes", {}) or {}
        blueprint_id = attrs.get("model")
        if not blueprint_id:
            return
        bp_lib = self.world.get_blueprint_library()
        try:
            bp = bp_lib.find(blueprint_id)
        except IndexError:
            _log(f"blueprint '{blueprint_id}' not found for '{binding.name}'")
            return
        role_name = attrs.get("name", binding.name)
        bp.set_attribute("role_name", role_name)
        color = attrs.get("color")
        if color and bp.has_attribute("color"):
            bp.set_attribute("color", color)

        init_mods = self._collect_init_modifiers(binding)

        transform, is_relative = self._compute_transform(binding, init_mods)
        if transform is None:
            _log(f"no transform computed for '{binding.name}'")
            return

        actor = self._spawn_at(bp, transform, is_relative)
        if actor is None and not is_relative:
            wps = self.map.generate_waypoints(8.0)
            random.shuffle(wps)
            for wp in wps:
                t = wp.transform
                t.location.z += 0.5
                actor = self.world.try_spawn_actor(bp, t)
                if actor is not None:
                    break
        if actor is None:
            _log(f"failed to spawn '{binding.name}' ({blueprint_id})")
            return
        self._spawned.append(actor)
        self._spawn_transforms[binding.name] = transform
        self.ctx.bind_actor(binding.name, role_name, actor)
        loc = transform.location
        _log(f"spawned '{binding.name}' ({blueprint_id}) at "
             f"({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) "
             f"yaw={transform.rotation.yaw:.0f} relative={is_relative}")

    def _spawn_at(self, bp, transform, is_relative):
        """Spawn at a transform, nudging upward on overlap instead of scattering."""
        attempts = 6 if is_relative else 1
        t = carla.Transform(
            carla.Location(x=transform.location.x,
                           y=transform.location.y,
                           z=transform.location.z),
            transform.rotation,
        )
        for _ in range(attempts):
            try:
                actor = self.world.try_spawn_actor(bp, t)
            except Exception:  # noqa: BLE001
                actor = None
            if actor is not None:
                return actor
            t.location.z += 0.4
        return None

    def _collect_init_modifiers(self, binding: nodes.ScenarioActorBinding) -> List[nodes.Modifier]:
        do = self.annotated.scenario.do
        if do is None:
            return []
        mods: List[nodes.Modifier] = []
        for ac in _iter_action_calls(do.body):
            if ac.actor != binding.name or ac.with_block is None:
                continue
            for m in ac.with_block.modifiers:
                if m.at == "start":
                    mods.append(m)
        return mods

    def _compute_transform(self, binding: nodes.ScenarioActorBinding,
                            init_mods: List[nodes.Modifier]):
        if carla is None or self.map is None:
            return None, False
        position_mod = next((m for m in init_mods if m.name == "position"), None)
        transform = None
        is_relative = False

        if position_mod is not None:
            named = position_mod.args.named
            if "x" in named:
                x = float(self.ctx.eval(named["x"]))
                y = float(self.ctx.eval(named["y"]))
                z = float(self.ctx.eval(named["z"]))
                h = float(self.ctx.eval(named["h"])) if "h" in named else 0.0
                transform = carla.Transform(
                    carla.Location(x=x, y=y, z=z + 0.5),
                    carla.Rotation(yaw=h * 180.0 / 3.14159265),
                )
                is_relative = True
            else:
                ref_tf = None
                step_sign = 0  # +1 ahead, -1 behind
                if "ahead_of" in named:
                    ref_tf = self._resolve_ref_transform(named["ahead_of"])
                    step_sign = +1
                elif "behind" in named:
                    ref_tf = self._resolve_ref_transform(named["behind"])
                    step_sign = -1
                if ref_tf is not None and "distance" in named:
                    distance = float(self.ctx.eval(named["distance"]))
                    ref_wp = self.map.get_waypoint(ref_tf.location, project_to_road=True)
                    if ref_wp is not None:
                        steps = ref_wp.next(distance) if step_sign > 0 else ref_wp.previous(distance)
                        if steps:
                            t = steps[0].transform
                            t = carla.Transform(
                                carla.Location(t.location.x, t.location.y,
                                               t.location.z + 0.5),
                                t.rotation,
                            )
                            transform = t
                            is_relative = True

            if transform is not None and "facing" in named:
                ref_actor = self._resolve_ref_actor(named["facing"])
                if ref_actor is not None:
                    transform = carla.Transform(
                        transform.location,
                        carla.Rotation(
                            pitch=transform.rotation.pitch,
                            yaw=transform.rotation.yaw + 180.0,
                            roll=transform.rotation.roll,
                        ),
                    )

        if transform is not None:
            return transform, is_relative

        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            return None, False
        required_ahead = self._referenced_ahead.get(binding.name, 0.0)
        if required_ahead > 0.0:
            chosen = self._spawn_point_with_road_ahead(spawn_points, required_ahead)
            if chosen is not None:
                return chosen, False
        return spawn_points[hash(binding.name) % len(spawn_points)], False

    def _spawn_point_with_road_ahead(self, spawn_points, distance):
        """Pick a spawn point that has at least ``distance`` of road ahead.

        Mirrors the search in ``record_scenario_collision.py`` so a dependent
        actor placed ``distance`` ahead lands on connected, drivable road.
        """
        margin = distance + 5.0
        for sp in spawn_points:
            wp = self.map.get_waypoint(sp.location, project_to_road=True)
            if wp is None:
                continue
            if wp.next(margin):
                return sp
        return None

    def _resolve_ref_actor(self, expr):
        ref_name = self.ctx.eval(expr)
        if isinstance(ref_name, str):
            return self.ctx.actor(ref_name)
        if hasattr(ref_name, "carla_actor"):
            return ref_name.carla_actor
        return None

    def _resolve_ref_transform(self, expr):
        """Transform of the referenced binding.

        Prefers the transform the actor was spawned at (valid before the first
        world.tick()); falls back to the live actor transform.
        """
        name = self._eval_ref_name(expr)
        if name is not None and name in self._spawn_transforms:
            return self._spawn_transforms[name]
        actor = self._resolve_ref_actor(expr)
        if actor is not None:
            return actor.get_transform()
        return None


def _iter_action_calls(member) -> List[nodes.ActionCall]:
    out: List[nodes.ActionCall] = []
    if isinstance(member, nodes.Composition):
        for m in member.members:
            out.extend(_iter_action_calls(m))
    elif isinstance(member, nodes.ActionCall):
        out.append(member)
    return out
