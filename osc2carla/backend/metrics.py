"""Run metrics, with collision occurrence as the headline signal.

Collision occurrence is used here as a proxy for *scenario execution success*:
each benchmark scenario is written with an intended outcome, and the question
these runs ask is whether swapping the ego's controller still produces it.

The distinction matters because the proxy cuts both ways.  For the three
crash scenarios the scenario has executed correctly when the ego IS hit -- a
run with no collision means the scripted conflict never actually developed.
For ``stop_sign`` the scenario has executed correctly when the ego is NOT hit.
So "collision occurred" is the raw measurement and "success" is that
measurement compared against the scenario's declared intent; the intent lives
with the experiment configuration, not in here, because it is a property of
the scenario rather than of the simulator.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

from .simapi import sim as carla


class MetricsCollector:
    """Collision + motion statistics for one binding over one run."""

    def __init__(self, world, actor, binding: str, attach_sensor: bool = True):
        self.world = world
        self.actor = actor
        self.binding = binding
        self._sensor = None
        self._collisions: List[Dict[str, Any]] = []
        self._sim_time = 0.0
        self._distance = 0.0
        self._speed_sum = 0.0
        self._speed_n = 0
        self._max_speed = 0.0
        self._min_leader_gap: Optional[float] = None
        self._last_loc = None
        self._external: Optional[List[Dict[str, Any]]] = None

        if attach_sensor and carla and world is not None and actor is not None:
            bp = world.get_blueprint_library().find("sensor.other.collision")
            self._sensor = world.spawn_actor(bp, carla.Transform(), attach_to=actor)
            self._sensor.listen(self._on_collision)

    def use_external_collisions(self, collisions: List[Dict[str, Any]]) -> None:
        """Read collisions from an already-attached sensor (e.g. the Recorder)."""
        self._external = collisions

    def _on_collision(self, event) -> None:
        imp = event.normal_impulse
        self._collisions.append({
            "frame": event.frame,
            "sim_time": self._sim_time,
            "other": event.other_actor.type_id,
            "other_role": (getattr(event.other_actor, "attributes", {}) or {}).get("role_name", ""),
            "impulse_mag": math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2),
        })

    @property
    def collisions(self) -> List[Dict[str, Any]]:
        return self._external if self._external is not None else self._collisions

    def tick(self, sim_time: float, leader_gap: Optional[float] = None) -> None:
        self._sim_time = sim_time
        if self.actor is None:
            return
        loc = self.actor.get_location()
        if self._last_loc is not None:
            self._distance += math.sqrt((loc.x - self._last_loc[0]) ** 2
                                        + (loc.y - self._last_loc[1]) ** 2)
        self._last_loc = (loc.x, loc.y)
        v = self.actor.get_velocity()
        sp = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        self._speed_sum += sp
        self._speed_n += 1
        self._max_speed = max(self._max_speed, sp)
        if leader_gap is not None:
            if self._min_leader_gap is None or leader_gap < self._min_leader_gap:
                self._min_leader_gap = leader_gap

    @staticmethod
    def _is_vehicle(hit: Dict[str, Any]) -> bool:
        return str(hit.get("other", "")).startswith("vehicle.")

    def summary(self, **extra: Any) -> Dict[str, Any]:
        all_hits = self.collisions
        # Scenario scaffolding is not a scenario outcome. The benchmark files
        # place a ground-decal marker at the conflict point as a distance
        # reference, and the ego sometimes drives over it -- CARLA does report
        # a contact for static.prop.dirtdebris01, contrary to what a single
        # drive-over test suggested. Counting that as a collision would make a
        # clean run look like a crash, so the headline metric counts only
        # vehicle-vs-vehicle contacts and static contacts are reported apart.
        hits = [h for h in all_hits if self._is_vehicle(h)]
        static_hits = [h for h in all_hits if not self._is_vehicle(h)]
        impulses = [h.get("impulse_mag", 0.0) for h in hits]
        # Sensors report a contact per physics substep while bodies stay in
        # contact, so the event count measures how long the crash lasted as
        # much as how many distinct crashes there were. Partner identity is
        # the more reliable descriptor.
        partners = sorted({h.get("other", "") for h in hits})
        roles = sorted({h.get("other_role", "") for h in hits if h.get("other_role")})
        times = [h.get("sim_time", 0.0) for h in hits if h.get("sim_time") is not None]
        out: Dict[str, Any] = {
            "binding": self.binding,
            "collision_occurred": bool(hits),
            "n_collision_events": len(hits),
            "first_collision_time": min(times) if times else None,
            "peak_impulse": max(impulses) if impulses else 0.0,
            "total_impulse": sum(impulses),
            "collision_partners": partners,
            "collision_partner_roles": roles,
            "distance_travelled_m": round(self._distance, 2),
            "mean_speed_mps": round(self._speed_sum / self._speed_n, 3) if self._speed_n else 0.0,
            "max_speed_mps": round(self._max_speed, 3),
            "min_leader_gap_m": (round(self._min_leader_gap, 2)
                                 if self._min_leader_gap is not None else None),
            # Instrumentation contacts, excluded from the headline metric.
            "n_static_contacts": len(static_hits),
            "static_contact_partners": sorted({h.get("other", "") for h in static_hits}),
        }
        out.update(extra)
        return out

    def write(self, path: str, **extra: Any) -> Dict[str, Any]:
        data = self.summary(**extra)
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        return data

    def close(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
                self._sensor.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._sensor = None
