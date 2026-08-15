"""End-to-end driver for osc2carla."""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import py_trees

from .frontend import nodes
from .middle import analyse


def _scenario_sim_duration(member, ctx) -> float:
    """Largest ``wait elapsed(<t>)`` in the do-body, used to bound the run.

    The collision scenario uses ``wait elapsed(22s)`` as an explicit
    termination guard; honour it so the recording ends with the scenario
    instead of running to the wall-clock timeout.
    """
    best = 0.0
    if isinstance(member, nodes.Composition):
        for m in member.members:
            best = max(best, _scenario_sim_duration(m, ctx))
    elif isinstance(member, nodes.Wait):
        cond = member.condition
        if isinstance(cond, nodes.Elapsed):
            try:
                val = ctx.eval(cond.duration)
                best = max(best, float(val))
            except Exception:  # noqa: BLE001
                pass
    return best


def _connect_carla(host: str, port: int, timeout: float):
    import carla  # type: ignore
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client


def _maybe_load_world(client, map_name: str):
    world = client.get_world()
    cur_map = world.get_map().name
    if cur_map.endswith(map_name) or map_name.endswith(cur_map.split("/")[-1]):
        return world
    return client.load_world(map_name)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OpenSCENARIO 2 compiler/runtime for CARLA")
    parser.add_argument("scenario", help="Path to .osc file")
    parser.add_argument("--stdlib", default=None, help="Override stdlib directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--carla-timeout", type=float, default=240.0)
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="Max wall time to run the scenario tree (seconds)")
    parser.add_argument("--sim-duration", type=float, default=None,
                        help="Simulated seconds to run before stopping. Defaults to "
                             "the largest 'wait elapsed(...)' guard in the scenario.")
    parser.add_argument("--fixed-dt", type=float, default=0.05)
    parser.add_argument("--no-sync", action="store_true",
                        help="Do not switch CARLA to synchronous mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse + semantic analyse + build BT, but do not connect to CARLA")
    parser.add_argument("--emit-python", default=None,
                        help="Write a standalone Python script equivalent to the "
                             "compiled scenario at this path (or '-' for stdout) "
                             "and exit without connecting to CARLA.")
    parser.add_argument("--record-video", default=None,
                        help="If set, attach an RGB camera and a collision sensor to "
                             "--record-actor and encode an MP4 at this path.")
    parser.add_argument("--record-actor", default=None,
                        help="Scenario binding name to record from (defaults to the "
                             "first vehicle binding).")
    parser.add_argument("--record-width", type=int, default=1280)
    parser.add_argument("--record-height", type=int, default=720)
    parser.add_argument("--record-fps", type=int, default=20)
    args = parser.parse_args(argv)

    annotated = analyse(args.scenario, stdlib_dir=args.stdlib)
    if args.emit_python is not None:
        from .backend.codegen import emit
        src = emit(annotated, os.path.abspath(args.scenario))
        if args.emit_python == "-":
            sys.stdout.write(src)
        else:
            with open(args.emit_python, "w") as fh:
                fh.write(src)
            print(f"[osc2carla] wrote {args.emit_python}", file=sys.stderr)
        return 0
    print(f"[osc2carla] scenario '{annotated.scenario.name}' parsed OK", file=sys.stderr)
    print(f"  bindings : {[b.name for b in annotated.scenario.actors]}", file=sys.stderr)
    print(f"  variables: {[v.name for v in annotated.scenario.variables]}", file=sys.stderr)

    from .backend import BehaviorTreeBuilder, ExecutionContext, Recorder, ScenarioInitializer

    if args.dry_run:
        ctx = ExecutionContext(annotated)
        tree = BehaviorTreeBuilder(annotated, ctx).build()
        print("[osc2carla] behaviour tree (dry-run):", file=sys.stderr)
        print(py_trees.display.ascii_tree(tree), file=sys.stderr)
        return 0

    client = _connect_carla(args.host, args.port, args.carla_timeout)
    map_attr = next((b for b in annotated.scenario.actors if b.type_name == "map"), None)
    map_name = "Town10HD_Opt"
    if map_attr is not None:
        attrs = getattr(map_attr, "attributes", {}) or {}
        map_name = attrs.get("map_file", map_name)
    print(f"[osc2carla] loading map: {map_name}", file=sys.stderr)
    world = _maybe_load_world(client, map_name)
    carla_map = world.get_map()

    original_settings = world.get_settings()
    if not args.no_sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_dt
        world.apply_settings(settings)

    ctx = ExecutionContext(annotated, world=world, carla_map=carla_map)
    initializer = ScenarioInitializer(world, carla_map, annotated, ctx)
    initializer.initialise()
    if not args.no_sync:
        world.tick()

    recorder = None
    if args.record_video:
        rec_binding = args.record_actor
        if rec_binding is None:
            for b in annotated.scenario.actors:
                if b.type_name == "vehicle":
                    rec_binding = b.name
                    break
        rec_actor = ctx.actor(rec_binding) if rec_binding else None
        if rec_actor is not None:
            recorder = Recorder(world, rec_actor, args.record_video,
                                width=args.record_width,
                                height=args.record_height,
                                fps=args.record_fps)
            print(f"[osc2carla] recording {rec_binding!r} -> {args.record_video}",
                  file=sys.stderr)

    tree = BehaviorTreeBuilder(annotated, ctx).build()
    behaviour_tree = py_trees.trees.BehaviourTree(root=tree)
    behaviour_tree.setup(timeout=15)

    ctx.blackboard["go_signal"] = True

    sim_cap = args.sim_duration
    if sim_cap is None:
        sim_cap = _scenario_sim_duration(annotated.scenario.do.body, ctx) \
            if annotated.scenario.do is not None else 0.0
    if sim_cap and sim_cap > 0:
        print(f"[osc2carla] scenario sim duration: {sim_cap:.1f}s", file=sys.stderr)

    start = time.time()
    sim_t = 0.0
    try:
        while time.time() - start < args.timeout:
            if not args.no_sync:
                world.tick()
                sim_t += args.fixed_dt
            else:
                sim_t = time.time() - start
            ctx.advance_tick(sim_t)
            behaviour_tree.tick()
            if recorder is not None:
                recorder.tick(sim_t)
            if sim_cap and sim_cap > 0 and sim_t >= sim_cap:
                print(f"[osc2carla] reached scenario duration {sim_cap:.1f}s",
                      file=sys.stderr)
                break
            if behaviour_tree.root.status in (py_trees.common.Status.SUCCESS,
                                              py_trees.common.Status.FAILURE):
                print(f"[osc2carla] tree finished with {behaviour_tree.root.status}",
                      file=sys.stderr)
                break
    finally:
        if recorder is not None:
            out = recorder.finalize()
            if out:
                print(f"[osc2carla] wrote {out} ({len(recorder.collisions)} collision events)",
                      file=sys.stderr)
        if not args.no_sync:
            world.apply_settings(original_settings)
        for a in list(getattr(initializer, "_spawned", [])):
            try:
                a.destroy()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
