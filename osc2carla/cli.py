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
    parser.add_argument("--ego-policy", default=None,
                        help="Hand the ego's actuation to an external policy "
                             "instead of the compiled behaviour tree. Either a "
                             "built-in name ('idm', 'constant') or an import "
                             "path 'module:ClassName'. The rest of the scenario "
                             "(NPC timelines, events, monitors) is unchanged.")
    parser.add_argument("--ego-actor", default=None,
                        help="Binding the --ego-policy drives (defaults to "
                             "--record-actor, else the first vehicle binding).")
    parser.add_argument("--policy-param", action="append", default=[],
                        metavar="K=V",
                        help="Policy parameter override, repeatable "
                             "(e.g. --policy-param v0=8.3).")
    parser.add_argument("--metrics-out", default=None,
                        help="Write a JSON run summary (collision occurrence, "
                             "impulses, motion stats) to this path.")
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
    from .backend.metrics import MetricsCollector
    from .backend.policy import (ExternalEgoController, parse_policy_params,
                                 resolve_policy)

    def _default_ego_binding():
        if args.ego_actor:
            return args.ego_actor
        if args.record_actor:
            return args.record_actor
        for b in annotated.scenario.actors:
            if b.type_name == "vehicle":
                return b.name
        return None

    ego_binding = _default_ego_binding() if args.ego_policy else None
    external_actors = {ego_binding} if ego_binding else set()
    policy_params = parse_policy_params(args.policy_param)

    if args.dry_run:
        ctx = ExecutionContext(annotated)
        tree = BehaviorTreeBuilder(annotated, ctx,
                                   external_actors=external_actors).build()
        if args.ego_policy:
            print(f"[osc2carla] ego policy: {args.ego_policy} driving "
                  f"{ego_binding!r}", file=sys.stderr)
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

    tree = BehaviorTreeBuilder(annotated, ctx,
                               external_actors=external_actors).build()
    behaviour_tree = py_trees.trees.BehaviourTree(root=tree)
    behaviour_tree.setup(timeout=15)

    # --- external ego controller entry point ---------------------------------
    controller = None
    if args.ego_policy:
        policy_cls = resolve_policy(args.ego_policy)
        controller = ExternalEgoController(world, carla_map, ctx, ego_binding,
                                           policy_cls(), params=policy_params)
        unknown = getattr(controller.policy, "_unknown", None)
        if unknown:
            print(f"[osc2carla] warning: ignored unknown policy params {unknown}",
                  file=sys.stderr)
        print(f"[osc2carla] ego policy {args.ego_policy!r} driving {ego_binding!r}; "
              f"behaviour-tree actuation for that binding is disabled",
              file=sys.stderr)

    metrics = None
    if args.metrics_out:
        metric_actor = ctx.actor(ego_binding) if ego_binding else \
            (recorder.target if recorder is not None else None)
        metric_binding = ego_binding or (args.record_actor or "ego")
        if metric_actor is None:
            metric_actor = ctx.actor(metric_binding)
        # Reuse the recorder's sensor when there is one, so a run cannot be
        # double-counted by two collision sensors on the same actor.
        metrics = MetricsCollector(world, metric_actor, metric_binding,
                                   attach_sensor=(recorder is None))
        if recorder is not None:
            metrics.use_external_collisions(recorder._collisions)

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
            # After the tree, so the policy has the last word on the ego. The
            # tree no longer actuates that binding, so there is no contention.
            leader_gap = None
            if controller is not None:
                obs = controller.tick(sim_t)
                if obs.leader is not None:
                    leader_gap = obs.leader.gap
            if recorder is not None:
                recorder.tick(sim_t)
            if metrics is not None:
                metrics.tick(sim_t, leader_gap=leader_gap)
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
        if metrics is not None:
            summary = metrics.write(
                args.metrics_out,
                scenario=os.path.splitext(os.path.basename(args.scenario))[0],
                scenario_path=os.path.abspath(args.scenario),
                ego_policy=(args.ego_policy or "scripted"),
                policy_params=policy_params,
                sim_duration=sim_t,
                sim_duration_requested=sim_cap,
                fixed_dt=args.fixed_dt,
            )
            metrics.close()
            print(f"[osc2carla] metrics -> {args.metrics_out} "
                  f"(collision_occurred={summary['collision_occurred']}, "
                  f"events={summary['n_collision_events']})", file=sys.stderr)
        if controller is not None:
            controller.teardown()
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
