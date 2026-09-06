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


def _connect_local(args):
    """Build the in-process simulator. Nothing is connected to; it is local."""
    from .localsim import Client
    return Client(args.host, args.port, turn_preference=args.junction_turn,
                  fixed_delta_seconds=args.fixed_dt)


def _maybe_load_world(client, map_name: str):
    world = client.get_world()
    cur_map = world.get_map().name
    if cur_map.endswith(map_name) or map_name.endswith(cur_map.split("/")[-1]):
        return world
    return client.load_world(map_name)


def _load_local_world(client, map_name: str, town_override: Optional[str]):
    """Load a local town, reporting when the scenario's map has no stand-in."""
    from .localsim.towns import BUILTIN_TOWNS, resolve_town_name
    requested = town_override or map_name
    town, aliased = resolve_town_name(requested)
    if aliased:
        print(f"[osc2carla] no local road network for {requested!r}; using "
              f"town {town!r} instead -- {BUILTIN_TOWNS[town].description} "
              f"A scenario with hard-coded CARLA coordinates will not stage "
              f"the same conflict here.", file=sys.stderr)
    return client.load_world(town)


def _display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _resolve_render_mode(args) -> str:
    """'window', 'headless' or 'off' from --render-mode auto."""
    if args.render_mode != "auto":
        return args.render_mode
    if _display_available():
        return "window"
    return "headless" if args.record_video else "off"


def _running_leaves(node, out=None):
    """Names of the RUNNING leaves of a behaviour tree, for the overlay."""
    import py_trees as _pt
    out = [] if out is None else out
    children = getattr(node, "children", None) or []
    if node.status == _pt.common.Status.RUNNING and not children:
        out.append(node.name)
    for child in children:
        _running_leaves(child, out)
    return out


def _vision_strip_aspect(rig) -> float:
    """Width-to-height ratio of the rig's stitched camera strip.

    Read off the DECLARED cameras rather than off a captured panel, because the
    recorder has to fix the band's height before the first frame -- every frame
    in an MP4 is the same size -- and the first frame may be one where the rig
    delivered nothing. 0 means "no cameras", and the caller draws no band.
    """
    from .backend.sensors import CameraSpec
    cameras = [s for s in rig.specs if isinstance(s, CameraSpec)]
    if not cameras:
        return 0.0
    strip_w = sum(int(c.width) for c in cameras)
    strip_h = max(int(c.height) for c in cameras)
    if strip_w <= 0 or strip_h <= 0:
        return 0.0
    return strip_w / float(strip_h)


def _vision_hook(controller, mode: str):
    """The recorder's per-tick vision payload, or None for no panel.

    Returns None -- meaning "record exactly as before" -- unless there is an ego
    policy with an attached rig, so a scripted or analytic arm is unchanged: a
    band of black under every frame would be noise, not information.

    The lines beside the frames are the parts of the hand-off a viewer cannot
    read off the images: the speed the policy was told, the command it returned,
    and the route point it was steering at. A frame that looks right beside a
    command that is not tells you the rig is fine and the policy is not, which
    is the distinction a video is being watched for.
    """
    if mode == "off" or controller is None:
        return None
    rig = getattr(controller, "rig", None)
    if rig is None or not rig.active:
        if mode == "on":
            print("[osc2carla] --vision-panel on, but the ego policy attached "
                  "no sensor rig; recording without a panel", file=sys.stderr)
        return None

    def payload():
        obs = controller.last_observation
        lines = []
        if obs is not None:
            # What the POLICY was told, when the policy plumbing kept a copy of
            # it. A bridged policy is handed a resampled route, so showing the
            # controller's own sampling here would put a number on the panel
            # that nothing in the run ever saw.
            sent = getattr(controller.policy, "last_observation_payload", None)
            route = (sent or {}).get("route") if isinstance(sent, dict) \
                else obs.route_ego()
            far = route[-1] if route else None
            lines.append(
                "%s  t=%5.2fs  v=%4.1f m/s  decisions=%d"
                % (getattr(controller.policy, "name", "policy"), obs.t,
                   obs.speed, controller.decisions))
            command = controller.command
            if command is not None:
                lines.append("throttle=%.2f  brake=%.2f  steer=%+.2f"
                             % (command.throttle, command.brake, command.steer))
            if far is None:
                lines.append("route: none -- the policy is driving unconditioned")
            else:
                lines.append("route %d pts, far=(%+.1f, %+.1f) m"
                             % (len(route), far[0], far[1]))
        missing = [name for name in rig.names if name not in
                   ((obs.sensors if obs is not None else {}) or {})]
        if missing:
            lines.append("no data this tick from: " + ", ".join(missing))
        return {"panel": controller.vision_panel(), "lines": lines}

    return payload


def _binding_labels(annotated, ctx) -> dict:
    """actor id -> scenario binding name, so the overlay can name the cars."""
    labels = {}
    for binding in annotated.scenario.actors:
        actor = ctx.actor(binding.name)
        if actor is not None:
            labels[actor.id] = binding.name
    return labels


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OpenSCENARIO 2 compiler and runtime: CARLA, or the bundled local simulator")
    parser.add_argument("scenario", nargs="?",
                        help="Path to .osc file")
    parser.add_argument("--stdlib", default=None, help="Override stdlib directory")
    parser.add_argument("--backend", choices=("carla", "pygame"), default="carla",
                        help="Where the compiled behaviour tree executes: a "
                             "CARLA server, or the bundled standalone "
                             "simulator with a pygame bird's-eye view "
                             "(default: carla).")
    parser.add_argument("--town", default=None,
                        help="Local backend only: road network to load, "
                             "overriding the scenario's map_file. See "
                             "--list-towns.")
    parser.add_argument("--list-towns", action="store_true",
                        help="Print the local backend's road networks and exit.")
    parser.add_argument("--junction-turn", default="straight",
                        choices=("straight", "left", "right"),
                        help="Which manoeuvre the ego's route takes where a "
                             "lane branches, since a route is walked with "
                             "next()[0] and CARLA's own ordering of the "
                             "branches means nothing (default: straight). On "
                             "the local backend it orders the map's own "
                             "successors; on CARLA it chooses the exit when "
                             "the ego's route is planned. A property of the "
                             "scenario -- the harness sets it per scenario "
                             "from the benchmark's declared intent.")
    parser.add_argument("--ego-light", default=None,
                        choices=("green", "yellow", "red"),
                        help="The signal phase the ego meets at its junction, "
                             "set at spawn and frozen for the episode. This "
                             "dialect has no action that sets a signal phase, "
                             "so left unset the phase is whatever the "
                             "simulator's own cycle happened to be showing -- "
                             "and on the junction families it decides the run. "
                             "A property of the scenario: the harness sets it "
                             "per scenario from the benchmark's declared "
                             "intent.")
    parser.add_argument("--render-mode", default="auto",
                        choices=("auto", "window", "headless", "off"),
                        help="Local backend only: 'window' opens a pygame "
                             "view, 'headless' renders off-screen (needed "
                             "for --record-video without a display), 'off' "
                             "simulates without drawing. 'auto' picks a "
                             "window when a display exists (default).")
    parser.add_argument("--render-scale", type=float, default=None,
                        help="Local backend only: pixels per metre. Default "
                             "fits the whole town in the window.")
    parser.add_argument("--no-follow", action="store_true",
                        help="Local backend only: keep the camera on the whole "
                             "town instead of following --record-actor.")
    parser.add_argument("--realtime", action="store_true",
                        help="Local backend only: play back at wall-clock "
                             "speed instead of as fast as possible.")
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
    parser.add_argument("--policy-hz", type=float, default=0.0,
                        help="Decision rate for --ego-policy, holding the last "
                             "command in between. 0 (default) decides on every "
                             "simulation tick, which is what the analytic "
                             "policies do. A network usually wants less: a VLA "
                             "asked for 20 decisions per simulated second "
                             "spends the run inside forward().")
    parser.add_argument("--vision-panel", default="auto",
                        choices=("auto", "on", "off"),
                        help="Draw what the ego policy's sensor rig saw under "
                             "the chase cam in --record-video. 'auto' (default) "
                             "draws it whenever a rig is attached.")
    parser.add_argument("--trace-out", default=None,
                        help="Directory to write a per-tick canonical trace "
                             "(states.jsonl + scene.json) for the harness's "
                             "metrics package. Needs metrics/recording to be "
                             "findable above this repository.")
    parser.add_argument("--trace-rate-hz", type=float, default=10.0,
                        help="Sampling rate of --trace-out (default 10 Hz; the "
                             "physics rate is --fixed-dt).")
    parser.add_argument("--metrics-out", default=None,
                        help="Write a JSON run summary (collision occurrence, "
                             "impulses, motion stats) to this path.")
    args = parser.parse_args(argv)

    if args.list_towns:
        from .localsim.towns import BUILTIN_TOWNS, TOWN_ALIASES
        print("local backend road networks:")
        for name, town in sorted(BUILTIN_TOWNS.items()):
            print(f"  {name:<10} {town.description}")
        print("\nCARLA map names accepted as aliases:")
        for carla_name, local in sorted(TOWN_ALIASES.items()):
            print(f"  {carla_name:<14} -> {local}")
        print("\nCoordinate reference sheets (PNG + Markdown) for authoring "
              "scenarios:\n  python -m osc2carla.localsim.mapview")
        return 0

    if not args.scenario:
        parser.error("a scenario path is required")

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
    from .backend.initializer import clear_leftover_actors
    from .backend import simapi
    from .backend.metrics import MetricsCollector
    from .backend.trace import SceneTracer, make_recorder
    from .backend.policy import (ExternalEgoController, parse_policy_params,
                                 resolve_policy)

    if not args.dry_run:
        try:
            simapi.bind(args.backend)
        except ImportError as exc:
            print(f"[osc2carla] backend {args.backend!r} unavailable: {exc}\n"
                  f"            {simapi.BACKENDS[args.backend]}",
                  file=sys.stderr)
            return 2

    def _default_ego_binding():
        if args.ego_actor:
            return args.ego_actor
        if args.record_actor:
            return args.record_actor
        for b in annotated.scenario.actors:
            if b.type_name == "vehicle":
                return b.name
        return None

    # Set before any route is planned. A bridged policy's observation builder is
    # reached through a class name and can be given no argument of its own, so
    # the preference the scenario declared travels this way rather than through
    # every constructor between here and there.
    from .backend import route as route_plan
    route_plan.set_default_preference(args.junction_turn)

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

    local = args.backend == "pygame"
    if local:
        args.no_sync = False        # the local simulator only steps on tick()
    client = _connect_local(args) if local \
        else _connect_carla(args.host, args.port, args.carla_timeout)
    map_attr = next((b for b in annotated.scenario.actors if b.type_name == "map"), None)
    map_name = "Town10HD_Opt"
    if map_attr is not None:
        attrs = getattr(map_attr, "attributes", {}) or {}
        map_name = attrs.get("map_file", map_name)
    print(f"[osc2carla] backend: {simapi.describe()}", file=sys.stderr)
    print(f"[osc2carla] loading map: {map_name}", file=sys.stderr)
    world = _load_local_world(client, map_name, args.town) if local \
        else _maybe_load_world(client, map_name)
    carla_map = world.get_map()

    original_settings = world.get_settings()
    if not args.no_sync:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_dt
        world.apply_settings(settings)

    if not local:
        # Before anything is spawned, and on the live path -- not only in the
        # script codegen emits, which is where this check first landed and where
        # the harness never runs. A world whose town was already loaded keeps
        # whatever a crashed run left standing, and the next cast goes on top of
        # it. Measured on red_light: two egos at the same coordinates, the
        # collision sensor naming `vehicle.tesla.model3` with role `ego` as the
        # partner on every one of 181 ticks, 0.4 m travelled, and a full set of
        # metrics describing a scenario that never ran.
        cleared = clear_leftover_actors(world)
        if cleared:
            print(f"[osc2carla] cleared {cleared} actor(s) left over from a "
                  f"previous run on this world", file=sys.stderr)

    ctx = ExecutionContext(annotated, world=world, carla_map=carla_map)
    initializer = ScenarioInitializer(world, carla_map, annotated, ctx)
    initializer.initialise()
    if not args.no_sync:
        world.tick()

    # The ego's junction phase, before the first policy decision is taken.
    signal_note = {"requested": None}
    if not local:
        from .backend import signals
        signal_actor = ctx.actor(ego_binding) if ego_binding else None
        if signal_actor is None:
            signal_actor = ctx.actor(_default_ego_binding() or "")
        signal_note = signals.apply(world, carla_map, signal_actor,
                                    args.ego_light)
        if signal_note.get("requested"):
            print("[osc2carla] ego signal: %s" % signal_note.get("note"),
                  file=sys.stderr)
        if not args.no_sync:
            world.tick()

    rec_binding = args.record_actor
    if rec_binding is None:
        for b in annotated.scenario.actors:
            if b.type_name == "vehicle":
                rec_binding = b.name
                break
    rec_actor = ctx.actor(rec_binding) if rec_binding else None

    # --- external ego controller entry point ---------------------------------
    # Built before the recorder so the recording can draw the rig this
    # controller attaches; it needs the world, the map and the context, none of
    # which depend on the behaviour tree.
    controller = None
    if args.ego_policy:
        policy_cls = resolve_policy(args.ego_policy)
        controller = ExternalEgoController(world, carla_map, ctx, ego_binding,
                                           policy_cls(), params=policy_params,
                                           turn_preference=args.junction_turn,
                                           decision_hz=args.policy_hz)
        unknown = getattr(controller.policy, "_unknown", None)
        if unknown:
            print(f"[osc2carla] warning: ignored unknown policy params {unknown}",
                  file=sys.stderr)
        print(f"[osc2carla] ego policy {args.ego_policy!r} driving {ego_binding!r}"
              + (f" at {args.policy_hz:g} Hz" if args.policy_hz > 0 else "")
              + "; behaviour-tree actuation for that binding is disabled",
              file=sys.stderr)

    recorder = None
    viewer = None
    if local:
        render_mode = _resolve_render_mode(args)
        if render_mode != "off":
            from .localsim.render import BevRenderer, RendererUnavailable
            try:
                viewer = BevRenderer(
                    world, rec_actor, output_video=args.record_video,
                    width=args.record_width, height=args.record_height,
                    fps=args.record_fps, display=(render_mode == "window"),
                    scale=args.render_scale, follow=not args.no_follow,
                    caption=f"osc2carla — {annotated.scenario.name}")
                recorder = viewer
                print(f"[osc2carla] bird's-eye view: {render_mode}"
                      + (f", recording -> {args.record_video}"
                         if args.record_video else ""), file=sys.stderr)
            except RendererUnavailable as exc:
                print(f"[osc2carla] {exc}", file=sys.stderr)
                return 2
        elif args.record_video:
            print("[osc2carla] --record-video needs rendering; "
                  "--render-mode off ignores it", file=sys.stderr)
    elif args.record_video and rec_actor is not None:
        panel = _vision_hook(controller, args.vision_panel)
        # A rig with range sensors but no cameras has nothing to draw, and an
        # empty band under every frame is noise rather than a check.
        aspect = _vision_strip_aspect(controller.rig) if panel else 0.0
        if not aspect:
            panel = None
        recorder = Recorder(world, rec_actor, args.record_video,
                            width=args.record_width,
                            height=args.record_height,
                            fps=args.record_fps,
                            vision=panel, vision_aspect=aspect)
        print(f"[osc2carla] recording {rec_binding!r} -> {args.record_video}"
              + (" (with the policy's own camera rig)" if panel else ""),
              file=sys.stderr)

    tree = BehaviorTreeBuilder(annotated, ctx,
                               external_actors=external_actors).build()
    behaviour_tree = py_trees.trees.BehaviourTree(root=tree)
    behaviour_tree.setup(timeout=15)

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

    tracer = None
    if args.trace_out:
        trace_rec, note = make_recorder(
            args.trace_out, rate_hz=args.trace_rate_hz,
            context={"method": "osc2runner", "backend": args.backend,
                     "scenario": annotated.scenario.name, "town": map_name,
                     "ego_binding": ego_binding, "fixed_dt": args.fixed_dt,
                     "ego_policy": args.ego_policy,
                     "ego_signal": signal_note})
        if note:
            print("[osc2carla] trace: %s" % note, file=sys.stderr)
        tracer = SceneTracer(trace_rec, ctx, carla_map, ego_binding=ego_binding)
        tracer.declare_scene()

    ctx.blackboard["go_signal"] = True

    sim_cap = args.sim_duration
    if sim_cap is None:
        sim_cap = _scenario_sim_duration(annotated.scenario.do.body, ctx) \
            if annotated.scenario.do is not None else 0.0
    if sim_cap and sim_cap > 0:
        print(f"[osc2carla] scenario sim duration: {sim_cap:.1f}s", file=sys.stderr)

    hud = None
    if viewer is not None:
        from .localsim.render import HudState
        hud = HudState(scenario=annotated.scenario.name,
                       backend=simapi.sim.name or args.backend,
                       town=carla_map.name,
                       ego_policy=args.ego_policy,
                       ego_binding=ego_binding,
                       bindings=_binding_labels(annotated, ctx),
                       note=f"routes take the {args.junction_turn} exit at "
                            "junctions")

    start = time.time()
    sim_t = 0.0
    try:
        while time.time() - start < args.timeout:
            if viewer is not None and viewer.paused:
                # Hold the world still but keep the window responsive.
                hud.paused = True
                if not viewer.tick(sim_t, hud):
                    print("[osc2carla] window closed", file=sys.stderr)
                    break
                viewer.throttle()
                continue
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
            if viewer is not None:
                hud.paused = False
                hud.events = dict(ctx.blackboard)
                hud.active_leaves = _running_leaves(behaviour_tree.root)
                hud.tree_status = str(behaviour_tree.root.status)
                if not viewer.tick(sim_t, hud):
                    print("[osc2carla] window closed", file=sys.stderr)
                    break
                if args.realtime:
                    viewer.throttle(int(round(1.0 / max(args.fixed_dt, 1e-3))))
            elif recorder is not None:
                recorder.tick(sim_t)
            if metrics is not None:
                metrics.tick(sim_t, leader_gap=leader_gap)
            if tracer is not None:
                tracer.tick(sim_t)
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
                ego_policy_detail=(controller.describe()
                                   if controller is not None else None),
                sim_duration=sim_t,
                sim_duration_requested=sim_cap,
                fixed_dt=args.fixed_dt,
            )
            metrics.close()
            print(f"[osc2carla] metrics -> {args.metrics_out} "
                  f"(collision_occurred={summary['collision_occurred']}, "
                  f"events={summary['n_collision_events']})", file=sys.stderr)
        if tracer is not None:
            # After metrics.write, so the collision list the trace records is
            # the same one osc2carla_metrics.json reports, and before the
            # actors are destroyed.
            tracer.note_collisions(metrics.collisions if metrics is not None
                                   else [], ego_binding)
            out = tracer.close()
            if out:
                print("[osc2carla] trace -> %s" % out, file=sys.stderr)
        if controller is not None:
            controller.teardown()
        if recorder is not None:
            out = recorder.finalize()
            if out:
                print(f"[osc2carla] wrote {out} ({len(recorder.collisions)} collision events)",
                      file=sys.stderr)
        if not local and not args.no_sync:
            world.apply_settings(original_settings)
        for a in list(getattr(initializer, "_spawned", [])):
            try:
                a.destroy()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
