# osc2carla: OpenSCENARIO 2.1 → CARLA, step by step

Internals of this **reimplementation** of the compiler in
*"Compiling OpenSCENARIO 2.1 for Scenario-Based Testing in CARLA"*
(Gamage & Gamage, arXiv:2604.16452). Start with [README.md](README.md) to
install and replicate; this file explains what each stage does. The paper
PDF is [`paper.pdf`](paper.pdf) in this directory.

The pipeline takes a declarative `.osc` scenario file and turns it into a live, closed-loop CARLA simulation that can be rendered as a MP4 file — with no external logic solver and no static code generation: the scenario *is* a py_trees  
behavior tree ticked in lockstep with the simulator.

```
 .osc file                                                       MP4 video
    │                                                                ▲
    ▼                                                                │
 ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────────────┐
 │ 1. FRONTEND  │   │ 2. MIDDLE    │   │ 3. BACKEND & RUNTIME             │
 │ ANTLR4 lexer │──▶│ ModelBuilder │──▶│ ScenarioInitializer (spawn)      │
 │ + parser     │   │ 2-pass       │   │ BehaviorTreeBuilder (py_trees)   │
 │ ASTTransform │   │ symbol table │   │ MethodRegistry (action→behavior) │
 │ → typed AST  │   │ → annotated  │   │ ExecutionContext (runtime eval)  │
 └──────────────┘   │   AST        │   │ Recorder (camera→frames→ffmpeg)  │
                    └──────────────┘   └──────────────────────────────────┘
                                                     ▲   │ apply_control()
                                          world.tick │   ▼ per tick, 20 Hz
                                              ┌──────────────┐
                                              │ CARLA 0.9.16 │
                                              └──────────────┘
```

Entry point: `python -m osc2carla <scenario.osc> [flags]` → `osc2carla/cli.py:main()`.

---

## Stage 1 — Frontend: text → typed AST

**Files:** `osc2carla/frontend/generated/`* (ANTLR4), `frontend/transformer.py`, `frontend/nodes.py`

TLDR: the compiler reuses the OpenSCENARIO 2.1 grammar from **PMSF py-osc2**
(Pierre R. Mai / PMSF IT Consulting) and an ANTLR-generated parser as the
**frontend**, then adds its own `ASTTransformer`, semantic `ModelBuilder`,
and CARLA backend on top. 

1. The ANTLR4 lexer/parser generated from the PMSF py-osc2 grammar
  (`grammar/openscenario2.g4`) turns the source into a parse tree.
   OSC2 is indentation-sensitive like Python, so a `DenterHelper` injects
   INDENT/DEDENT tokens.
2. `ASTTransformer` (`transformer.py`) walks the parse tree with the Visitor
  pattern and builds a small, strongly-typed dataclass AST (`nodes.py`):
   declarations (`ActorDecl`, `ScenarioDecl`, `ActionDecl`, `ModifierDecl`,
   `PhysicalTypeDecl`, `UnitDecl`, `EnumDecl`), scenario body
   (`Composition` for `serial/parallel/one_of`, `ActionCall`, `Wait`, `Emit`,
   `Modifier`), and expressions (`BinaryExpr`, `Call`, `MemberAccess`,
   `PhysicalLiteral`, event nodes `Rise`/`Fall`/`Elapsed`/`EventRef`).

No scoping or type checking happens here — the AST is purely syntactic
(paper §3.1). Parse errors are collected and raised as `ParseError`.

## Stage 2 — Middle-end: two-pass semantic analysis

**Files:** `osc2carla/middle/builder.py`, `middle/scope.py`

`analyse()` (`builder.py:30`) first parses the bundled stdlib stubs
(`stdlib/types.osc` — physical types, units, enums; `stdlib/domain.osc` —
actors, actions, modifiers), then the user file, and runs `ModelBuilder`
twice over all ASTs (paper §3.2):

- **Definition pass** (`builder.py:58`): registers every named entity into a
`GlobalScope` symbol table — physical types, a **unit table**
(`kph → factor 0.27778, type speed`), enums, actors with nested field
scopes, actions, modifiers, and per-scenario scopes for bindings and vars.
- **Resolution pass** (`builder.py:126`): resolves actor inheritance chains
(`vehicle inherits traffic_participant` merges the `name`/`model` fields),
validates that each scenario binding refers to a known actor type, and
evaluates `**keep` constraints**. Only the pattern
`keep(it.<field> == <literal>)` is folded into a `binding.attributes` dict
(e.g. `{"model": "vehicle.tesla.model3", "name": "hero"}`); any other keep
form is silently skipped (`builder.py:169`).

Output: an `AnnotatedScenario` — the first scenario in the file (only one is
supported) plus its scope, unit table, and enum table.

## Stage 3 — Backend: spawn, compile to behavior tree, tick

### 3a. Simulator connection and world setup (`cli.py`)

The CLI connects a `carla.Client`, reads the map name from the `map`
binding's `keep(it.map_file == "...")` (default `Town10HD_Opt`), loads the
world, and switches it to **synchronous mode** with
`fixed_delta_seconds = 0.05` (20 Hz). Determinism comes from this: the
server only advances when the client calls `world.tick()`.

`--backend pygame` substitutes `osc2carla.localsim.Client` here and nothing
else changes: the local simulator implements the same `Client` / `World` /
`Map` surface and is *only* ever synchronous, so the tick loop below is
literally the same loop. See 3g.

### 3b. ScenarioInitializer — declarative spawning (`backend/initializer.py`)

Before the tree runs, the initializer scans the `do` body for modifiers
tagged `at: start` on each actor's `assign_position()` call and spawns the
bindings in declaration order (paper §3.3, "lazy spawning"). Three placement
paradigms:


| Placement            | OSC2 syntax                                          | Mechanism (`initializer.py`)                                                                                                                   |
| -------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Default map          | no `position` modifier                               | picks a map spawn point; if another actor is anchored to this one, picks one with enough road ahead (`_spawn_point_with_road_ahead`, line 228) |
| Relative topological | `position(distance: 35m, ahead_of: hero, at: start)` | projects the reference's spawn transform onto the OpenDRIVE lane, walks `waypoint.next(d)` / `.previous(d)` (line 188)                         |
| Absolute Cartesian   | `position(x:…, y:…, z:…, h:…, at: start)`            | direct `carla.Transform` (line 169)                                                                                                            |


`facing: <actor>` flips yaw by 180°. The blueprint's `role_name` is set from
`keep(it.name == …)` — that is how the runtime finds actors again, and how
ROS 2 topic namespaces would be derived. On spawn collision it nudges +0.4 m
up and retries rather than scattering.

### 3c. BehaviorTreeBuilder + MethodRegistry — AST → py_trees (`backend/behavior_tree.py`, `method_registry.py`)

The `do` body maps 1:1 onto py_trees composites (`behavior_tree.py:62`):


| OSC2                     | py_trees                                                                          |
| ------------------------ | --------------------------------------------------------------------------------- |
| `serial:`                | `Sequence(memory=True)`                                                           |
| `parallel:`              | `Parallel(SUCCESS_ON_ALL)`                                                        |
| `one_of:`                | `Parallel(SUCCESS_ON_ONE)` — first child to succeed cancels the siblings          |
| `wait <cond>`            | `_ConditionLeaf` — re-evaluates the expression **every tick**, RUNNING until true |
| `emit NAME`              | `_EmitLeaf` — sets `ctx.blackboard["NAME"] = True`                                |
| `actor.action() with: …` | looked up in the `MethodRegistry`                                                 |


The idiomatic pattern `one_of: [drive(), wait <cond>]` is how OSC2 expresses
"drive **until** condition": `drive` returns RUNNING forever, so the
`one_of` only exits when the wait leaf fires.

The **MethodRegistry** (`method_registry.py`) is a decorator-based dispatch
table decoupling the OSC2 ontology from the CARLA API. Each entry in
`backend/atomic_behaviors.py` registers a factory:


| Action                                       | Behavior class                   | Control law                                                                                                                               |
| -------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `vehicle.drive` + `speed(v)` modifier        | `WaypointFollowerLite` (line 38) | PID on speed (kp 0.6, ki 0.05, kd 0.1) → throttle/brake; steering from yaw error to a 5 m-lookahead OpenDRIVE waypoint. Never terminates. |
| `vehicle.change_speed(target, rate_profile)` | `ChangeTargetSpeed` (line 112)   | `smooth` → P-controller (gain 0.4, capped 0.7); `asap` → bang-bang full throttle/brake. SUCCESS when                                      |
| `vehicle.change_lane(num_of_lanes, side)` + optional `speed(v)` | `LaneChangeLite`                 | resolves the target lane by stepping `num_of_lanes` neighbours to `side`, steers at a 6 m lookahead point **on that lane**, SUCCESS after 8 m on it *and* within 3° of its heading. `side:` is read in the actor's frame, and the lookahead is taken along the actor's direction of travel, so a change into an oncoming lane works. With `speed(v)`, the same PID as `drive()`; without it, a fixed throttle. |
| `vehicle.ram(target)`                        | `RamTarget` (line 302)           | pure pursuit: every tick recompute bearing to the target's **current** location, full throttle.                                           |
| `vehicle.set_lights(mode)`                   | `SetLights` (line 249)           | `carla.VehicleLightState` flags.                                                                                                          |
| `environment.assign_celestial_position`      | `AssignCelestial` (line 285)     | sun azimuth/elevation via weather API.                                                                                                    |
| `vehicle.assign_position`                    | `_Noop` (line 231)               | runtime no-op — placement already happened in the initializer.                                                                            |


Anything not in the registry compiles to a **silent `Success` leaf** named
`Unmapped[...]` (`behavior_tree.py:106`) — see the gaps table below.

### 3d. ExecutionContext — the runtime "VM" (`backend/context.py`)

The context is what makes the scenario *closed-loop*: declarative
expressions stay live and are recursively re-evaluated against the simulator
every tick (paper §3.3, "Runtime Context Management").

- **Quantities & units:** `35kph` → `Quantity(9.72, "speed")` via the unit
table; arithmetic like `v_npc_slow + 20kph` or `lag * 3` works on
`Quantity` (SI values), so variables can be derived from each other.
- **Actor handles:** identifiers resolve to `_ActorHandle`s with O(1) cached
lookup of the CARLA actor by `role_name`. Live queries:
  - `hero.speed` → magnitude of `get_velocity()` right now;
  - `npc.position.ahead_of(hero)` → signed OpenDRIVE `s`-coordinate
  difference when both are on the same road, else a forward-vector dot
  product (topological, not straight-line);
  - `a.object_distance(reference: b, direction: euclidean)` → 3-D distance.
- **Edge detection:** `rise(expr)` / `fall(expr)` keep per-node previous
values in `ctx._edge_prev` and fire exactly on the False→True / True→False
transition. `elapsed(t)` starts its timer at first evaluation.
- **Blackboard events:** `emit X` / `wait @X` are a shared dict — the
cross-actor synchronization mechanism. The CLI seeds
`blackboard["go_signal"] = True` before the first tick so scenarios can
gate on `wait @go_signal`.

### 3e. The tick loop (`cli.py:164–183`)

```python
while wall_time < timeout:
    world.tick()                 # advance CARLA physics by 0.05 s
    ctx.advance_tick(sim_t)      # update simulated clock
    behaviour_tree.tick()        # every RUNNING leaf: query state, apply control
    recorder.tick(sim_t)         # grab one camera frame
    # stop on: sim-duration cap (largest `wait elapsed` found in the file,
    # unless --sim-duration overrides) or tree SUCCESS/FAILURE
```

Each tick, every active behavior reads fresh state (velocities, transforms,
waypoints) and writes a fresh `VehicleControl` — sense → decide → act at
20 Hz. That per-tick loop is the closed loop.

### 3f. Recorder — frames → MP4 (`backend/recorder.py`)

`--record-video out.mp4 --record-actor hero` attaches to the chosen binding:

- an RGB chase camera (`sensor.camera.rgb`, 1280×720, FOV 95°, mounted
x=−9 m, z=+5 m, pitch −22°),
- a collision sensor whose impulse events are logged and overlaid.

Every tick one frame is popped from the sensor queue, annotated with
`t=…s hits=N` (red `COLLISION! impulse=…` after a hit) via OpenCV, and
written as `frame_%05d.png`. `finalize()` destroys the sensors and encodes
the frame directory with
`ffmpeg -framerate 20 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4`.

The CLI then restores async world settings and destroys spawned actors.

On the local backend, `localsim/render.py:BevRenderer` takes the recorder's
place behind an identical `tick(sim_time)` / `finalize()` / `collisions`
interface. There is no camera sensor to pop frames from: it draws the lane
graph and the actors' collision boxes top-down with pygame, overlays the
blackboard events and the RUNNING leaves of the tree, and writes the same
`frame_%05d.png` sequence for the same ffmpeg call.

### 3g. Backend selection (`backend/simapi.py`, `localsim/`)

The atomic behaviours, the initializer, the ego-policy hand-off and the
metrics collector are written against CARLA's actor API. They used to reach
it with a guarded `import carla`, which made "no simulator" the only
alternative to CARLA. `simapi` replaces that with a proxy object bound at
run time:

```python
from .simapi import sim as carla     # module-shaped proxy, never None
```

`simapi.bind("carla")` points it at the real API — done automatically at
import when CARLA is installed, so the CARLA path is byte-for-byte the same
behaviour as before. `simapi.bind("pygame")` points it at
`localsim/api.py`, which re-declares the slice of CARLA's surface this
compiler actually touches. The only edit the behaviour code needed was
turning `if carla is None:` into `if not carla:`, because a proxy is never
`None`.

Everything upstream of that line — grammar, AST, scope resolution, unit
table, `MethodRegistry`, the py_trees tree, `--ego-policy` — is untouched
and shared. What the local backend supplies instead of a UE4 server is a
synthesised lane graph (`localsim/towns.py`, `roadmap.py`), a kinematic
bicycle model carrying 2-D momentum (`localsim/actors.py`), and oriented-box
collision detection with an impulse response (`localsim/collision.py`).

Its limit is geometry, not mechanism: there are no OpenDRIVE towns without
the CARLA binary, so scenarios that hard-code Town10HD_Opt coordinates run
without staging their conflict. See the README section "Local simulator
backend".

---

## Construct provenance: what the paper actually sanctions

Every construct the benchmark scenarios use, and where the paper puts it.
"L1/L2/L3" are Listings 1–3 (the `hello_world` case study); "T2" is Table 2's
capability checklist.

| Construct | Paper | Where |
|---|---|---|
| `serial` / `parallel` / `one_of` | ✓ | T2 *Scenario Composition*; L2/L3 |
| `wait @EVENT` / `emit EVENT` | ✓ | T2 *Conditional triggers*; L2 line 80, L3 line 126 |
| `wait elapsed(t)` | ✓ | T2 *Temporal modifiers*; L2 line 73 |
| `rise(...)` / `fall(...)` | ✓ | T2 *Condition & Expression*; L2 lines 61, 89 |
| `drive()` + `speed(...)` | ✓ | T2 *Move/drive/walk*; L2 line 59 |
| `speed(v, rate_profile:)` | ✓ | L3 line 120 |
| `keep_lane()` | ✓ | T2 *Spatial Modifiers* — added for this pass, see below |
| `change_speed(target:, rate_profile:)` | ✓ | T2 *Speed control*; L2 line 84 |
| `change_lane(num_of_lanes:, side:)` | ✓ | T2 *Lateral modifier*; L2 line 67 |
| `change_lane(...) with: speed(v)` | ✓ | T2 *Lateral modifier* + *Speed control*, composed as any action + modifier |
| `assign_position()` + `position(x,y,z,h)` | ✓ | T2 *Assign position/orientation*; L1 line 49 |
| `position(distance:, ahead_of:/behind:)` | ✓ | T2 *Relative modifiers*, *Space gap*; L1 line 45 |
| `set_lights(mode:)` | ✓ | L1 line 37, L2 lines 72–74 |
| `assign_celestial_position(azimuth:, elevation:)` | ✓ | L1 line 36 |
| `object_distance(reference:, direction:)` | ✓ | L3 line 122 |
| `position.ahead_of(other)` | ✓ | L2 line 61, L3 line 106 |
| `actor.speed < literal` | ✓ | L2 line 93 |
| `keep(it.field == literal)` | ✓ | L1 lines 7–22 |
| `stationary_object` | ✓ | L1 line 20 |
| **`ram(target:)`** | **✗** | **nowhere — removed** |

`ram` was the single exception, and it is gone from every file under
`scenarios/benchmark/` and `scenarios/local/benchmark/`. Table 2 does sanction
*adding* actions (*Extensibility → Custom actions → MethodRegistry
decorators*), so registering one was not itself a departure — but the action
is not part of the paper's own vocabulary, and every adversarial outcome in
the benchmark rested on it.

**What replaced it.** Nothing, in the sense that no new action was needed. The
striking vehicle keeps executing `drive()` on its own lane at its own declared
speed. This baseline's `drive()` is a waypoint+PID controller with no
car-following, no yielding and no collision-avoidance term, so a vehicle that
does not change what it is doing is a vehicle that does not give way. The
conflict is produced by geometry and timing; the `emit`/`wait` handshakes still
mark the phases. Two consequences, both honest:

- Impacts are softer. `ram` applied full throttle regardless of the declared
  speed, so it accelerated a 19 kph car to ~50 kph. Now the declared speed is
  the actual speed, and `left_turn`'s peak impulse falls from ~19500 to ~5900.
- Timing matters more. The striking vehicle has to *be* somewhere at the right
  moment rather than homing in, so each scenario's approach distances were
  re-tuned against the local backend.

### `keep_lane`, added for this pass

Table 2 lists `keep_lane` under *Spatial Modifiers* and this baseline did not
implement it. It is needed once `ram` is gone: `drive()` calls
`get_waypoint()` fresh every tick, so a vehicle that leaves a junction
carrying lateral error is captured by whichever lane is nearest — and a right
turn leaves ~2 m of it, more than half a lane. `ram` hid this by homing on the
target's actual position.

`drive() with: keep_lane()` latches the first ordinary lane the leaf sees and
steers to that lane's centreline, re-latching on the way out of a junction
(the connector decides which lane you emerge in; you then hold it). It is
implemented in `WaypointFollowerLite._hold_lane` using only `lane_id`,
`is_junction` and `get_left_lane()`/`get_right_lane()`, so it works unchanged
on both backends.

### `change_lane`, made to work for this pass

`change_lane` was registered from the start, and no runnable scenario used it:
`hello_world.osc` and `nl2.osc` are the only files that did, and neither runs
live. The three highway scenarios are lateral by definition, so the leaf had
to be made to hold up. Four changes, all in `LaneChangeLite`:

- **The walk to the target lane follows `side`, not lane-id arithmetic.** The
  old code stepped `abs(target - current)` times in the direction of the sign
  difference. Across a centre line the ids change sign, so from lane −1 to
  lane +1 it walked *right* — the wrong way — and an overtake into oncoming
  traffic could not be expressed at all.
- **`side:` is read in the actor's frame.** `get_left_lane()` is relative to
  the lane's own direction of travel, which for a car passing in the oncoming
  lane is the opposite of its own. The manoeuvre asks for the driver's left,
  so the leaf takes the sense from the actor's heading; a return leg out of
  the oncoming lane therefore steers the right way.
- **Steering aims at a lookahead point on the target lane**, walked in the
  actor's direction of travel (`previous()` on a lane it is running against),
  rather than at the abeam projection. The path becomes a blend rather than a
  right-angle sidestep, and there is nothing left to jitter once the two
  coincide.
- **Completion needs alignment, not just distance.** SUCCESS now requires 8 m
  on the new lane *and* a heading within 3° of it (with a 30 m bound so the
  leaf always terminates). Handing over mid-yaw leaves the next action to
  inherit the drift, and the next action is often `change_speed`, which
  commands the wheels straight — enough to cross out of the lane just taken.

A `speed(...)` modifier on `change_lane` is honoured with the same PID
`drive()` uses. Without it the manoeuvre is a fixed-throttle one, so its
duration depends on the gradient of the road — on Town04's highway, a 5%
descent — and a choreographed conflict cannot tolerate that.

Non-driving lanes are now rejected as targets: on the Town04 highway the lane
left of the innermost one is the median shoulder, and CARLA hands it back from
`get_left_lane()` like any other. The local simulator reports a `lane_type` of
`Driving` for every synthesised lane, so the test is the same on both
backends.

---

## What is implemented vs. not (relative to the paper)

The paper's Table 2 claims broad coverage. This baseline implements the
core faithfully but several checklist rows are stubs or silent no-ops —
worth knowing before you write scenarios.

**Implemented and working:**

- serial / parallel / one_of composition; `wait` on `rise`/`fall`/`elapsed`/
boolean expressions/`@events`; `emit`
- `drive` + `speed` modifier (PID waypoint following), `change_speed`
(smooth/asap), `change_lane`, `ram`, `set_lights`,
`assign_celestial_position`
- `at: start` placement: absolute (x/y/z/h), relative topological
(`ahead_of`/`behind` + `distance`), `facing`
- physical types/units/enums, derived variables, actor inheritance,
`keep(it.field == literal)`, live spatial queries
(`speed`, `position.ahead_of`, `object_distance`)
- synchronous deterministic execution, video recording, `--dry-run`
(prints the ASCII behavior tree), `--emit-python` (standalone codegen via
`backend/codegen.py`)

**Not implemented (or silently degraded) in this baseline:**


| Feature                                                                                                                                    | Paper claim                      | Reality here                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `lane(1, at: start)` / `lane(side: right, side_of: …)`                                                                                     | "OpenDRIVE lane offsets ✓"       | parsed, then **ignored** — the initializer only reads the `position` modifier (`initializer.py:163`); actors get a default spawn point |
| `speed(0kph, at: start)`                                                                                                                   | initial kinematics               | ignored at spawn (vehicles start at rest anyway)                                                                                       |
| `acceleration` modifier                                                                                                                    | "PID interpolation ✓"            | declared in `domain.osc`, no registered behavior                                                                                       |
| `keep_lane`, `follow_path`, `follow_trajectory`, time/space gap, pedestrians (`person.walk`), traffic signals, weather beyond sun position | ✓ in Table 2                     | absent from `MethodRegistry`; any such action compiles to a silent `Success` leaf `Unmapped[...]`                                      |
| `keep()` other than `it.field == literal`                                                                                                  | constraint solving               | skipped without warning (`builder.py:169`)                                                                                             |
| `rate_profile:` on the `speed` *modifier* (e.g. paper Listing 3 line 120)                                                                  | smooth profile                   | ignored; only `change_speed(rate_profile: …)` honors it                                                                                |
| `object_distance(direction: longitudinal)`                                                                                                 | route-based s-t                  | only `euclidean` is 3-D exact; other directions use `|s_a − s_b|` on projected OpenDRIVE waypoints (not a true route s-t)              |
| multiple scenarios / `namespace` / `export` / `extend` / `every` / `coverage`                                                              | —                                | grammar parses them; the transformer/middle-end drops them (first scenario wins)                                                       |
| Traffic Manager, collision avoidance, right-of-way                                                                                         | "consistent collision avoidance" | none — controls are raw `apply_control()`; actors will happily crash (that is a feature for adversarial testing)                       |


**Baseline pitfall found while building the demo:** after
`change_speed(target: 0kph, rate_profile: asap)` brings the car to a full
stop, `ChangeTargetSpeed.update()` computes `err = 0 − |v| = 0` on the tick
where velocity is exactly zero, takes the `err >= 0` branch, and applies
`throttle = 1.0` before returning SUCCESS (`atomic_behaviors.py:129–133`).
CARLA persists the last `VehicleControl`, so with no behavior active
afterwards the "stopped" car silently accelerates away. The idiomatic fix
inside OSC2 is to hold the stop with an active controller:

```
hero.change_speed(target: 0kph, rate_profile: asap)
wait hero.speed < 0.3kph
one_of:                       # hold at standstill
    hero.drive() with:
        speed(0kph)
    wait elapsed(9s)
```

Consequence: the paper's `hello_world.osc` case study **compiles** here
(`--dry-run` reproduces the tree from Listings 2–3), but running it
faithfully would need Town06 and lane-based initial placement. This baseline
implements neither. A later, separate tree (`../osc2carla`) adds those and
other extensions; it is not required to replicate the demos in this directory.

---

## Demo 1 — `scenarios/closed_loop_demo.osc`: lead-vehicle brake check

Written for this walkthrough; uses only features that are genuinely
implemented, and exercises **every closed-loop mechanism**:

1. **Relative spawn** — green Audi (`npc`) is placed
  `position(distance: 35m, ahead_of: hero, at: start)` by walking OpenDRIVE
   waypoints from the hero's spawn transform.
2. **Closed-loop pursuit** — hero drives at `v_hero_fast = v_npc_slow + 15kph`
  (derived variable, unit arithmetic) under PID waypoint following, while
   `wait rise(hero.object_distance(reference: npc, direction: euclidean) < close_gap)`
   re-measures the live gap **every tick**, and fires on the 20 m crossing.
3. **Car following** — the `one_of` cancels the fast drive; hero re-enters
  `drive()` at the npc's speed (15 kph), tailgating at a stable gap.
4. **Cross-actor event handshake** — at t = 12 s the npc emits
  `NPC_BRAKING` and brake-checks (`change_speed(target: 0kph, rate_profile:  asap)`); the hero's `wait @NPC_BRAKING` sees the blackboard flag on the
   same tick and slams its own brakes. Both `wait <actor>.speed < 0.3kph`
   conditions confirm the halt from live velocity queries, then each car
   holds the stop with a 0-speed `drive()` inside a `one_of` (see the
   baseline pitfall above for why the hold is necessary).

Compiled tree (`--dry-run`):

```
parallel
--> Celestial[az=0.0,el=90.0]
[-] serial                                 # npc (lead)
    --> AssignPosition[npc]
    --> Wait[@go_signal]
    (o) one_of
        --> Drive[npc]                     # 15 kph PID, RUNNING forever
        --> Wait[elapsed]                  # 12 s
    --> Emit[NPC_BRAKING]
    --> ChangeSpeed[npc->0.0]              # asap = bang-bang brake
    --> Wait[npc.speed<PhysicalLiteral]
    --> Emit[NPC_STOPPED]
    (o) one_of
        --> Drive[npc]                     # hold at 0 kph
        --> Wait[elapsed]
[-] serial                                 # hero (follower)
    --> AssignPosition[hero]
    --> Wait[@go_signal]
    (o) one_of
        --> Drive[hero]                    # 35 kph catch-up
        --> Wait[rise(Call<close_gap)]     # live distance query
    (o) one_of
        --> Drive[hero]                    # 15 kph speed matching
        --> Wait[@NPC_BRAKING]             # blackboard event
    --> ChangeSpeed[hero->0.0]
    --> Wait[hero.speed<PhysicalLiteral]
    (o) one_of
        --> Drive[hero]                    # hold at 0 kph
        --> Wait[elapsed]
--> Wait[elapsed]                          # 22 s termination guard
```

## Demo 2 — `scenarios/scenario_collision.osc`: adversarial ram (ships with the baseline)

Shows the other flavor of closed-loop control plus collision
instrumentation: the npc spawns 25 m ahead of the ego **rotated 180°**
(`facing: ego`), then `npc.ram(target: ego)` re-aims at the ego's *current*
position every tick at full throttle while the ego holds still and then
coasts. The ego-mounted collision sensor stamps each impact's impulse onto
the video frames.

## Running the demos

No GPU on the login node, so CARLA runs in a Slurm job. From this directory:

```bash
# compile-only sanity check (works anywhere, no CARLA needed):
bash run_dry_run.sh

# execute + record on the bundled local simulator (no CARLA, no GPU):
./run_local_demo.sh --headless

# full run on a GPU node (boots CARLA off-screen, records both demos):
sbatch sbatch_closed_loop_demo.sh
```

The job script (`run_record_closed_loop_demo.sh`) boots
`CarlaUE4.sh -RenderOffScreen` on port 2000, waits for the RPC port, then for
each scenario runs a dry-run followed by

```bash
python -m osc2carla scenarios/<name>.osc \
    --sim-duration 22 --record-video <out>.mp4 \
    --record-actor hero --record-fps 20 --record-width 1280 --record-height 720
```

Outputs land in `../install/run_output/<name>/`: `<name>.mp4`, the raw
`_frames/` PNGs, `dry_run.log` (the tree), and `record.log` (spawn +
execution trace).

## Results (one recorded run: Slurm job 5776012, L40S)

Videos from that job are in `../install/run_output/` (raw annotated frames in
the sibling `*_frames/` directories). A new run should match the qualitative
timeline; spawn coordinates and collision counts can differ.

**`closed_loop_demo/closed_loop_demo.mp4`** — 440 frames, 22 s, **0
collisions**. Observable phase transitions (none time-scripted on the hero
side):


| t (s)   | What you see                                                    | Mechanism firing                                                    |
| ------- | --------------------------------------------------------------- | ------------------------------------------------------------------- |
| 0.0     | hero at spawn point (−64.6, 24.5); green npc visible 35 m ahead | relative topological placement                                      |
| 0–7     | hero accelerates to 30 kph, gap visibly shrinking               | `WaypointFollowerLite` PID                                          |
| ~7      | hero stops closing, settles behind the npc                      | `rise(object_distance < 20m)` edge fired → speed matching at 15 kph |
| 12.0    | npc brake-checks; hero pitches forward braking on the same tick | `emit NPC_BRAKING` → `wait @NPC_BRAKING`                            |
| 14.6–22 | both stopped, gap held constant                                 | `wait speed < 0.3kph` + 0-speed hold `drive()`                      |


Spawn log (`closed_loop_demo/record.log`):

```
[initializer] spawned 'hero' (vehicle.tesla.model3) at (-64.6, 24.5, 0.6) yaw=0 relative=False
[initializer] spawned 'npc'  (vehicle.audi.tt)      at (-45.3, -1.9, 0.5) yaw=-90 relative=True
```

Note `relative=True` and the different yaw: the npc's transform came from
walking 35 m of OpenDRIVE waypoints ahead of the hero, around a bend.

**`scenario_collision/scenario_collision.mp4`** — 22 s, **361 collision
events**: the npc rams the ego at ~t = 2.5 s and keeps pushing it through
the intersection at full throttle, with the impulse magnitude of every
contact overlaid in red on the frames.

## What to look at in the videos

1. `scenarios/closed_loop_demo.osc` has **no trajectory, no waypoint list, no
   timing table** on the hero: only intent (speeds, gaps, events).
2. `--dry-run` maps the OSC2 text 1:1 onto the behaviour tree. `one_of` means
   "do X until Y".
3. `backend/atomic_behaviors.py` (`WaypointFollowerLite.update`): the PID
   loop is the closed loop — read velocity, compute error, write control,
   every 50 ms.
4. `closed_loop_demo.mp4` — none of the hero's three phase transitions is
   time-scripted (the npc's 12 s brake-check is the only timed trigger).
5. `scenario_collision.mp4` — adversarial closed loop + collision overlay.

