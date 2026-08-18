# osc2carla_implement

Independent **reimplementation** of the compiler in

> S. Gamage and D. Gamage, *Compiling OpenSCENARIO 2.1 for Scenario-Based Testing in CARLA*, arXiv:2604.16452.

The paper did not release code. This tree implements the three-stage pipeline from the paper (ANTLR frontend → two-pass semantic analysis → py_trees / CARLA backend) and records the two demos that actually run on CARLA 0.9.16.

Paper PDF (local copy): [`paper.pdf`](paper.pdf) (same file as `compiling OpenScenario2 for CARLA.pdf`).

This is **not** the authors’ code, and it is **not** the later `../osc2carla` tree (TF++, DQL ego, traffic-light control, …). Use this directory to understand and replicate the paper baseline.

Internals of each compiler stage: [PIPELINE_WALKTHROUGH.md](PIPELINE_WALKTHROUGH.md). Third-party licenses: [THIRD_PARTY.md](THIRD_PARTY.md).

## Layout

```
osc2carla_implement/
├── osc2carla/            # compiler package (frontend / middle / backend)
│   └── localsim/         # standalone simulator + pygame BEV renderer
├── grammar/              # OpenSCENARIO 2.1 ANTLR grammar (from py-osc2)
├── stdlib/               # types.osc + domain.osc stubs used at analyse time
├── scenarios/            # .osc inputs (see table below)
│   └── local/            # scenarios written against the local backend's town
├── third_party/py-osc2/  # vendored grammar source + MPL-2.0 license
├── requirements.txt
├── env.sh                # sets OSC2CARLA_ROOT and PYTHONPATH
├── run_dry_run.sh        # no CARLA: parse + print behaviour trees
├── run_local_demo.sh     # no CARLA: run + record on the local simulator
├── run_record_closed_loop_demo.sh   # boots CARLA, records the two paper demos
├── run_record_simple_example.sh     # shorter single-car smoke recording
├── run_record_scenario_collision.sh # collision demo only
└── sbatch_closed_loop_demo.sh       # Slurm wrapper for this cluster
```

## What to replicate

| Scenario | Role | CARLA needed? |
|---|---|---|
| `scenarios/closed_loop_demo.osc` | Lead-vehicle brake check (relative spawn, live gap, blackboard events) | yes, Town10HD_Opt |
| `scenarios/scenario_collision.osc` | Head-on ram + collision overlay | yes, Town10HD_Opt |
| `scenarios/simple_example.osc` | One-car cruise / slow / stop smoke test | yes, Town10HD_Opt |
| `scenarios/hello_world.osc` | Paper case study (Listings 2–3) | **dry-run only** here (needs Town06 + `lane()` placement, neither of which this baseline implements) |
| `scenarios/nl2/nl2.osc` | Extra NL-template experiment, **not from the paper** | optional |
| `scenarios/local/local_crossing.osc` | Junction failure-to-yield, written against the local backend's `grid` town | **no** — `--backend pygame` |
| `scenarios/local/benchmark/*.osc` | The four benchmark scenarios ported to the local `grid` town | **no** — `--backend pygame` |

`--dry-run` on `hello_world.osc` is the compile-only check against the paper’s behaviour tree. Running it live would need Town06 and lane-based spawn, which this baseline does not implement (see “Coverage vs. the paper” below).

## Requirements

- Python **3.8** (CARLA 0.9.16’s official wheel is `cp38`)
- `pip install -r requirements.txt`
- For a live run: **CARLA 0.9.16** server + its `PythonAPI/carla` on `PYTHONPATH`
- For `--record-video`: OpenCV (`cv2`) and `ffmpeg`
- GPU for the UE4 server (off-screen rendering is fine)
- For `--backend pygame`: `pygame` only — no CARLA, no GPU, no server, and no
  Python 3.8 constraint

On this cluster the parent tree already has the server and venv:

```bash
module load StdEnv/2020 gcc/9.3.0 python/3.8.10 opencv/4.5.5
source ../install/env.sh    # AV_SERVER, AV_VENV, AV_RUN, activates venv
source env.sh               # this package on PYTHONPATH
```

`opencv` is provided by the module, not by the venv. Elsewhere: `pip install opencv-python`.

## 1. Compile-only check (no CARLA, no GPU)

From this directory:

```bash
source env.sh
# if you have the parent venv:
#   source ../install/env.sh && source env.sh
python -m osc2carla scenarios/closed_loop_demo.osc --dry-run
```

Or all paper-relevant scenarios at once:

```bash
bash run_dry_run.sh
```

You should see `parsed OK` and an ASCII behaviour tree. `closed_loop_demo` starts with:

```
parallel
--> Celestial[az=0.0,el=90.…]
[-] serial          # npc
    --> AssignPosition[npc]
    …
[-] serial          # hero
    --> AssignPosition[hero]
    …
--> Wait[elapsed]
```

`hello_world.osc` should also parse; that is the paper case-study tree.

## 1b. Local simulator backend (no CARLA, no GPU)

`--backend pygame` runs the *same* compiled behaviour tree against a
standalone simulator bundled in `osc2carla/localsim/`, drawn top-down with
pygame. Nothing in that package imports `carla`; nothing about it needs a
server, a GPU, or Python 3.8. It is the laptop loop for writing and debugging
scenarios.

```bash
pip install pygame
source env.sh
python -m osc2carla scenarios/local/local_crossing.osc --backend pygame
```

Or all four local-runnable scenarios, recorded:

```bash
./run_local_demo.sh --headless
```

### What is shared, and what is not

The frontend, the semantic analysis, the `MethodRegistry`, the py_trees tree
and the `--ego-policy` hand-off are **the same code on both backends**. The
atomic behaviours reach whichever simulator is bound through
`osc2carla/backend/simapi.py`, a proxy object that stands in for the `carla`
module:

```python
from .simapi import sim as carla       # was: try: import carla
...
control = carla.VehicleControl(throttle=1.0)
```

`simapi.bind("carla")` points it at the real CARLA API (this happens
automatically at import when CARLA is installed, so the CARLA path is
unchanged); `simapi.bind("pygame")` points it at
`osc2carla/localsim/api.py`, which re-declares the slice of CARLA's surface
the backend actually uses — `Location`/`Rotation`/`Transform`,
`VehicleControl`, `VehicleLightState`, `World.tick()/spawn_actor()`,
`Map.get_waypoint().next()`, `Actor.apply_control()`,
`sensor.other.collision`. Because the proxy is never `None`, the old
"is a simulator available?" guards read `if not carla:` instead of
`if carla is None:`; that is the only edit the behaviour code needed.

What the local simulator provides:

| Piece | Implementation |
|---|---|
| Road network | `localsim/towns.py` — synthesised lane graphs, since OpenDRIVE towns ship with the CARLA binary |
| Waypoint API | `localsim/roadmap.py` — lane polylines, successor/predecessor graph, nearest-lane index |
| Vehicle physics | `localsim/actors.py` — kinematic bicycle model carrying 2-D momentum, so an impact can push and spin a car |
| Collisions | `localsim/collision.py` — oriented-box SAT, impulse response, one event per contacting pair per tick (CARLA reports per substep, which is why `metrics.py` treats the count as contact *duration*) |
| Rendering | `localsim/render.py` — bird's-eye pygame view, window or off-screen, PNG frames → MP4 via ffmpeg |

### Fidelity: what this is not

**It previews scenario logic; it does not reproduce CARLA measurements.**
There is no photorealistic sensor model, no tyre model, and — the one that
bites — no CARLA town geometry. `localsim/towns.py` synthesises grid networks:

```
$ python -m osc2carla --list-towns
grid        3x3 junctions, 80 m spacing. One four-way junction, at (80, 80).
loop        Single rectangular circuit. No four-way junction.
wide_grid   4x4 junctions over x,y in [-80, 160]. Four four-way junctions --
            (0,0), (0,80), (80,0), (80,80) -- with 66 m approaches on every arm.
```

### Map reference sheets

Authoring a scenario against a CARLA town means opening the map in the
simulator and reading coordinates off it. The local towns are synthesised, so
there is nothing to open — `mapview` is the substitute:

```bash
python -m osc2carla.localsim.mapview --spawn-points
```

It writes `maps/<town>.png` and `maps/<town>.md` (both checked in). The PNG
draws the network to scale over a 20 m grid and labels every junction centre
with the arms it actually has; the Markdown lists the same figures as tables
you can paste from — one row per carriageway giving its fixed coordinate,
direction, lane id, heading in radians, and the range of straight road it
covers:

| runs along | fixed coord | direction | lane | `h` (rad) | travel range |
|---|---|---|---|---|---|
| x | y = 81.75 | east (+x) | -1 | +0.0000 | x in [7, 153] |
| y | x = 78.25 | south (+y) | -1 | +1.5708 | y in [7, 153] |

Two things the sheets make obvious, both of which constrain where a scenario
can be staged:

- **Not every junction is four-way.** The grid's edges and corners have two
  or three arms; the `arms` column says which. A conflict that needs an
  opposing approach — an unprotected left, a red-light T-bone — has to sit at
  an `NESW` junction.
- **`h` grows clockwise**, because +y is south. Southbound is `h: 1.5708rad`,
  northbound `h: -1.5708rad`, westbound `h: 3.1416rad`. Getting the sign
  wrong spawns a car facing into oncoming traffic, and `drive()` will
  cheerfully follow the lane it snaps to.

CARLA map names are aliased onto these (`Town10HD_Opt` → `grid`) so an
unmodified `.osc` file loads and reports the substitution. The consequence:

- Scenarios that place actors by **map spawn point or relative topology**
  (`closed_loop_demo`, `scenario_collision`, `simple_example`) transfer as-is
  and behave qualitatively as they do on CARLA.
- Scenarios that **hard-code Town10HD_Opt coordinates** — everything under
  `scenarios/benchmark/` — will load and run, but their actors land wherever
  those coordinates fall in the grid, and the staged conflict does not
  develop. `run_local_demo.sh` deliberately skips them.
  `scenarios/local/local_crossing.osc` is the local-town counterpart: same
  mechanisms (relative placement, live distance monitor, cross-actor
  `emit`/`wait`, `drive()`), geometry that exists here.

One more difference worth knowing: this baseline's `drive()` follows
`waypoint.next()[0]`, so which way a vehicle goes through a junction is a
property of the map, not of anything the scenario can request. CARLA's order
comes from the OpenDRIVE file; here it is explicit, and `--junction-turn
{straight,left,right}` selects it.

### The benchmark scenarios, ported

`scenarios/local/benchmark/` stages the same four natural-language scenarios
as `scenarios/benchmark/`, at the centre junction **(80, 80)** of the `grid`
town instead of at Town10HD_Opt junctions 189 / 841 / 134:

```bash
./experiments/run_experiments_local.sh            # both policy arms, all four
python experiments/make_report.py experiments/results_local     -c experiments/benchmark_local.json
```

| Scenario | `--junction-turn` | Intent | Observed (scripted arm) |
|---|---|---|---|
| `red_light` | `straight` | violator T-bones the ego inside the junction | collision at t = 6.95 s, partner `violator` |
| `right_turn` | `right` | ego rear-ended after turning right | collision at t = 9.35 s, partner `rear_ender` |
| `left_turn` | `left` | oncoming car strikes the ego mid-turn | collision at t = 8.65 s, partner `oncoming` |
| `stop_sign` | `straight` | precedence negotiated, nothing is hit | **no collision** — the pass condition |

The scripted arm produces the intended outcome, with the intended partner, in
all four. Under `--ego-policy idm` only `left_turn` does: IDM is rear-ended by
its own follower in `red_light`, meets the wrong vehicle in `right_turn`, and
is struck by the crosser in `stop_sign` — it has no term for a crossing
conflict, so it does not yield.

These are **ports, not the same runs**. The geometry is synthesised, so the
absolute numbers are not comparable with `benchmark.json`; the intended
*outcome* per scenario is, and that is what `expect_collision` encodes. Each
file's header states its own deviations.

**How one scenario mixes manoeuvres.** `drive()` follows `next()[0]`, and
`--junction-turn` is a whole-map preference, so at first sight every actor in
a run must take the same exit. The town's lane-pairing rule supplies the
difference: a **left** turn is connected only from the innermost lane, and a
**right** turn only between outermost lanes — as on a real road. A vehicle on
the other lane has no such connector and falls through to going straight. So
in `left_turn`, the ego and the two cars queued with it sit on the inner
eastbound lane and turn; `oncoming_b` sits on the outer westbound lane and
does not.

**No `ram`.** These scenarios use only constructs the paper documents — see
[Construct provenance](PIPELINE_WALKTHROUGH.md) for the line-by-line mapping.
The striking vehicle keeps executing `drive()` on its own lane at its own
declared speed; `drive()` has no car-following, yielding or collision-avoidance
term, so a vehicle that does not change what it is doing is one that does not
give way. Two consequences: impacts are softer than the `ram` versions (`ram`
applied full throttle regardless of the declared speed), and approach
distances had to be re-tuned, because the striking vehicle now has to *be*
somewhere at the right moment rather than home in on the ego.

`right_turn` needed one thing this baseline lacked: `keep_lane()`, a Table 2
spatial modifier. Its corner is the tightest manoeuvre in the town and
pure-pursuit exits it with ~2 m of lateral error, so `drive()`'s per-tick
`get_waypoint()` hands the ego to the neighbouring lane. `keep_lane()` latches
the lane the connector fed it into and holds it. `ram` used to mask this by
homing on the ego's actual position.

### Keys and flags

In a window: `SPACE` pause, `TAB` next camera target, `F` toggle follow,
`+`/`-` zoom, `Q` quit.

| Flag | Purpose |
|---|---|
| `--backend pygame` | run on the local simulator instead of CARLA |
| `--list-towns` | print the local road networks and the CARLA aliases |
| `--town NAME` | override the scenario's `map_file` |
| `--junction-turn` | manoeuvre `drive()` takes at a junction |
| `--render-mode` | `auto` / `window` / `headless` / `off` |
| `--render-scale` | pixels per metre (default: follow at 7, or fit the town) |
| `--no-follow` | keep the whole town in frame instead of chasing an actor |
| `--realtime` | play at wall-clock speed rather than as fast as possible |

`--record-video`, `--metrics-out`, `--ego-policy`, `--policy-param` and
`--sim-duration` work identically on both backends; the metrics JSON has the
same schema either way.

## 2. Full run (CARLA + MP4)

Start a CARLA 0.9.16 server (RPC port 2000 by default), then:

```bash
python -m osc2carla scenarios/closed_loop_demo.osc \
    --host 127.0.0.1 --port 2000 \
    --sim-duration 22 \
    --record-video closed_loop_demo.mp4 \
    --record-actor hero --record-fps 20 \
    --record-width 1280 --record-height 720
```

```bash
python -m osc2carla scenarios/scenario_collision.osc \
    --sim-duration 22 \
    --record-video scenario_collision.mp4 \
    --record-actor ego
```

On this cluster, one GPU job boots the server and records both demos:

```bash
sbatch sbatch_closed_loop_demo.sh
```

Outputs go to `../install/run_output/<name>/`:

- `<name>.mp4` — annotated chase-cam video
- `<name>.mp4_frames/` — raw PNGs
- `dry_run.log` — behaviour tree
- `record.log` — spawn + tick trace

## Expected results

Qualitative behaviour should match. Exact spawn coordinates and collision counts can shift with CARLA spawn-point order.

**`closed_loop_demo`** (~22 s, **0 collisions**)

| t (s) | What you should see | Mechanism |
|---|---|---|
| 0 | Tesla at a map spawn; green Audi ~35 m ahead on the same road | relative topological `position(ahead_of:)` |
| 0–7 | Tesla closes the gap (~30 kph) | PID `drive` + `speed` |
| ~7 | Tesla stops closing, matches the Audi | `rise(object_distance < 20m)` |
| 12 | Audi brake-checks; Tesla brakes on the same tick | `emit NPC_BRAKING` / `wait @NPC_BRAKING` |
| ~15–22 | both stopped, gap held | `wait speed < 0.3kph` + 0-speed hold `drive()` |

**`scenario_collision`** (~22 s, **many** collision events)

NPC spawns ~25 m ahead, rotated 180°, then `ram(target: ego)` at full throttle. The ego-mounted collision sensor overlays impulse on the frames. First impact is around t ≈ 2.5 s; the NPC keeps pushing through the intersection.

## Coverage vs. the paper

Implemented and used by the demos: `serial` / `parallel` / `one_of`, `wait` / `emit`, `drive`+`speed`, `keep_lane`, `change_speed`, `change_lane`, `set_lights`, `assign_celestial_position`, relative and absolute `position(…, at: start)`, physical units, `keep(it.field == literal)`, live `speed` / `ahead_of` / `object_distance`.

Every one of those appears in the paper — in Table 2's capability checklist, in
Listings 1–3, or both.

**Not in the paper: `ram(target:)`.** It was a local extension registered
through the `MethodRegistry` (which Table 2 does sanction, under *Extensibility
→ Custom actions*), used to force the adversarial outcomes. The word does not
occur anywhere in the paper, whose action vocabulary has no pursuit primitive
and whose only case study is a collision-*avoidance* scenario. It has been
removed from every scenario under `scenarios/benchmark/` and
`scenarios/local/benchmark/`; the striking vehicle now simply keeps driving its
own lane, which in a runtime with no yielding term is exactly what failing to
yield looks like. `scenarios/scenario_collision.osc` still uses it — ramming
the ego is that demo's entire premise, so it cannot be expressed without it.

Silently missing or degraded (details in [PIPELINE_WALKTHROUGH.md](PIPELINE_WALKTHROUGH.md)):

- `lane(…, at: start)` is parsed and **ignored** (default spawn point)
- `follow_path`, `follow_trajectory`, time gap / space gap / headway,
  pedestrians, traffic lights, road conditions, most `keep()` forms — all
  claimed ✓ in the paper's Table 2
- unmapped actions compile to a silent `Success` leaf named `Unmapped[…]`
- `hello_world.osc` therefore **compiles** but does not run as in the paper on this cluster (no Town06; no lane spawn)

## CLI

```
python -m osc2carla <scenario.osc> [options]
```

| Flag | Purpose |
|---|---|
| `--dry-run` | parse + analyse + print the behaviour tree; no CARLA |
| `--emit-python PATH` | write a standalone Python script and exit |
| `--record-video PATH` | chase-cam MP4 + collision overlay |
| `--record-actor NAME` | binding to attach the camera to (`hero`, `ego`, …) |
| `--sim-duration SEC` | stop after this many simulated seconds (default: largest `wait elapsed` in the file) |
| `--host` / `--port` | CARLA RPC (default `127.0.0.1:2000`) |
| `--backend {carla,pygame}` | execution backend; `pygame` is the bundled local simulator (see §1b) |
| `--metrics-out PATH` | JSON run summary (collision occurrence, impulses, motion stats) |
| `--ego-policy NAME` | hand the ego's actuation to an external policy (`idm`, `constant`, or `module:Class`) |
