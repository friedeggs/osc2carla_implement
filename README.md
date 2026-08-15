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
├── grammar/              # OpenSCENARIO 2.1 ANTLR grammar (from py-osc2)
├── stdlib/               # types.osc + domain.osc stubs used at analyse time
├── scenarios/            # .osc inputs (see table below)
├── third_party/py-osc2/  # vendored grammar source + MPL-2.0 license
├── requirements.txt
├── env.sh                # sets OSC2CARLA_ROOT and PYTHONPATH
├── run_dry_run.sh        # no CARLA: parse + print behaviour trees
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

`--dry-run` on `hello_world.osc` is the compile-only check against the paper’s behaviour tree. Running it live would need Town06 and lane-based spawn, which this baseline does not implement (see “Coverage vs. the paper” below).

## Requirements

- Python **3.8** (CARLA 0.9.16’s official wheel is `cp38`)
- `pip install -r requirements.txt`
- For a live run: **CARLA 0.9.16** server + its `PythonAPI/carla` on `PYTHONPATH`
- For `--record-video`: OpenCV (`cv2`) and `ffmpeg`
- GPU for the UE4 server (off-screen rendering is fine)

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

Implemented and used by the demos: `serial` / `parallel` / `one_of`, `wait` / `emit`, `drive`+`speed`, `change_speed`, `change_lane`, `ram`, `set_lights`, `assign_celestial_position`, relative and absolute `position(…, at: start)`, physical units, `keep(it.field == literal)`, live `speed` / `ahead_of` / `object_distance`.

Silently missing or degraded (details in [PIPELINE_WALKTHROUGH.md](PIPELINE_WALKTHROUGH.md)):

- `lane(…, at: start)` is parsed and **ignored** (default spawn point)
- `keep_lane`, `follow_path`, pedestrians, traffic lights, most `keep()` forms
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
