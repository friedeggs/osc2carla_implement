# This repository, behind the `scenario_orchestration` harness

Three files, and nothing in the rest of the repository knows they exist:

```text
scenario_orchestration/
├── capabilities.json            what this repository supports, machine-readable
├── run.py                       the standardized entry point the harness runs
└── osc2carla_policy_bridge.py   an external ego_policy_v1 policy, loaded in
```

The contract itself -- the two JSON documents, the output format, the status
vocabulary -- is the harness's, and lives in its `third_party/README.md`. This
file is only about what is specific to osc2carla_implement.

## The entry point

```bash
python scenario_orchestration/run.py \
    --scenario-request request.json \
    --policy-request  policy.json \
    --output-dir      results/raw/<experiment_id>
```

`run.py` turns the scenario request into an `osc2carla` invocation -- which
`.osc` file, on which backend, for how long, at what step -- and the policy
request into `--ego-policy` and its parameters. It writes `method_result.json`,
and copies the run summary beside it as `osc2carla_metrics.json`.

Everything operator-side is an environment variable, because it describes the
machine rather than the experiment. `run.py`'s module docstring is the list.

## Ego policies

Three routes, tried in this order:

| the request asks for | what happens |
|---|---|
| `idm`, `idm_assertive`, `idm_conservative`, `constant` | realized natively; the request's SI parameter names are translated into this repository's (`desired_speed_mps` -> `v0`, ...) |
| `scripted` / `none` | no ego policy: the compiled behaviour tree drives the ego, as it drives everything else |
| anything else | loaded from its own repository's `scenario_orchestration/policy.py` through `osc2carla_policy_bridge.py` |

An analytic policy is fully described by its parameters, which is why it needs
no policy repository and why every method should realize it natively rather than
import one.

## Sensor policies

A policy declaring `observation_space: sensor` -- SimLingo, TFv6 -- is driven
through the same bridge. What it additionally gets is a **sensor rig**, and the
rig is the policy's own: `osc2carla/backend/sensors.py` attaches exactly what the
policy's `sensors()` returns and chooses nothing itself. A rig chosen here would
be one no checkpoint was trained behind, and the resulting numbers would be this
repository's behaviour published under the model's name.

```
policy.sensors()  ->  [{"name": "rgb_front", "width": 1024, "height": 512,
                        "fov": 110.0, "x": -1.5, "y": 0.0, "z": 2.0}, ...]
                          |
                  osc2carla.backend.sensors.SensorRig
                          |
observation["sensor"]["cameras"]["rgb_front"] = HxWx3 uint8 RGB
```

Cameras, `sensor.lidar.ray_cast` and `sensor.other.radar` are supported. Every
sensor is filed under `cameras` by its declared name, LiDAR and radar included:
that is the key the installed policy adapters read, and renaming it here would
be renaming it in repositories that are not ours.

Four things about that path are load-bearing:

* **Capture is synchronous.** CARLA delivers sensor data asynchronously even in
  synchronous mode. Each sensor has its own queue and a capture blocks for the
  measurement *stamped with the tick being described*, discarding anything
  older. Keeping the newest frame instead hands the policy an image from an
  arbitrary earlier tick under load, which is the classic way an agent stops
  being reproducible without anything appearing to fail.
* **Sweeping sensors are retimed to the world's tick rate.** A LiDAR at 20 rev/s
  in a 20 Hz world delivers a full sweep; the same LiDAR in a 60 Hz world
  delivers a 120 degree wedge, and not the one in front.
* **A sensor that delivered nothing is absent, never zero-filled.** A model
  cannot tell an all-zero raster from a clear road, so the policy has to be the
  one that decides what a missing sensor means. The count is reported per run.
* **The route is resampled.** `ExternalEgoController` samples the road every 2 m
  because that is what its own controllers want; the reference agents in this
  family index a 1 m grid from 2.5 m ahead. The bridge converts, so a target
  point lands where the model expects it rather than twice as far out.

`--policy-hz` decouples the policy's decision rate from the simulation tick
rate, holding the last command in between. It defaults to deciding on every
tick, which is what the analytic policies have always done; a VLA asked for 20
decisions per simulated second spends the run inside `forward`.

The rig needs the **CARLA backend**. The bundled local simulator provides
`sensor.other.collision` and nothing else, so `run.py` refuses a sensor policy
there with that reason rather than driving it against an empty rig.

### What the policy sees

```python
{"t": 4.2,
 "ego": {"speed_mps": 7.9, "x": .., "y": .., "heading_rad": ..},
 "route": [[x, y], ...],                 # ego frame: +x forward, +y right
 "speed_limit_kph": 50.0,
 "sensor": {"cameras": {...}, "frame": 91823},
 "leader": {"gap_m": 12.4, "speed_mps": 6.1, ...} | None,
 "speed_mps": 7.9, "x": .., "y": .., "heading_rad": ..}   # the flat keys, kept
```

and returns `{"control": {"throttle": .., "steer": .., "brake": ..}, ...}`. The
older flat form and a bare `acceleration_mps2` are still accepted.

Relative `checkpoint` and `parameters.weights` paths in the policy request are
resolved against the harness root before the policy sees them -- the harness
writes them relative to itself, and the policy resolves them against a working
directory that is the method's.

### What a run records about its own sensing

`osc2carla_metrics.json` carries `ego_policy_detail`: the rig asked for, the rig
actually attached, the rate it was sampled at, anything that failed to attach or
dropped a frame, the decision count, and whatever the policy reports about
itself -- for a VLA, the text it generated. A result is only readable against
the sensing it actually had.

`OSC2CARLA_RECORD_VIDEO=1` writes `<family>.mp4` beside it: the chase cam, and
under it the policy's own frames beside the command it returned. That panel is
the cheapest check on a rig there is -- a camera mounted wrong, pointed
backwards, or a frame late produces entirely plausible metrics and an obviously
wrong picture.

## Tests

No server and no checkpoint required:

```bash
PYTHONPATH=. python tests/test_sensor_rig.py    # the rig and the bridge
PYTHONPATH=. python tests/test_idm_policy.py    # the analytic arm
PYTHONPATH=. python tests/test_localsim.py      # the local backend
```
