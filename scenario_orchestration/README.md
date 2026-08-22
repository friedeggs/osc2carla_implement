# External repositories

Every scenario execution method and ego policy lives here as an independent
repository, normally a git submodule:

```bash
git submodule update --init third_party/orchestration
```

The central harness never imports these repositories' internal modules. The
integration boundary is a subprocess and two JSON documents (DESIGN.md
section 5), so a method may use whatever Python, CUDA, or simulator version it
needs.

## What an execution-method repository must expose

```text
<repository>/
└── scenario_orchestration/
    ├── capabilities.json
    └── run.py
```

`run.py` is invoked as:

```bash
python scenario_orchestration/run.py \
    --scenario-request /path/to/request.json \
    --policy-request /path/to/policy.json \
    --output-dir /path/to/results/raw/<experiment_id>
```

### Inputs

`request.json` (a `ScenarioRequest`) carries the scenario family's semantic
identity, the evaluation protocol, the seed, and the method's own native
implementation entry:

```json
{
  "schema_version": "1.0.0",
  "experiment_id": "red_light__orchestration__idm__s003",
  "scenario_family": "red_light",
  "semantic_id": "red_light",
  "algorithm": "orchestration",
  "seed": 3,
  "evaluation": {
    "horizon_s": 20,
    "target_event": "red_light_interaction",
    "tick_rate_hz": 10,
    "success_criteria": ["scenario_realized", "no_collision"]
  },
  "implementation": {
    "family": "red_light",
    "method": "orchestration",
    "native_id": "red_light",
    "parameters": {"town": "Town05"}
  },
  "metrics": ["scenario_realized", "time_to_event", "collision", "near_collision"]
}
```

`policy.json` (a `PolicyRequest`) describes the ego policy in method-agnostic
terms. Translating it into the repository's own policy plumbing is the
repository's job — that is what keeps integration cost at `M + P` instead of
`M x P`:

```json
{
  "schema_version": "1.0.0",
  "name": "idm",
  "interface": "ego_policy_v1",
  "implementation": "idm.policy.IDMPolicy",
  "observation_space": "state",
  "action_space": "control",
  "repository": "third_party/idm",
  "entry_point": "scenario_orchestration/policy.py",
  "checkpoint": null,
  "parameters": {"desired_speed_mps": 8.0},
  "seed": 3
}
```

`SCENARIO_ORCHESTRATION_EXPERIMENT_ID` and `SCENARIO_ORCHESTRATION_SEED` are
also exported, since many runners seed themselves from the environment.

### Outputs

Write `method_result.json` into `--output-dir`:

```json
{
  "status": "success",
  "metrics": {
    "scenario_realized": true,
    "collision": false,
    "time_to_event": 4.25,
    "my_method_specific_counter": 17
  },
  "trace_path": "trace.json",
  "reason": null
}
```

- `status` is one of `success`, `failure`, `timeout`, `error`. Compatibility is
  decided by the harness, so a method never reports `incompatible` itself.
- `metrics` may mix canonical names (see `contracts/metrics.py`) with anything
  else; the harness splits them and preserves the rest under `method_metrics`.
- `trace_path` is relative to `--output-dir` unless absolute. Large traces may
  be written under `results/traces/` and referenced by path.
- A non-zero exit is recorded as `failure`; exceeding the configured budget as
  `timeout`; exiting cleanly without a report as `error`.

Anything the process prints is captured into `stdout.log` and `stderr.log`.

### capabilities.json

A machine-readable copy of what the repository supports. The harness's own
declaration lives in `configs/algorithm/<name>.yaml`; keeping this file in the
method repository lets the two be cross-checked:

```json
{
  "name": "orchestration",
  "schema_version": "1.0.0",
  "scenario_families": ["red_light", "cut_in", "lane_change"],
  "policy_interfaces": ["ego_policy_v1"],
  "observation_spaces": ["state", "sensor"],
  "action_spaces": ["control", "trajectory", "waypoints"],
  "closed_loop": true
}
```

## What an ego-policy repository must expose

```text
<repository>/
└── scenario_orchestration/
    └── policy.py
```

`policy.py` must construct the policy from a `PolicyRequest` and expose the
declared `ego_policy_v1` interface: given an observation in the declared
observation space, return an action in the declared action space. Method
repositories load it through their own runner; the harness only routes the
request.

A working reference implementation of the method side of this contract lives in
`tests/fixtures/stub_method/scenario_orchestration/run.py`.
