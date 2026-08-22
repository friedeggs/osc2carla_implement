#!/usr/bin/env python3
"""Sample the IDM parameter space over the local benchmark, for statistics.

    python experiments/sweep_idm_local.py [-n 64] [--seed 20260818] [--jobs 8]

Why parameters and not repeats
------------------------------
The local simulator is deterministic: a fixed step, no physics substepping, no
server, and a stable spawn-point digest. Running the same configuration twice
reproduces it bit for bit -- verified, and asserted by ``--check-determinism``.
So unlike the CARLA benchmark, where ``REPEATS`` samples the simulator's own
variability, repeats here carry no information at all.

The distribution that does exist is over the *policy*. IDM has six parameters,
and the question the benchmark asks -- does swapping the ego's controller still
produce the scenario's intended outcome? -- has an answer that depends on them.
This script samples that space with a Latin hypercube (stratified on every
axis, so a modest N still covers each parameter's range) and runs every
scenario in the config at each sample -- the four junction ones and the three
highway ones.

The scripted arm has no parameters, so it stays a single deterministic run per
scenario. It is a reference point, not a distribution, and the report says so.

Ranges are the conventional urban-car envelope around the library defaults
(v0 8.333, T 1.5, a_max 1.5, b 2.0, delta 4.0, s0 2.0), wide enough to include
timid and aggressive drivers without leaving the model's valid regime.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: parameter -> (low, high). Units as in policy.IDMPolicy.DEFAULTS.
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "v0":    (5.0, 13.0),    # desired free-flow speed, m/s (18 - 47 kph)
    "T":     (0.6, 2.5),     # desired time headway, s
    "a_max": (0.8, 2.5),     # maximum acceleration, m/s^2
    "b":     (1.2, 3.5),     # comfortable deceleration, m/s^2
    "s0":    (1.0, 4.0),     # minimum standstill gap, m
    "delta": (2.0, 6.0),     # free-flow acceleration exponent
}


def latin_hypercube(n: int, seed: int) -> List[Dict[str, float]]:
    """One stratified sample per parameter per draw, independently shuffled."""
    rng = random.Random(seed)
    out: List[Dict[str, float]] = [{} for _ in range(n)]
    for name, (lo, hi) in PARAM_RANGES.items():
        strata = list(range(n))
        rng.shuffle(strata)
        for i, k in enumerate(strata):
            u = (k + rng.random()) / n           # jitter inside stratum k
            out[i][name] = round(lo + u * (hi - lo), 4)
    return out


def run_one(scenario: str, duration: int, turn: str, town: str, tag: str,
            out_dir: str, policy_args: List[str], python: str) -> Tuple[str, int]:
    cmd = [python, "-m", "osc2carla",
           os.path.join("scenarios", "local", "benchmark", f"{scenario}.osc"),
           "--backend", "pygame", "--town", town,
           "--junction-turn", turn,
           "--render-mode", "off",
           "--record-actor", "ego",
           "--sim-duration", str(duration),
           "--metrics-out", os.path.join(out_dir, f"{tag}.json")] + policy_args
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        with open(os.path.join(out_dir, f"{tag}.log"), "w") as fh:
            fh.write(proc.stderr)
    return tag, proc.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", "--samples", type=int, default=64,
                    help="IDM parameter samples (default 64)")
    ap.add_argument("--seed", type=int, default=20260818,
                    help="RNG seed for the hypercube (default 20260818)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel runs")
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "benchmark_local.json"))
    ap.add_argument("-o", "--out-dir", default=os.path.join(HERE, "results_local_sweep"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--check-determinism", action="store_true",
                    help="Run one cell three times and assert the metrics match.")
    args = ap.parse_args(argv)

    with open(args.config) as fh:
        cfg = json.load(fh)
    # town matters as much as junction_turn: the junction scenarios are written
    # against `grid`, the highway ones against `highway` / `two_lane`. Running
    # one on the other's road network still runs -- the actors just land
    # somewhere that stages nothing -- so the town has to come from the config
    # rather than be assumed.
    scenarios = [(s["name"], s["sim_duration"], s.get("junction_turn", "straight"),
                  s.get("town", "grid"))
                 for s in cfg["scenarios"]]
    os.makedirs(args.out_dir, exist_ok=True)

    jobs: List[tuple] = []
    # scripted reference: one deterministic run per scenario
    for name, dur, turn, town in scenarios:
        jobs.append((name, dur, turn, town, f"{name}__scripted", []))
    # idm arm: one run per scenario per parameter sample
    samples = latin_hypercube(args.samples, args.seed)
    for i, params in enumerate(samples, start=1):
        pargs: List[str] = ["--ego-policy", "idm"]
        for k, v in sorted(params.items()):
            pargs += ["--policy-param", f"{k}={v}"]
        for name, dur, turn, town in scenarios:
            jobs.append((name, dur, turn, town, f"{name}__idm__r{i:03d}", pargs))

    if args.check_determinism:
        name, dur, turn, town = scenarios[0]
        seen = set()
        for k in range(3):
            run_one(name, dur, turn, town, "determinism_probe", args.out_dir,
                    [], args.python)
            with open(os.path.join(args.out_dir, "determinism_probe.json")) as fh:
                d = json.load(fh)
            seen.add((d["first_collision_time"], d["peak_impulse"],
                      d["distance_travelled_m"], d["n_collision_events"]))
        os.remove(os.path.join(args.out_dir, "determinism_probe.json"))
        assert len(seen) == 1, f"local backend is not deterministic: {seen}"
        print(f"[sweep] determinism verified on {name}: 3 runs, 1 distinct result")

    print(f"[sweep] {len(samples)} parameter samples x {len(scenarios)} scenarios "
          f"+ {len(scenarios)} scripted = {len(jobs)} runs, seed={args.seed}")
    failures = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(run_one, n, d, t, w, tag, args.out_dir, p, args.python)
                for (n, d, t, w, tag, p) in jobs]
        for fut in futs:
            tag, rc = fut.result()
            done += 1
            if rc != 0:
                failures += 1
                print(f"[sweep] FAILED {tag} rc={rc}")
            if done % 40 == 0:
                print(f"[sweep] {done}/{len(jobs)}")

    with open(os.path.join(args.out_dir, "samples.json"), "w") as fh:
        json.dump({"seed": args.seed,
                   "n_samples": args.samples,
                   "ranges": PARAM_RANGES,
                   "defaults_note": "library defaults are v0 8.333, T 1.5, "
                                    "a_max 1.5, b 2.0, delta 4.0, s0 2.0",
                   "sampling": "Latin hypercube, stratified per parameter",
                   "samples": {f"r{i:03d}": s for i, s in enumerate(samples, 1)}},
                  fh, indent=2, sort_keys=True)
    print(f"[sweep] {done - failures}/{done} runs ok -> {args.out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
