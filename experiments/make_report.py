#!/usr/bin/env python3
"""Build the ego-policy benchmark report from experiments/results/**.json.

    python experiments/make_report.py [RESULTS_DIR] [-o OUT_HTML]

Emits a self-contained HTML page (inline SVG charts, no external assets) and
prints a text summary. Deliberately dependency-free: the analysis environment
has numpy but no matplotlib, and inline SVG renders more crisply in a report
that is meant to be shared as a web page.

Accepts both ``<scenario>__<policy>.json`` (single run) and
``<scenario>__<policy>__rN.json`` (repeat N), and aggregates over repeats --
these outcomes are not fully repeatable, so a per-cell rate is the honest
statistic rather than a single sample.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

SERIES_COLORS = {"scripted": "var(--series-a)", "idm": "var(--series-b)"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load(results_dir: str, config_path: str
         ) -> Tuple[dict, Dict[Tuple[str, str], List[dict]]]:
    with open(config_path) as fh:
        cfg = json.load(fh)
    runs: Dict[Tuple[str, str], List[dict]] = {}
    pattern = os.path.join(results_dir, "**", "*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        if os.path.basename(path) == "summary.json":
            continue
        with open(path) as fh:
            data = json.load(fh)
        if "collision_occurred" not in data:
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        parts = base.split("__")
        if len(parts) < 2:
            continue
        scenario, policy = parts[0], parts[1]
        runs.setdefault((scenario, policy), []).append(data)
    return cfg, runs


def _median(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def evaluate(cfg, runs) -> List[dict]:
    """Aggregate each (scenario, policy) cell against the scenario's intent."""
    policies = [p["id"] for p in cfg["policies"]]
    rows = []
    for sc in cfg["scenarios"]:
        expected = bool(sc["expect_collision"])
        intended = sc.get("intended_partner")
        for pid in policies:
            cell = runs.get((sc["name"], pid))
            if not cell:
                continue
            n = len(cell)
            occurred = [bool(r.get("collision_occurred")) for r in cell]
            roles = [[x for x in (r.get("collision_partner_roles") or []) if x]
                     for r in cell]
            # Headline proxy: outcome matches the scenario's declared intent.
            proxy = [o == expected for o in occurred]
            # Refinement: for a crash scenario, was it the scripted antagonist?
            if expected:
                hit_intended = [p and (intended in rl) for p, rl in zip(proxy, roles)]
            else:
                hit_intended = list(proxy)
            partner_counts = Counter(x for rl in roles for x in set(rl))
            rows.append({
                "scenario": sc["name"],
                "policy": pid,
                "group": sc.get("group", "all"),
                "town": sc.get("town"),
                "intent": sc["intent"],
                "expect_collision": expected,
                "intended_partner": intended,
                "n_runs": n,
                "n_collision": sum(occurred),
                "collision_rate": sum(occurred) / n,
                "n_proxy": sum(proxy),
                "proxy_rate": sum(proxy) / n,
                "n_intended": sum(hit_intended),
                "intended_rate": sum(hit_intended) / n,
                "median_first_t": _median([r.get("first_collision_time") for r in cell]),
                "median_peak": _median([r.get("peak_impulse") for r in cell
                                        if r.get("collision_occurred")]),
                "median_events": _median([r.get("n_collision_events") for r in cell]),
                "mean_speed": statistics.mean([r.get("mean_speed_mps", 0.0) for r in cell]),
                "mean_distance": statistics.mean([r.get("distance_travelled_m", 0.0)
                                                  for r in cell]),
                "partners": partner_counts,
            })
    return rows


# --------------------------------------------------------------------------
# Statistics over a sampled parameter space
# --------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial rate.

    Preferred over the normal approximation because these rates sit near 0 and
    1, where the normal interval runs outside [0, 1] and understates the
    uncertainty of a unanimous cell.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _intended(record: dict, expect: bool, partner: Optional[str]) -> bool:
    occurred = bool(record.get("collision_occurred"))
    if occurred != expect:
        return False
    if not expect:
        return True
    return partner in (record.get("collision_partner_roles") or [])


def parameter_sensitivity(cfg, runs) -> List[dict]:
    """Which policy parameters move the outcome, by median split.

    For every scenario and every sampled parameter, the runs are split at that
    parameter's median and the intended-outcome rate compared across the two
    halves. A median split rather than a fitted model: with a few dozen samples
    over six parameters, a difference of proportions is the strongest claim the
    data supports, and it needs no assumption about the shape of the response.
    """
    out: List[dict] = []
    for sc in cfg["scenarios"]:
        expect = bool(sc["expect_collision"])
        partner = sc.get("intended_partner")
        cell = [r for r in runs.get((sc["name"], "idm"), []) if r.get("policy_params")]
        if len(cell) < 8:
            continue
        names = sorted({k for r in cell for k in r["policy_params"]})
        rows = []
        for name in names:
            pairs = [(float(r["policy_params"][name]),
                      _intended(r, expect, partner)) for r in cell
                     if name in r["policy_params"]]
            if len(pairs) < 8:
                continue
            cut = statistics.median(v for v, _ in pairs)
            lo = [ok for v, ok in pairs if v <= cut]
            hi = [ok for v, ok in pairs if v > cut]
            if not lo or not hi:
                continue
            r_lo, r_hi = sum(lo) / len(lo), sum(hi) / len(hi)
            rows.append({
                "param": name, "cut": cut,
                "n_lo": len(lo), "rate_lo": r_lo, "ci_lo": wilson(sum(lo), len(lo)),
                "n_hi": len(hi), "rate_hi": r_hi, "ci_hi": wilson(sum(hi), len(hi)),
                "delta": r_hi - r_lo,
                "values": [(v, ok) for v, ok in pairs],
            })
        rows.sort(key=lambda d: -abs(d["delta"]))
        out.append({"scenario": sc["name"], "n": len(cell), "params": rows})
    return out


def strip_plot(values: List[Tuple[float, bool]], label: str,
               width: int = 620, height: int = 92) -> str:
    """One dot per sample, placed on the parameter axis and split by outcome."""
    if not values:
        return ""
    xs = [v for v, _ in values]
    lo, hi = min(xs), max(xs)
    span = (hi - lo) or 1.0
    pad_l, pad_r, mid = 54, 16, height / 2
    plot_w = width - pad_l - pad_r

    def px(v: float) -> float:
        return pad_l + (v - lo) / span * plot_w

    dots = []
    for v, ok in values:
        y = mid - 17 if ok else mid + 17
        fill = "var(--ok-fg)" if ok else "var(--bad-fg)"
        dots.append(f'<circle cx="{px(v):.1f}" cy="{y:.1f}" r="3.4" fill="{fill}" '
                    f'fill-opacity="0.75"/>')
    ticks = []
    for frac in (0.0, 0.5, 1.0):
        v = lo + frac * span
        ticks.append(f'<text x="{px(v):.1f}" y="{height - 4}" class="tick" '
                     f'text-anchor="middle">{v:.2f}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" class="chart" '
            f'role="img" aria-label="{_esc(label)} sensitivity">'
            f'<line x1="{pad_l}" y1="{mid}" x2="{width - pad_r}" y2="{mid}" '
            f'class="axis"/>'
            f'<text x="6" y="{mid - 13}" class="tick">intended</text>'
            f'<text x="6" y="{mid + 21}" class="tick">not</text>'
            f'{"".join(dots)}{"".join(ticks)}</svg>')


def sensitivity_section(sens: List[dict]) -> str:
    if not sens:
        return ""
    parts = ["<h2>Policy-parameter sensitivity</h2>",
             "<p>The local simulator is deterministic, so repeats of one configuration "
             "carry no information. The distribution here is over the <em>policy</em>: "
             "each IDM run draws a different parameter vector from a Latin hypercube "
             "over the six IDM parameters. These tables ask which of them move the "
             "outcome.</p>"]
    for entry in sens:
        parts.append(f'<h3><code>{_esc(entry["scenario"])}</code> '
                     f'<span class="key">({entry["n"]} parameter samples)</span></h3>')
        rows = ["<div class='scroll'><table class='data'><thead>"
                "<tr><th>parameter</th><th>median</th>"
                "<th>intended rate, low half</th><th>intended rate, high half</th>"
                "<th>&Delta;</th></tr></thead><tbody>"]
        for r in entry["params"]:
            strong = ' class="mark"' if abs(r["delta"]) >= 0.25 else ""
            rows.append(
                f'<tr{strong}><td><code>{_esc(r["param"])}</code></td>'
                f'<td>{r["cut"]:.2f}</td>'
                f'<td>{r["rate_lo"]:.0%} <span class="key">'
                f'[{r["ci_lo"][0]:.0%}&ndash;{r["ci_lo"][1]:.0%}], n={r["n_lo"]}</span></td>'
                f'<td>{r["rate_hi"]:.0%} <span class="key">'
                f'[{r["ci_hi"][0]:.0%}&ndash;{r["ci_hi"][1]:.0%}], n={r["n_hi"]}</span></td>'
                f'<td>{r["delta"]:+.0%}</td></tr>')
        rows.append("</tbody></table></div>")
        parts.append("".join(rows))
        top = entry["params"][0] if entry["params"] else None
        if top is not None and abs(top["delta"]) >= 0.15:
            parts.append(f'<p class="key">Most influential: <code>'
                         f'{_esc(top["param"])}</code>. Each dot is one sample, '
                         f'placed by its <code>{_esc(top["param"])}</code> value.</p>')
            parts.append(strip_plot(top["values"], top["param"]))
        else:
            parts.append('<p class="key">No parameter shifts the intended rate by '
                         'more than 15 points across its median split: within this '
                         'envelope the outcome is decided by the scenario, not by '
                         'the tuning.</p>')
    parts.append('<p class="key">Brackets are 95% Wilson intervals. A median split '
                 'is a difference of proportions, not a fitted response &mdash; with '
                 'a few dozen samples over six parameters it is the strongest claim '
                 'the data supports, and six comparisons per scenario means an '
                 'isolated 15-point gap is unremarkable.</p>')
    return "".join(parts)


# --------------------------------------------------------------------------
# SVG charting
# --------------------------------------------------------------------------

def _esc(s: Any) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def grouped_bars(categories: List[str], series: List[Dict[str, Any]],
                 ylabel: str = "", value_fmt: str = "{:.1f}",
                 height: int = 240, na_label: str = "n/a",
                 vmax_override: Optional[float] = None) -> str:
    """Grouped vertical bar chart as inline SVG.

    A value of ``None`` renders as an explicit label rather than a zero-height
    bar: some cells have a genuinely absent measurement (no collision means no
    impact time) and drawing that as 0 would assert something false.
    """
    pad_l, pad_r, pad_t, pad_b = 54, 14, 16, 48
    width = 640
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    vals = [v for s in series for v in s["values"] if v is not None]
    vmax = vmax_override if vmax_override is not None else (max(vals) if vals else 1.0)
    if vmax <= 0:
        vmax = 1.0
    if vmax_override is None:
        vmax *= 1.2

    n_cat = len(categories)
    group_w = plot_w / max(n_cat, 1)
    bar_w = min(46.0, group_w / (len(series) + 0.8))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'preserveAspectRatio="xMidYMid meet" class="chart">']
    for i in range(5):
        frac = i / 4.0
        y = pad_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+plot_w}" '
                     f'y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+4:.1f}" class="tick" '
                     f'text-anchor="end">{value_fmt.format(vmax*frac)}</text>')
    if ylabel:
        cy = pad_t + plot_h / 2
        parts.append(f'<text x="13" y="{cy:.1f}" class="axis-label" '
                     f'text-anchor="middle" transform="rotate(-90 13 {cy:.1f})">'
                     f'{_esc(ylabel)}</text>')

    for ci, cat in enumerate(categories):
        gx = pad_l + ci * group_w
        total_w = len(series) * bar_w + (len(series) - 1) * 6
        x0 = gx + (group_w - total_w) / 2
        for si, s in enumerate(series):
            v = s["values"][ci]
            x = x0 + si * (bar_w + 6)
            color = SERIES_COLORS.get(s["id"], "var(--series-a)")
            if v is None:
                parts.append(f'<text x="{x+bar_w/2:.1f}" y="{pad_t+plot_h-6}" '
                             f'class="na" text-anchor="middle">{na_label}</text>')
                continue
            h = plot_h * (min(v, vmax) / vmax)
            y = pad_t + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                         f'height="{max(h,0.8):.1f}" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" class="barval" '
                         f'text-anchor="middle">{value_fmt.format(v)}</text>')
        parts.append(f'<text x="{gx+group_w/2:.1f}" y="{height-pad_b+20}" '
                     f'class="cat" text-anchor="middle">{_esc(cat)}</text>')

    parts.append(f'<line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" '
                 f'y2="{pad_t+plot_h}" class="axis"/>')
    parts.append("</svg>")
    return "".join(parts)


def legend(series: List[Dict[str, Any]]) -> str:
    items = "".join(
        f'<span class="legend-item"><i style="background:'
        f'{SERIES_COLORS.get(s["id"], "var(--series-a)")}"></i>{_esc(s["label"])}</span>'
        for s in series)
    return f'<div class="legend">{items}</div>'


def group_summary(cfg, rows) -> List[dict]:
    """Aggregate the per-cell rows into the suite's scenario families.

    The benchmark is not one homogeneous set: the junction scenarios turn
    across conflicting traffic and the highway scenarios do not turn at all,
    and a policy swap does not have to behave the same way on both. Reporting
    a single mean over all seven hides exactly the comparison the highway
    scenarios were added to make. Groups come from each scenario's ``group``
    key; a config that declares none is reported as one family.
    """
    order, seen = [], set()
    for sc in cfg["scenarios"]:
        g = sc.get("group", "all")
        if g not in seen:
            seen.add(g)
            order.append(g)
    labels = cfg.get("group_labels", {})
    out = []
    for g in order:
        entry = {"group": g, "label": labels.get(g, g), "policies": {}}
        entry["scenarios"] = [sc["name"] for sc in cfg["scenarios"]
                              if sc.get("group", "all") == g]
        entry["towns"] = sorted({sc.get("town") for sc in cfg["scenarios"]
                                 if sc.get("group", "all") == g
                                 and sc.get("town")})
        for pid in [p["id"] for p in cfg["policies"]]:
            cells = [r for r in rows if r["group"] == g and r["policy"] == pid]
            if not cells:
                continue
            n = sum(c["n_runs"] for c in cells)
            k = sum(c["n_intended"] for c in cells)
            entry["policies"][pid] = {
                "n_cells": len(cells), "n_runs": n, "n_intended": k,
                "rate": (k / n) if n else 0.0, "ci": wilson(k, n),
                "n_full": sum(1 for c in cells if c["intended_rate"] >= 0.999),
                "speed": statistics.mean([c["mean_speed"] for c in cells]),
            }
        out.append(entry)
    return out


def group_section(cfg, rows, groups) -> str:
    """Per-family result table. Skipped when the suite is one family."""
    if len(groups) < 2:
        return ""
    pol = [(p["id"], p["label"]) for p in cfg["policies"]]
    head = ("<tr><th>Family</th><th>Scenarios</th><th>Map(s)</th>" +
            "".join(f"<th>{_esc(lab)}</th>" for _, lab in pol) + "</tr>")
    body = []
    for g in groups:
        cells = []
        for pid, _ in pol:
            e = g["policies"].get(pid)
            if e is None:
                cells.append('<td class="num">—</td>')
                continue
            lo, hi = e["ci"]
            cells.append(
                f'<td class="num">{e["rate"]*100:.0f}% '
                f'<span class="key">({e["n_intended"]}/{e["n_runs"]}) '
                f'[{lo*100:.0f}&ndash;{hi*100:.0f}]</span><br>'
                f'<span class="key">{e["n_full"]}/{e["n_cells"]} scenarios at '
                f'100%, mean {e["speed"]:.2f} m/s</span></td>')
        body.append(
            f'<tr><td><b>{_esc(g["label"])}</b></td>'
            f'<td>{"".join(f"<code>{_esc(n)}</code> " for n in g["scenarios"])}</td>'
            f'<td class="key">{_esc(", ".join(g["towns"]) or "—")}</td>'
            f'{"".join(cells)}</tr>')
    note = cfg.get("group_note", "")
    return ("<h2>By scenario family</h2>"
            '<div class="scroll"><table class="data"><thead>' + head +
            "</thead><tbody>" + "".join(body) + "</tbody></table></div>" +
            (f'<p class="key">{note}</p>' if note else "") +
            '<p class="key">Rates pool every run in the family, so a family with '
            'more scenarios carries more weight; the per-scenario breakdown is in '
            'the matrix above. Brackets are 95% Wilson intervals.</p>')


def outcome_matrix(rows, cfg) -> str:
    scenarios = [s["name"] for s in cfg["scenarios"]]
    policies = [(p["id"], p["label"]) for p in cfg["policies"]]
    head = "".join(f"<th>{_esc(s)}</th>" for s in scenarios)
    body = []
    for pid, plabel in policies:
        cells = []
        for sc in scenarios:
            r = next((x for x in rows if x["scenario"] == sc and x["policy"] == pid), None)
            if r is None:
                cells.append('<td class="cell miss">—</td>')
                continue
            rate = r["intended_rate"]
            if rate >= 0.999:
                cls = "ok"
            elif rate <= 0.001:
                cls = "bad"
            else:
                cls = "warn"
            note = f'{r["n_intended"]}/{r["n_runs"]} runs'
            if r["n_proxy"] > r["n_intended"]:
                note += f' · {r["n_proxy"] - r["n_intended"]} wrong-antagonist'
            cells.append(f'<td class="cell {cls}"><span class="mark">'
                         f'{rate*100:.0f}%</span><span class="note">{note}</span></td>')
        body.append(f"<tr><th>{_esc(plabel)}</th>{''.join(cells)}</tr>")
    return ('<div class="scroll"><table class="matrix">'
            f"<thead><tr><th></th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def results_table(rows) -> str:
    show_town = any(r.get("town") for r in rows)
    head = ("<tr><th>Scenario</th>" + ("<th>Map</th>" if show_town else "") +
            "<th>Ego policy</th><th>Runs</th>"
            "<th>Collision rate</th><th>Intended-conflict rate</th>"
            "<th>Median first contact</th><th>Median peak impulse</th>"
            "<th>Hit by (runs)</th><th>Mean speed</th></tr>")
    body = []
    for r in rows:
        first = f'{r["median_first_t"]:.2f} s' if r["median_first_t"] is not None else "—"
        peak = f'{r["median_peak"]:,.0f}' if r["median_peak"] else "—"
        partners = ", ".join(f"{k} ({v})" for k, v in r["partners"].most_common()) or "—"
        cr = f'{r["collision_rate"]*100:.0f}% ({r["n_collision"]}/{r["n_runs"]})'
        lo, hi = wilson(r["n_intended"], r["n_runs"])
        ci = (f' <span class="key">[{lo*100:.0f}&ndash;{hi*100:.0f}]</span>'
              if r["n_runs"] > 1 else "")
        ir = f'{r["intended_rate"]*100:.0f}% ({r["n_intended"]}/{r["n_runs"]}){ci}'
        town = (f'<td class="key">{_esc(r.get("town") or "—")}</td>'
                if show_town else "")
        body.append(
            f'<tr><td><code>{_esc(r["scenario"])}</code></td>{town}'
            f'<td>{_esc(r["policy"])}</td><td class="num">{r["n_runs"]}</td>'
            f'<td class="num">{cr}</td><td class="num">{ir}</td>'
            f'<td class="num">{first}</td><td class="num">{peak}</td>'
            f'<td>{_esc(partners)}</td>'
            f'<td class="num">{r["mean_speed"]:.2f} m/s</td></tr>')
    return ('<div class="scroll"><table class="data"><thead>' + head +
            "</thead><tbody>" + "".join(body) + "</tbody></table></div>")


DEFAULT_REPRO = """<pre><code>./experiments/run_experiments.sh                                    # 8 recorded runs
REPEATS=20 ./experiments/run_experiments.sh results/repeats --no-video
python experiments/make_report.py                                   # rebuild this page</code></pre>
<p class="key">Requires a CARLA 0.9.16 server on port 2000 and the cp38 interpreter; the
runner refuses to start on a Python that cannot import CARLA. A no-video run takes ~6 s,
so the repeat sweep is cheap; recording video is what costs time.</p>
</div>"""


DEFAULT_NOTES = """<ul>
<li><b>Repeatability.</b> CARLA's physics substepping is not bit-reproducible, so a
marginal conflict can flip between runs of an identical file. Where a cell carries
repeats, check whether it was unanimous before quoting its rate; where it carries a
single run, treat it as a sample rather than a measurement and re-run with
<code>REPEATS=</code>. The runner exists for that
(<code>REPEATS=20 ./experiments/run_experiments.sh results/repeats --no-video</code>).</li>
<li><b>Instrumentation is excluded.</b> The benchmark scenarios place a ground-decal
marker at each conflict point as a distance reference. CARLA does occasionally report a
contact when the ego drives over it, so the metric counts only vehicle-versus-vehicle
contacts; static-prop contacts are recorded separately as
<code>n_static_contacts</code>. Counting them would let a clean run register as a
crash.</li>
<li><b>The NPCs have no driver model.</b> <code>drive()</code> is a waypoint+PID
controller that never yields and never brakes for anything. Any ego that deviates from
the scripted timing therefore risks being rear-ended by its own scripted follower — which
is exactly what happens in <code>red_light</code> under IDM. That is a property of the
benchmark, not a defect in IDM, and it is the single biggest caveat on these numbers.</li>
<li><b>Lateral control is shared.</b> IDM is longitudinal only; steering reuses the same
pure-pursuit rule as the compiled <code>drive()</code>, so both arms follow the identical
route through each junction and the comparison isolates the longitudinal policy.</li>
</ul>"""


CSS = """
:root{
  --bg:#ffffff; --fg:#15181d; --muted:#5b6472; --line:#e2e6ec; --card:#f7f8fa;
  --series-a:#4f6fb0; --series-b:#c9722f;
  --ok-bg:#e7f4ec; --ok-fg:#1c6b3f; --warn-bg:#fdf2dc; --warn-fg:#8a5a10;
  --bad-bg:#fceaea; --bad-fg:#992222; --accent:#4f6fb0;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14161a; --fg:#e8eaee; --muted:#98a2b3; --line:#2a2f37; --card:#1b1e24;
    --series-a:#7d9ede; --series-b:#e09353;
    --ok-bg:#14301f; --ok-fg:#78d39b; --warn-bg:#33270f; --warn-fg:#e7bd6d;
    --bad-bg:#331a1a; --bad-fg:#f09a9a; --accent:#7d9ede;
  }
}
:root[data-theme="dark"]{
  --bg:#14161a; --fg:#e8eaee; --muted:#98a2b3; --line:#2a2f37; --card:#1b1e24;
  --series-a:#7d9ede; --series-b:#e09353;
  --ok-bg:#14301f; --ok-fg:#78d39b; --warn-bg:#33270f; --warn-fg:#e7bd6d;
  --bad-bg:#331a1a; --bad-fg:#f09a9a; --accent:#7d9ede;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);margin:0;
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:40px 20px 80px}
h1{font-size:1.9rem;line-height:1.25;margin:0 0 6px}
h2{font-size:1.25rem;margin:44px 0 12px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:1.02rem;margin:26px 0 8px}
.sub{color:var(--muted);margin:0 0 8px}
code{background:var(--card);padding:.12em .38em;border-radius:4px;
  font:0.87em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:12px 14px;overflow-x:auto}
pre code{background:none;padding:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:16px 18px;margin:18px 0}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:.86rem;min-width:620px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600}
td.num{text-align:right;font-variant-numeric:tabular-nums}
table.matrix{min-width:520px}
table.matrix th:first-child{width:210px}
.cell{text-align:center;border-radius:6px}
.cell .mark{display:block;font-weight:700;font-size:1.05rem}
.cell .note{display:block;font-size:.72rem;color:var(--muted)}
.cell.ok{background:var(--ok-bg)} .cell.ok .mark{color:var(--ok-fg)}
.cell.warn{background:var(--warn-bg)} .cell.warn .mark{color:var(--warn-fg)}
.cell.bad{background:var(--bad-bg)} .cell.bad .mark{color:var(--bad-fg)}
.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .axis{stroke:var(--muted);stroke-width:1}
.chart .tick,.chart .cat,.chart .barval,.chart .na,.chart .axis-label{
  fill:var(--muted);font-size:11px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.chart .barval{fill:var(--fg);font-size:10.5px}
.chart .cat{fill:var(--fg);font-size:11.5px}
.chart .na{font-style:italic}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:6px 0 2px;font-size:.84rem;color:var(--muted)}
.legend-item{display:flex;align-items:center;gap:7px}
.legend i{width:13px;height:13px;border-radius:3px;display:inline-block}
.key{font-size:.85rem;color:var(--muted);margin-top:8px}
.finding{border-left:3px solid var(--accent);padding:2px 0 2px 14px;margin:16px 0}
.finding b{color:var(--fg)}
ul{padding-left:20px}
"""


_NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _numword(n: int) -> str:
    return _NUMWORD.get(n, str(n))


def build_html(cfg, rows, sens=None) -> str:
    sens = sens or []
    scen = [s["name"] for s in cfg["scenarios"]]
    pol = [(p["id"], p["label"]) for p in cfg["policies"]]
    meta = [{"id": pid, "label": lab} for pid, lab in pol]
    by = {(r["scenario"], r["policy"]): r for r in rows}

    def col(pid, key, pct=False):
        out = []
        for sc in scen:
            r = by.get((sc, pid))
            v = r[key] if r else None
            out.append(v * 100 if (pct and v is not None) else v)
        return out

    s_intended = [{**m, "values": col(m["id"], "intended_rate", pct=True)} for m in meta]
    s_coll = [{**m, "values": col(m["id"], "collision_rate", pct=True)} for m in meta]
    s_first = [{**m, "values": col(m["id"], "median_first_t")} for m in meta]
    s_speed = [{**m, "values": col(m["id"], "mean_speed")} for m in meta]

    n_runs = sum(r["n_runs"] for r in rows)
    idm_rows = [r for r in rows if r["policy"] == "idm"]
    base_rows = [r for r in rows if r["policy"] == "scripted"]
    idm_full = sum(1 for r in idm_rows if r["intended_rate"] >= 0.999)
    base_full = sum(1 for r in base_rows if r["intended_rate"] >= 0.999)

    # Measurement notes are a property of the experiment, not of the report.
    # A config that states its own gets those; otherwise the CARLA benchmark's
    # defaults stand, which is what this generator was first written for.
    repro = cfg.get("report_reproducing")
    repro_html = ("<pre class='note'>" + _esc(repro) + "</pre>") if repro \
        else DEFAULT_REPRO

    custom = cfg.get("report_notes")
    if custom:
        notes_html = "<ul>" + "".join(f"<li>{n}</li>" for n in custom) + "</ul>"
    else:
        notes_html = DEFAULT_NOTES

    groups = group_summary(cfg, rows)
    families_html = group_section(cfg, rows, groups)

    # The suite's shape is a property of the config, not of this generator: it
    # started as four junction scenarios and now carries highway ones too, so
    # every count in the prose below is derived rather than written out.
    n_scen = len(cfg["scenarios"])
    crash = [s for s in cfg["scenarios"] if s["expect_collision"]]
    clean = [s for s in cfg["scenarios"] if not s["expect_collision"]]
    reps = sorted({r["n_runs"] for r in rows})
    reps_txt = (f"Each cell was repeated {reps[0]} times."
                if len(reps) == 1 and reps[0] > 1
                else ("Each cell is a single run." if reps == [1]
                      else f"Cells carry {min(reps)}-{max(reps)} runs each."))
    towns = [t for t in (cfg.get("town_summary"),) if t] or \
        sorted({s.get("town") for s in cfg["scenarios"] if s.get("town")})
    towns_txt = ", ".join(towns) if isinstance(towns, list) else str(towns)
    title = cfg.get("report_title",
                    f"Swapping the ego controller on {_numword(n_scen)} scenarios")
    backend = cfg.get("report_backend", "CARLA 0.9.16")
    clean_txt = " and ".join(f"<code>{_esc(s['name'])}</code>" for s in clean)

    p = cfg["idm_parameters"]
    idm_params = (f'v0={p["v0"]} m/s, T={p["T"]} s, a_max={p["a_max"]} m/s², '
                  f'b={p["b"]} m/s², δ={p["delta"]}, s0={p["s0"]} m')

    findings = []
    for sc in scen:
        s, i = by.get((sc, "scripted")), by.get((sc, "idm"))
        if not s or not i:
            continue
        if i["intended_rate"] >= 0.999:
            v = "reproduced the intended conflict in every run"
        elif i["intended_rate"] > 0:
            v = (f'reproduced the intended conflict in only '
                 f'{i["n_intended"]}/{i["n_runs"]} runs')
        elif i["n_proxy"] > 0:
            wrong = ", ".join(k for k, _ in i["partners"].most_common(2))
            v = (f'never reproduced the intended conflict; it was hit by '
                 f'<code>{_esc(wrong)}</code> instead of the scripted '
                 f'<code>{_esc(i["intended_partner"])}</code>')
        elif i["expect_collision"]:
            v = "produced no collision at all — the scripted conflict never developed"
        else:
            wrong = ", ".join(k for k, _ in i["partners"].most_common(2))
            v = f'was hit by <code>{_esc(wrong)}</code>, which the scenario avoids'
        findings.append(
            f'<div class="finding"><b>{_esc(sc)}</b> — IDM {v}. '
            f'Mean ego speed {i["mean_speed"]:.2f} m/s vs {s["mean_speed"]:.2f} m/s '
            f'scripted; baseline reproduced its intent in '
            f'{s["n_intended"]}/{s["n_runs"]} runs.</div>')

    towns_sub = f" / {towns_txt}" if towns_txt else ""
    scen_dir = cfg.get("scenario_dir", "scenarios/benchmark")
    n_scen_word = _numword(n_scen)
    n_crash_word = _numword(len(crash))
    has_have = "has" if len(clean) == 1 else "have"

    return f"""<title>Ego Policy Benchmark</title>
<style>{CSS}</style>
<div class="wrap">
<h1>{title}</h1>
<p class="sub">IDM against the compiled behaviour tree, measured by collision occurrence
over {n_runs} runs on {backend}{towns_sub}.</p>

<div class="card">
<b>What was run.</b> Each of the {n_scen_word} scenarios in <code>{scen_dir}</code> was
executed under two arms: once exactly as compiled, and once with the ego's actuation
handed to an Intelligent Driver Model through the <code>--ego-policy</code> entry
point. Everything else — NPC timelines, spawn geometry, <code>emit</code>/<code>wait</code>
events, the monitors that fire the adversarial triggers — is identical between arms;
only the ego's controller changes. IDM ran with library defaults ({idm_params}) in all
{n_scen_word} scenarios, so no per-scenario tuning is hiding in the results. {reps_txt}
</div>

<h2>The metric</h2>
<p>The measurement is <b>collision occurrence</b> on the ego's collision sensor, used as
a proxy for whether the scenario still executed what it was written to exercise. The
proxy is directional — it is not "fewer crashes is better":</p>
<ul>
<li>the {n_crash_word} crash scenarios have executed correctly when the ego <b>is</b>
hit; a clean run means the scripted conflict never developed;</li>
<li>{clean_txt} {has_have} executed correctly when the ego is <b>not</b> hit.</li>
</ul>
<p>So the raw proxy is <code>collision_occurred == expect_collision</code>. Running the
experiment showed that this alone is too weak: a collision with the <em>wrong</em>
vehicle scores as a pass. The report therefore also checks the striking vehicle's
<code>role_name</code> against the scenario's scripted antagonist. The headline number
below is that stricter <b>intended-conflict rate</b>; where the two diverge, the cell is
annotated <em>wrong-antagonist</em>.</p>

<h2>Result</h2>
{outcome_matrix(rows, cfg)}
<p class="key">Percentage of runs reproducing the scenario's <em>intended</em> outcome.
Scripted baseline: {base_full}/{len(base_rows)} scenarios at 100%. IDM:
{idm_full}/{len(idm_rows)}.</p>

{"".join(findings)}

{families_html}

<h2>Charts</h2>

<h3>Intended-conflict reproduction rate</h3>
{legend(meta)}
{grouped_bars(scen, s_intended, ylabel="% of runs", value_fmt="{:.0f}", vmax_override=100.0)}
<p class="key">The headline metric: did the scenario exercise what it was written to
exercise, with the antagonist it was written around?</p>

<h3>Raw collision-occurrence rate</h3>
{legend(meta)}
{grouped_bars(scen, s_coll, ylabel="% of runs", value_fmt="{:.0f}", vmax_override=100.0)}
<p class="key">The unrefined proxy. Compare against the chart above: where this one is
higher, the ego was hit by something other than the scripted antagonist — the proxy
passing for the wrong reason.</p>

<h3>Median time to first contact</h3>
{legend(meta)}
{grouped_bars(scen, s_first, ylabel="seconds", value_fmt="{:.1f}", na_label="no hit")}
<p class="key">Earlier contact under IDM is the signature of a collision that is not the
scripted one — typically being rear-ended long before reaching the junction.</p>

<h3>Mean ego speed</h3>
{legend(meta)}
{grouped_bars(scen, s_speed, ylabel="m/s", value_fmt="{:.1f}")}
<p class="key">IDM is slower everywhere: it opens a following gap the scripted ego never
kept, which is the mechanism behind most of the divergence.</p>

{sensitivity_section(sens)}

<h2>Per-cell measurements</h2>
{results_table(rows)}
<p class="key">Event counts are contact reports per physics substep, so they measure how
long bodies stayed in contact as much as how many distinct impacts occurred; contact
time, peak impulse and partner identity are the meaningful columns.</p>

<h2>Measurement notes</h2>
{notes_html}

<h2>Reproducing</h2>
{repro_html}
"""


def text_summary(rows) -> str:
    out = ["", f"{'scenario':<12}{'policy':<10}{'runs':>5}{'coll':>8}"
               f"{'intended':>10}{'first_t':>9}{'speed':>8}  hit_by"]
    out.append("-" * 92)
    for r in rows:
        first = f'{r["median_first_t"]:.2f}' if r["median_first_t"] is not None else "-"
        partners = ", ".join(f"{k}({v})" for k, v in r["partners"].most_common()) or "-"
        out.append(
            f'{r["scenario"]:<12}{r["policy"]:<10}{r["n_runs"]:>5}'
            f'{r["n_collision"]:>4}/{r["n_runs"]:<3}'
            f'{r["n_intended"]:>6}/{r["n_runs"]:<3}'
            f'{first:>9}{r["mean_speed"]:>8.2f}  {partners}')
    return "\n".join(out)


def text_sensitivity(sens) -> str:
    if not sens:
        return ""
    out = ["", "policy-parameter sensitivity (intended-outcome rate, median split)"]
    for entry in sens:
        out.append(f'  {entry["scenario"]}  (n={entry["n"]} parameter samples)')
        for r in entry["params"]:
            flag = "  <-- " if abs(r["delta"]) >= 0.25 else "      "
            out.append(f'    {r["param"]:<7} cut={r["cut"]:>6.2f}   '
                       f'low {r["rate_lo"]:>4.0%} (n={r["n_lo"]:>2})   '
                       f'high {r["rate_hi"]:>4.0%} (n={r["n_hi"]:>2})   '
                       f'delta {r["delta"]:>+5.0%}{flag}')
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=os.path.join(HERE, "results"))
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "benchmark.json"))
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "report", "index.html"))
    args = ap.parse_args()

    cfg, runs = load(args.results_dir, args.config)
    sens = parameter_sensitivity(cfg, runs)
    if not runs:
        print(f"no result JSON found under {args.results_dir}")
        return 1
    rows = evaluate(cfg, runs)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(build_html(cfg, rows, sens))
    serialisable = [{**r, "partners": dict(r["partners"])} for r in rows]
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(serialisable, fh, indent=2, sort_keys=True)
    print(text_summary(rows))
    print(text_sensitivity(sens))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
