"""
Analysis for the initialization x restart factorial.

The question the paper asks is an *interaction* question -- do restarts
substitute for initialization? -- so the analysis is built around it rather than
around a ranking table:

  1. Main effects. Friedman across the 7 initializers within each trigger.
  2. Interaction, stated so it can fail. If restarts substitute for
     initialization, the *spread* of performance across initializers must
     collapse once a trigger is switched on. We report that spread per trigger,
     and Kendall's tau between the initializer ranking under `none` and under
     each trigger. Substitution means small spread and a scrambled ranking;
     compounding means preserved spread and a stable ranking.
  3. Trigger attribution. Of the restarts GBNM's combined trigger fires, what
     fraction come from the degeneracy test rather than the size test? Luersen &
     Le Riche (2004) warned that a shrinking simplex gets tagged degenerate
     before it gets tagged small; this is that warning as a number.
  4. Conditioning. Where the Hadamard ratio ends up, by initializer and trigger.

Usage:
    python benchmarks/scripts/analyze_init_restart.py [--bbob]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load(path):
    d = json.loads(Path(path).read_text())
    return d["config"], d["records"]


def _median_by(records, keyfn, value="best"):
    acc = defaultdict(list)
    for r in records:
        v = r.get(value)
        if v is not None and np.isfinite(v):
            acc[keyfn(r)].append(v)
    return {k: float(np.median(v)) for k, v in acc.items()}


def _rank_within_problem(records, inits, trigger):
    """Mean rank of each initializer across problems, at a fixed trigger.

    Ranked within (problem, dim, seed) so that problems on wildly different
    scales cannot dominate, which is the whole reason for a rank statistic here.
    """
    cells = defaultdict(dict)
    for r in records:
        if r["trigger"] != trigger:
            continue
        cells[(r["problem"], r["dim"], r["seed"])][r["init"]] = r["best"]

    rows = [
        [c[i] for i in inits]
        for c in cells.values()
        if len(c) == len(inits) and all(np.isfinite(c[i]) for i in inits)
    ]
    if not rows:
        return None, None, 0
    arr = np.asarray(rows)
    ranks = np.apply_along_axis(stats.rankdata, 1, arr)
    mean_ranks = ranks.mean(axis=0)
    # Friedman across initializers: is any initializer distinguishable at all?
    try:
        _, p = stats.friedmanchisquare(*arr.T)
    except ValueError:
        p = float("nan")
    return mean_ranks, p, len(rows)


def report(config, records, label):
    inits = list(config["initializations"]) if "initializations" in config else sorted(
        {r["init"] for r in records}
    )
    triggers = list(config["triggers"]) if "triggers" in config else sorted(
        {r["trigger"] for r in records}
    )

    out = [f"# {label}", "", f"`{len(records)}` runs. Config: `{json.dumps(config)}`", ""]

    # ---------------------------------------------------------------- 1 & 2
    out += [
        "## Main effect of initialization, and whether the trigger changes it",
        "",
        "Mean rank across problems (1 = best of 7 initializers), ranked within",
        "(problem, dim, seed). `spread` is max-min mean rank: it is how much the",
        "choice of initializer still matters once that trigger is switched on.",
        "",
        "| trigger | " + " | ".join(inits) + " | spread | Friedman p | cells |",
        "|" + "---|" * (len(inits) + 4),
    ]
    ranks_by_trigger = {}
    for t in triggers:
        mr, p, n = _rank_within_problem(records, inits, t)
        if mr is None:
            out.append(f"| {t} | " + " | ".join(["-"] * len(inits)) + " | - | - | 0 |")
            continue
        ranks_by_trigger[t] = mr
        out.append(
            f"| {t} | "
            + " | ".join(f"{x:.2f}" for x in mr)
            + f" | **{mr.max() - mr.min():.2f}** | {p:.2g} | {n} |"
        )

    out += ["", "### Is the initializer ranking stable across triggers?", ""]
    if "none" in ranks_by_trigger:
        base = ranks_by_trigger["none"]
        out += [
            "Kendall tau between each trigger's initializer ranking and the",
            "single-descent (`none`) ranking. tau near 1 means the trigger left the",
            "initializer ordering intact (they compound); tau near 0 means the",
            "trigger scrambled it (they substitute).",
            "",
            "| trigger | Kendall tau vs none | p |",
            "|---|---|---|",
        ]
        for t, mr in ranks_by_trigger.items():
            if t == "none":
                continue
            tau, p = stats.kendalltau(base, mr)
            out.append(f"| {t} | {tau:+.3f} | {p:.3g} |")

    # ------------------------------------------------------------------- 3
    out += [
        "",
        "## GBNM trigger attribution",
        "",
        "Under the combined `gbnm` trigger (small OR degenerate), which test",
        "actually fires? Luersen & Le Riche report the tolerances are hard to tune",
        "because a shrinking simplex is tagged degenerate before it is tagged small.",
        "",
        "| dim | restarts/run | by degeneracy | by size | degeneracy share |",
        "|---|---|---|---|---|",
    ]
    for dim in sorted({r["dim"] for r in records}):
        # Filter on construction. Pooling `gaussian_best` with `oriented` mixes
        # two very different restart regimes -- the oriented pairing cascades at
        # 700-850 restarts per run -- and yields an attribution share that
        # matches neither and contradicts the paper.
        rs = [
            r for r in records
            if r["trigger"] == "gbnm" and r["dim"] == dim
            and r.get("construction", "gaussian_best") == "gaussian_best"
        ]
        if not rs:
            continue
        deg = sum(r["restarts_by_degeneracy"] for r in rs)
        size = sum(r["restarts_by_size"] for r in rs)
        tot = deg + size
        if not tot:
            out.append(f"| {dim} | 0.0 | 0 | 0 | - |")
            continue
        out.append(
            f"| {dim} | {tot / len(rs):.1f} | {deg} | {size} | "
            f"**{deg / tot * 100:.1f}%** |"
        )

    # ------------------------------------------------------------------- 4
    out += [
        "",
        "## Conditioning",
        "",
        "Median over runs of the per-run **final** Hadamard ratio |det L|/prod||e||.",
        "1.0 = orthogonal equal-length edges; 0 = collapsed to a subspace.",
        "",
        "| init | " + " | ".join(triggers) + " |",
        "|" + "---|" * (len(triggers) + 1),
    ]
    med = _median_by(records, lambda r: (r["init"], r["trigger"]), "hadamard_final")
    for i in inits:
        out.append(
            f"| {i} | "
            + " | ".join(
                f"{med[(i, t)]:.2e}" if (i, t) in med else "-" for t in triggers
            )
            + " |"
        )

    out += [
        "",
        "### Conditioning over the whole run",
        "",
        "Median Hadamard ratio across the trace, and the fraction of iterations",
        "spent below 1e-6 -- i.e. inside GBNM's default degeneracy tolerance.",
        "",
        "| init | median Hadamard | frac. iters below 1e-6 |",
        "|---|---|---|",
    ]
    med_all = _median_by(records, lambda r: r["init"], "hadamard_median")
    frac = _median_by(records, lambda r: r["init"], "frac_iters_below_1e6")
    for i in inits:
        out.append(
            f"| {i} | {med_all.get(i, float('nan')):.3f} | "
            f"{frac.get(i, float('nan')):.3f} |"
        )

    # ------------------------------------------------------------------ 5
    # Mediation. The paper's claim is not that initializers differ -- everyone
    # knows that -- but that they differ *because* of conditioning. If that is
    # right, a run's conditioning should predict its outcome within a problem,
    # where the objective scale is held fixed.
    out += [
        "",
        "## Does conditioning explain the initialization effect?",
        "",
        "Spearman rho between a run's median Hadamard ratio and its final",
        "objective value, computed within each (problem, dim, trigger) cell and",
        "then pooled. Negative rho means better-conditioned runs ended lower,",
        "i.e. conditioning mediates the effect. A rho near zero would say",
        "conditioning is a bystander and the mechanism story is wrong.",
        "",
        "| dim | median rho | fraction of cells with rho < 0 | cells |",
        "|---|---|---|---|",
    ]
    for dim in sorted({r["dim"] for r in records}):
        rhos = []
        cells = defaultdict(list)
        for r in records:
            if r["dim"] != dim or r["hadamard_median"] is None:
                continue
            if np.isfinite(r["best"]) and np.isfinite(r["hadamard_median"]):
                # Key on the *function*, not the instance: BBOB runs one seed
                # per problem, so keying on the problem id leaves 7 points per
                # cell and every cell gets skipped. Instances of one function
                # share a scale, so pooling them is safe.
                key = (r.get("function_id", r["problem"]), r["trigger"])
                cells[key].append((r["hadamard_median"], r["best"]))
        for pairs in cells.values():
            if len(pairs) < 10:
                continue
            a, b = zip(*pairs)
            if len(set(b)) < 2 or len(set(a)) < 2:
                continue
            rho, _ = stats.spearmanr(a, b)
            if np.isfinite(rho):
                rhos.append(rho)
        if rhos:
            out.append(
                f"| {dim} | {np.median(rhos):+.3f} | "
                f"{np.mean([r < 0 for r in rhos]):.2f} | {len(rhos)} |"
            )

    # ------------------------------------------------------------------ 6
    # Factor C. Without this the trigger comparison is confounded: Kelley's arm
    # re-seeds locally (oriented) while every other trigger re-seeds globally
    # (gaussian_best), and on a multimodal suite that difference alone could
    # produce the whole effect.
    if any("construction" in r for r in records) and any(
        "solved_tau0.001" in r for r in records
    ):
        constructions = sorted({r["construction"] for r in records})
        out += [
            "",
            "## Trigger x construction (the confound, resolved)",
            "",
            "Fraction solved at tau = 1e-7. Reading *down* a column compares",
            "triggers at fixed construction; reading *across* a row isolates the",
            "re-seeding. `progress` (Kelley) exists only in its published pairing.",
            "",
            "| trigger | " + " | ".join(constructions) + " | delta |",
            "|" + "---|" * (len(constructions) + 2),
        ]
        for t in triggers:
            cells, vals = [], []
            for c in constructions:
                rs = [
                    r for r in records
                    if r["trigger"] == t and r.get("construction") == c
                ]
                if rs:
                    v = 100 * np.mean([r["solved_tau1e-07"] for r in rs])
                    vals.append(v)
                    cells.append(f"{v:.1f}%")
                else:
                    cells.append("-")
            delta = f"{vals[1] - vals[0]:+.1f}" if len(vals) == 2 else "-"
            out.append(f"| {t} | " + " | ".join(cells) + f" | {delta} |")

        out += [
            "",
            "Restarts per run, same layout:",
            "",
            "| trigger | " + " | ".join(constructions) + " |",
            "|" + "---|" * (len(constructions) + 1),
        ]
        for t in triggers:
            cells = []
            for c in constructions:
                rs = [
                    r for r in records
                    if r["trigger"] == t and r.get("construction") == c
                ]
                cells.append(
                    f"{np.mean([r['restarts'] for r in rs]):.1f}" if rs else "-"
                )
            out.append(f"| {t} | " + " | ".join(cells) + " |")

    # BBOB-only: Moré-Wild solved fractions, by function group.
    if any("group" in r for r in records):
        out += ["", "## Fraction solved (More-Wild, tau = 1e-3), by BBOB group", ""]
        groups = sorted({r["group"] for r in records})
        out += [
            "| trigger | " + " | ".join(g.replace("_", " ") for g in groups) + " |",
            "|" + "---|" * (len(groups) + 1),
        ]
        for t in triggers:
            cells = []
            for g in groups:
                rs = [
                    r for r in records if r["trigger"] == t and r["group"] == g
                ]
                cells.append(
                    f"{100 * np.mean([r['solved_tau0.001'] for r in rs]):.0f}%"
                    if rs
                    else "-"
                )
            out.append(f"| {t} | " + " | ".join(cells) + " |")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analytic", default=str(RESULTS / "init_restart_factorial.json"))
    # The merged set (n = 2/5/10/20). `bbob_factorial.json` is the pre-n=20
    # run and is kept only as an input to the merge; reading it here silently
    # regenerates the three-dimension results and contradicts the paper.
    ap.add_argument("--bbob", default=str(RESULTS / "bbob_factorial_merged.json"))
    ap.add_argument("--out", default=str(RESULTS / "INIT_RESTART_ANALYSIS.md"))
    args = ap.parse_args()

    sections = []
    for path, label in ((args.analytic, "Analytic suite"), (args.bbob, "COCO/BBOB suite")):
        if Path(path).exists():
            sections.append(report(*load(path), label))
        else:
            sections.append(f"# {label}\n\n_missing: {path}_")

    text = "\n\n---\n\n".join(sections)
    Path(args.out).write_text(text)
    print(text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
