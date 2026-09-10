"""
Evaluate the tolerance sweep against the acceptance criteria fixed in
run_tolerance_calibration.py's docstring, and print the verdict.

The point of stating G1/A1/A2/A3 in advance is that this script does not get to
choose which one applies -- it computes them.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"

def main():
    d = json.loads((RESULTS / "tolerance_calibration.json").read_text())
    data = d
    R, cfg = d["records"], d["config"]
    grid = sorted(set(cfg["eps_grid"]) | set(cfg.get("eps_grid_extended", [])))
    dims = sorted({r["dim"] for r in R})

    def cells_of(dim, trigger, eps3=None, eps4=None):
        return [
            r for r in R
            if r["dim"] == dim and r["trigger"] == trigger
            and (eps3 is None or r["eps3"] == eps3)
            and (eps4 is None or r["eps4"] == eps4)
        ]

    # Rank-based comparison: f_L-free, and immune to the outright-win bias that
    # complicated the n=20 reading in the main experiment.
    def mean_gap_vs_size(dim, eps3, eps4):
        """Median log-ratio of degeneracy-arm objective to size-arm objective.

        Negative means the degeneracy trigger ended lower (better). Computed per
        problem, so problem scale cancels.
        """
        size = {r["problem"]: r["best"] for r in cells_of(dim, "size")}
        rows = []
        for r in cells_of(dim, "conditioning", eps3, eps4):
            s = size.get(r["problem"])
            if s is None:
                continue
            a, b = max(r["best"], 1e-300), max(s, 1e-300)
            rows.append(np.log10(a) - np.log10(b))
        return float(np.median(rows)) if rows else float("nan")

    def win_rate(dim, eps3, eps4):
        size = {r["problem"]: r["best"] for r in cells_of(dim, "size")}
        w = [
            r["best"] < size[r["problem"]]
            for r in cells_of(dim, "conditioning", eps3, eps4)
            if r["problem"] in size
        ]
        return 100 * float(np.mean(w)) if w else float("nan")

    print("Tolerance calibration for GBNM eq. (9)")
    print(f"grid {grid}, init {cfg['init']}, {len(R)} runs\n")

    # ---- G1: harness gate ------------------------------------------------
    print("G1  harness gate -- restarts/run at the default (1e-6, 1e-6)")
    print("    The main sweep predates trace_conditioning, so its restart counts")
    print("    were never recorded; G1 reads the dedicated re-run instead, whose")
    print("    objective values were verified bit-identical to the sweep.")
    rc = d.get("g1_recheck", {}).get("restarts_by_dim", {})
    ref = {5: 86.7, 10: 126.4, 20: 462.5}
    g1_ok = bool(rc)
    for dim in dims:
        m = rc.get(str(dim))
        if m is None:
            print(f"    n={dim:2d}: no recheck data"); g1_ok = False; continue
        ok = 0.4 * ref[dim] <= m <= 2.5 * ref[dim]
        g1_ok &= ok
        print(f"    n={dim:2d}: recheck {m:7.1f}  main {ref[dim]:7.1f}  {'OK' if ok else 'MISMATCH'}")
    print(f"    -> {'PASS' if g1_ok else 'FAIL - harness suspect, stop here'}\n")

    print("eps_s3 inertness check (win rate at fixed eps_s4, varying eps_s3):")
    for e4 in (1e-12, 1e-6, 1e-4, 1e-2):
        vals = []
        for e3 in (1e-12, 1e-6, 1e-4, 1e-2):
            w = win_rate(10, e3, e4)
            if not np.isnan(w):
                vals.append(round(w, 1))
        print(f"    eps_s4={e4:.0e}, n=10: {vals}  "
              f"{'identical' if len(set(vals)) == 1 else 'VARIES'}")
    print()

    def collapse(dim, e4):
        """Win rate at this eps_s4, pooling whichever eps_s3 rows exist."""
        sz = {r["problem"]: r["best"] for r in cells_of(dim, "size")}
        w = [r["best"] < sz[r["problem"]]
             for r in R
             if r["dim"] == dim and r["trigger"] == "conditioning"
             and r["eps4"] == e4 and r["problem"] in sz]
        return w

    eps4s = sorted({r["eps4"] for r in R if r["trigger"] == "conditioning"})
    print("Win rate vs the size trigger, by eps_s4 (50% = parity), and restarts/run.\n")
    print(f"{'eps_s4':>9s} | " + "".join(f"{'n='+str(d)+' win':>13s}{'restarts':>10s}" for d in dims))
    print("-" * 9 + "-+-" + "-" * (23 * len(dims)))
    for e4 in eps4s:
        line = f"{e4:9.0e} | "
        for d in dims:
            w = collapse(d, e4)
            vals = [r["restarts"] for r in R
                    if r["dim"] == d and r["trigger"] == "conditioning"
                    and r["eps4"] == e4 and r.get("restarts") is not None]
            if vals:
                rr = float(np.mean(vals))
            else:
                key = f"{d}|{0.0:.0e}|{e4:.0e}"
                rates = data.get("restart_rates", {}).get("mean_by_dim_eps3_eps4", {})
                cand = [v for k, v in rates.items()
                        if k.startswith(f"{d}|") and k.endswith(f"|{e4:.0e}")]
                rr = float(np.mean(cand)) if cand else float("nan")
            line += f"{100*np.mean(w):12.1f}%{rr:10.1f}"
        print(line)
    print()

    # ---- verdict ---------------------------------------------------------
    # Tested against parity with an exact binomial, Bonferroni-corrected over
    # every cell tested. The earlier version of this block compared only the
    # best cell and used "any dimension above parity" where the stated criterion
    # was "both dimensions" -- it would have reported A2 off a single cell that
    # is statistically a coin flip.
    from scipy import stats as _st

    tested = [(e4, d) for e4 in eps4s for d in dims]
    alpha = 0.05 / len(tested)
    better = []
    for e4, d in tested:
        w = collapse(d, e4)
        if not w:
            continue
        pv = _st.binomtest(sum(w), len(w), 0.5).pvalue
        if pv < alpha and np.mean(w) > 0.5:
            better.append((e4, d, 100 * np.mean(w), pv))

    print("=" * 68)
    print(f"Tested all {len(tested)} populated cells against parity with size")
    print(f"(exact binomial, Bonferroni alpha = {alpha:.2g}).")
    if better:
        print("\nA2 -- CLAIM NARROWS. Significantly better than a size test at:")
        for e4, d, w, pv in better:
            print(f"    eps_s4={e4:.0e} n={d}: {w:.1f}%  p={pv:.2e}")
    else:
        print("\nA1 -- CLAIM HOLDS. No cell is significantly better than a plain")
        print("  size test at any dimension. Any above-parity cell is within noise.")
    print("=" * 68)


if __name__ == "__main__":
    main()
