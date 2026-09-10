"""
The inferential statistics the paper reports: corrected p-values and effect
sizes for every pairwise claim it makes.

Two gaps this closes. First, the trigger comparisons were read one at a time --
7 pairs x 4 dimensions -- with no correction, so at alpha=0.05 roughly 1.4 false
positives were expected under the global null. Second, nothing but p-values was
reported, and at these sample sizes a p-value says almost nothing about size.

Test: exact two-sided sign test on paired per-problem outcomes, Holm-corrected
across the whole family. Effect size: the paired win rate P(arm A ends lower |
same problem), which is the paired analogue of Vargha-Delaney A12 and is read on
the same 0.56/0.64/0.71 scale.

Both choices are deliberate. BBOB objective values carry per-instance offsets and
span many orders of magnitude, so a Wilcoxon signed-rank on the raw values is
dominated by whichever problems happen to have large absolute f -- it measures
the arithmetic size of differences on an arbitrary scale. The sign test asks only
which arm won on each problem, which is scale-free and is the same quantity the
rest of the paper reports. The unpaired A12 is wrong here for the same reason:
computed across all cross-pairs it compares arm A on one problem against arm B on
another, and problem difficulty swamps the arm effect (it returned 0.46-0.55 --
"negligible" -- for comparisons whose paired win rates run from 20% to 97%).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
DIMS = (2, 5, 10, 20)


def construction_of(t):
    return "oriented" if t == "progress" else "gaussian_best"


def paired_binary(records, a, b, dim, key="solved_tau1e-07"):
    """Per-problem solved/not-solved, a problem counted solved when a majority of
    the seven initializers solved it. Collapsing to the problem keeps the unit of
    analysis independent; the seven runs on one problem are not."""
    A, B = {}, {}
    for r in records:
        if r["dim"] != dim:
            continue
        if r["trigger"] == a and r["construction"] == construction_of(a):
            A.setdefault(r["problem"], []).append(r[key])
        if r["trigger"] == b and r["construction"] == construction_of(b):
            B.setdefault(r["problem"], []).append(r[key])
    common = sorted(set(A) & set(B))
    return (np.array([np.mean(A[p]) > 0.5 for p in common]),
            np.array([np.mean(B[p]) > 0.5 for p in common]))


def paired(records, a, b, dim):
    A = {r["problem"]: r["best"] for r in records
         if r["trigger"] == a and r["construction"] == construction_of(a) and r["dim"] == dim}
    B = {r["problem"]: r["best"] for r in records
         if r["trigger"] == b and r["construction"] == construction_of(b) and r["dim"] == dim}
    common = sorted(set(A) & set(B))
    return np.array([A[p] for p in common]), np.array([B[p] for p in common])


def paired_win_rate(x, y):
    """P(x ends lower | same problem), ties split. The paired analogue of A12."""
    wins = np.sum(x < y) + 0.5 * np.sum(x == y)
    return float(wins / len(x))


def magnitude(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def main():
    R = json.loads((RESULTS / "bbob_factorial_merged.json").read_text())["records"]
    R = [r for r in R if "construction" in r]

    # Every pairwise claim the paper makes, stated once so the family is explicit.
    pairs = [
        ("gbnm", "none"), ("conditioning", "none"), ("progress", "none"),
        ("size", "none"), ("flat", "none"),
        ("size", "gbnm"), ("flat", "gbnm"), ("progress", "size"),
    ]
    tests = []
    for a, b in pairs:
        for dim in DIMS:
            x, y = paired(R, a, b, dim)
            if len(x) < 10:
                continue
            # Aggregate the 7 initializers per problem before pairing, so the
            # unit of analysis is the problem, not the run.
            wins = int(np.sum(x < y))
            trials = int(np.sum(x != y))
            p = stats.binomtest(wins, trials, 0.5).pvalue if trials else 1.0
            tests.append({
                "a": a, "b": b, "dim": dim, "n": len(x), "p_raw": float(p),
                "A12": paired_win_rate(x, y),
            })

    # Holm across the whole family of comparisons the paper reads.
    order = np.argsort([t["p_raw"] for t in tests])
    m = len(tests)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * tests[idx]["p_raw"])
        running = max(running, adj)  # Holm is monotone non-decreasing
        tests[idx]["p_holm"] = running

    print("Exact sign test on paired per-problem outcomes, Holm-corrected across")
    print(f"the full family of {m} comparisons. `win` = P(first arm ends lower).\n")
    print(f"{'comparison':24s}{'n':>4s}{'N':>5s}{'p (raw)':>11s}{'p (Holm)':>11s}"
          f"{'win':>7s}  {'effect':<11s}verdict")
    print("-" * 96)
    for t in sorted(tests, key=lambda t: (t["a"], t["b"], t["dim"])):
        sig = t["p_holm"] < 0.05
        direction = "better" if t["A12"] > 0.5 else "worse"
        verdict = f"{t['a']} {direction}" if sig else "n.s."
        print(f"{t['a'] + ' vs ' + t['b']:24s}{t['dim']:>4d}{t['n']:>5d}"
              f"{t['p_raw']:>11.2e}{t['p_holm']:>11.2e}{t['A12']:>7.3f}  "
              f"{magnitude(t['A12']):<11s}{verdict}")

    survived = [t for t in tests if t["p_holm"] < 0.05]
    print(f"\n{len(survived)}/{m} comparisons survive Holm correction at alpha=0.05.")
    lost = [t for t in tests if t["p_raw"] < 0.05 <= t["p_holm"]]
    if lost:
        print("Lost to correction (significant raw, not after Holm):")
        for t in lost:
            print(f"  {t['a']} vs {t['b']} at n={t['dim']}: "
                  f"p {t['p_raw']:.3f} -> {t['p_holm']:.3f}, win={t['A12']:.3f}")
    else:
        print("No comparison was lost to the correction.")

    # ---- trend tests -----------------------------------------------------
    # The paper's dimensional claims are claims about a TREND, and four separate
    # point comparisons is both the wrong test and a needlessly weak one: it
    # spends multiplicity budget to answer a question nobody asked. Cochran-
    # Armitage on the share of discordant pairs, with log2(n) as the dose, tests
    # what is actually claimed.
    print("\n" + "=" * 96)
    print("Trend across dimension (Cochran-Armitage, dose = log2 n), Holm x3.\n")
    dose = np.log2(np.array(DIMS, dtype=float))
    for a in ("progress", "gbnm", "conditioning"):
        k, N = [], []
        for dim in DIMS:
            x, y = paired_binary(R, a, "none", dim)
            ka = int(np.sum(x & ~y)); kb = int(np.sum(~x & y))
            k.append(ka); N.append(ka + kb)
        k = np.array(k); N = np.array(N)
        pbar = k.sum() / N.sum(); dbar = (N * dose).sum() / N.sum()
        z = (((k - N * pbar) * (dose - dbar)).sum()
             / np.sqrt(pbar * (1 - pbar) * (N * (dose - dbar) ** 2).sum()))
        pv = 2 * (1 - stats.norm.cdf(abs(z)))
        shares = "  ".join(f"n={d}:{ki/Ni:.2f}" for d, ki, Ni in zip(DIMS, k, N))
        print(f"  {a + ' vs none':22s}{shares}   z={z:+.2f}  p={pv:.2e}  "
              f"Holm={min(1.0, 3 * pv):.2e}")

    out = RESULTS / "PAPER_STATISTICS.md"
    out.write_text("# Corrected statistics (sign test, Holm)\n\n```\n" + "\n".join(
        f"{t['a']} vs {t['b']} n={t['dim']}: p_raw={t['p_raw']:.3e} "
        f"p_holm={t['p_holm']:.3e} win={t['A12']:.3f} ({magnitude(t['A12'])})"
        for t in sorted(tests, key=lambda t: (t["a"], t["b"], t["dim"]))) + "\n```\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
