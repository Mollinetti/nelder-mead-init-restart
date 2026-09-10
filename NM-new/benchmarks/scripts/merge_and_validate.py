"""
Merge the n=20 BBOB run into the main record set and re-check every claim the
paper makes against the extended data.

Merging is safe without recomputing `f_L`: it is defined per *problem*, and no
BBOB problem id appears in both files (they differ in dimension), so each
problem's reference is still the best over the same 70 arms.

The validation half exists because adding a dimension can break the paper in
ways a spot check would miss. Two of the results are monotone trends over
n = 2, 5, 10; a fourth point can flatten or reverse them. This script states each
claim, recomputes it, and prints HOLDS / BROKEN so the decision to keep or
rewrite is made on evidence rather than on the numbers looking familiar.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
INITS = [
    "pfeffer", "spendley_full", "spendley_10pct", "spendley_2pct",
    "uniform", "gaussian", "positive_basis",
]
TRIGGERS = ["none", "progress", "conditioning", "gbnm", "size", "flat"]


def construction_of(trigger):
    return "oriented" if trigger == "progress" else "gaussian_best"


def merge(base_path, extra_path, out_path):
    base = json.loads(Path(base_path).read_text())
    extra = json.loads(Path(extra_path).read_text())
    ids_base = {r["problem"] for r in base["records"]}
    ids_extra = {r["problem"] for r in extra["records"]}
    overlap = ids_base & ids_extra
    if overlap:
        raise SystemExit(
            f"refusing to merge: {len(overlap)} problems appear in both files, so "
            "f_L would be inconsistent. Recompute instead."
        )
    merged = dict(base)
    merged["records"] = base["records"] + extra["records"]
    merged["config"] = {
        # Basenames, not full paths: this config is echoed into a committed
        # analysis artifact, and absolute paths leak the author's filesystem.
        "merged_from": [Path(base_path).name, Path(extra_path).name],
        "base": base["config"],
        "extra": extra["config"],
    }
    Path(out_path).write_text(json.dumps(merged, indent=1))
    dims = sorted({r["dim"] for r in merged["records"]})
    print(
        f"merged {len(base['records'])} + {len(extra['records'])} = "
        f"{len(merged['records'])} records, dims {dims}"
    )
    return merged["records"]


def solved(R, trigger, construction, tau=1e-7, dim=None):
    rs = [
        r for r in R
        if r["trigger"] == trigger
        and r.get("construction") == construction
        and (dim is None or r["dim"] == dim)
    ]
    return 100 * np.mean([r[f"solved_tau{tau:g}"] for r in rs]) if rs else float("nan")


def mean_ranks(R, trigger, construction, dim=None):
    cells = defaultdict(dict)
    for r in R:
        if (
            r["trigger"] == trigger
            and r.get("construction") == construction
            and (dim is None or r["dim"] == dim)
        ):
            cells[r["problem"]][r["init"]] = r["best"]
    rows = [[c[i] for i in INITS] for c in cells.values() if len(c) == len(INITS)]
    if not rows:
        return None
    return np.apply_along_axis(stats.rankdata, 1, np.asarray(rows)).mean(0)


def spread(R, trigger, construction, dim=None):
    mr = mean_ranks(R, trigger, construction, dim)
    return float(mr.max() - mr.min()) if mr is not None else float("nan")


def degeneracy_share(R, dim):
    rs = [
        r for r in R
        if r["trigger"] == "gbnm" and r.get("construction") == "gaussian_best"
        and r["dim"] == dim
    ]
    d = sum(r["restarts_by_degeneracy"] for r in rs)
    s = sum(r["restarts_by_size"] for r in rs)
    return 100 * d / (d + s) if (d + s) else float("nan")


def main():
    base = RESULTS / "bbob_factorial.json"
    extra = RESULTS / "bbob_factorial_d20.json"
    out = RESULTS / "bbob_factorial_merged.json"
    R = merge(base, extra, out)
    dims = sorted({r["dim"] for r in R})

    verdicts = []

    def claim(name, holds, detail):
        verdicts.append(holds)
        print(f"\n[{'HOLDS ' if holds else 'BROKEN'}] {name}\n    {detail}")

    # ---- Claim 1: trivial triggers beat principled ones, at both constructions
    lines = []
    all_ok = True
    for c in ("gaussian_best", "oriented"):
        triv = [solved(R, t, c) for t in ("size", "flat")]
        prin = [
            solved(R, t, c)
            for t in ("conditioning", "gbnm")
            if not np.isnan(solved(R, t, c))
        ]
        ok = min(triv) > max(prin)
        all_ok &= ok
        lines.append(
            f"{c}: trivial {min(triv):.1f}-{max(triv):.1f}% vs "
            f"principled {min(prin):.1f}-{max(prin):.1f}%  "
            f"gap {min(triv) - max(prin):+.1f}"
        )
    claim("Trivial triggers beat principled detectors", all_ok, "; ".join(lines))

    # ---- Claim 2: this also holds at the NEW dimension alone
    lines, ok20 = [], True
    for c in ("gaussian_best", "oriented"):
        triv = [solved(R, t, c, dim=20) for t in ("size", "flat")]
        prin = [solved(R, t, c, dim=20) for t in ("conditioning", "gbnm")]
        ok = min(triv) > max(prin)
        ok20 &= ok
        lines.append(
            f"{c}: trivial {min(triv):.1f}-{max(triv):.1f}% vs "
            f"principled {min(prin):.1f}-{max(prin):.1f}%"
        )
    claim("...and holds at n=20 alone", ok20, "; ".join(lines))

    # ---- Claim 3: Kelley is no better than never restarting
    k, n_ = solved(R, "progress", "oriented"), solved(R, "none", "gaussian_best")
    claim(
        "Kelley indistinguishable from no restart",
        abs(k - n_) < 3.0,
        f"progress {k:.1f}% vs none {n_:.1f}%  (delta {k - n_:+.1f})",
    )

    # ---- Claim 4: degeneracy share rises monotonically with dimension
    shares = [degeneracy_share(R, d) for d in dims]
    mono = all(b >= a - 0.5 for a, b in zip(shares, shares[1:]))
    claim(
        "Degeneracy share of GBNM firings rises with dimension",
        mono,
        "  ".join(f"n={d}: {s:.1f}%" for d, s in zip(dims, shares)),
    )

    # ---- Claim 5: adding degeneracy to a size test makes it worse
    g, s_ = solved(R, "gbnm", "gaussian_best"), solved(R, "size", "gaussian_best")
    claim(
        "gbnm (size OR degeneracy) scores below size alone",
        g < s_,
        f"gbnm {g:.1f}% vs size {s_:.1f}%  (delta {g - s_:+.1f})",
    )

    # ---- Claim 6: restarts absorb the initializer spread, less so as n grows
    rows, absorbed = [], []
    for d in dims:
        none_sp = spread(R, "none", "gaussian_best", d)
        best_sp = min(
            spread(R, t, "gaussian_best", d)
            for t in ("size", "flat", "gbnm", "conditioning")
        )
        a = 100 * (1 - best_sp / none_sp)
        absorbed.append(a)
        rows.append(f"n={d}: {none_sp:.2f}->{best_sp:.2f} ({a:.0f}%)")
    decreasing = all(b <= a + 3.0 for a, b in zip(absorbed, absorbed[1:]))
    claim(
        "Restarts absorb the initializer spread, decreasingly with dimension",
        min(absorbed) > 40 and decreasing,
        "  ".join(rows),
    )

    # ---- Claim 7: pfeffer worst under single descent, unremarkable with restarts
    mr_none = mean_ranks(R, "none", "gaussian_best")
    pf_none = mr_none[INITS.index("pfeffer")]
    pf_trig = [
        mean_ranks(R, t, "gaussian_best")[INITS.index("pfeffer")]
        for t in ("size", "flat", "gbnm", "conditioning")
    ]
    claim(
        "fminsearch init is worst under single descent, mid-pack with restarts",
        pf_none == mr_none.max() and max(pf_trig) < pf_none - 0.5,
        f"none {pf_none:.2f} (worst of 7: {pf_none == mr_none.max()}); "
        f"with triggers {min(pf_trig):.2f}-{max(pf_trig):.2f}",
    )

    # ---- Claim 8: the large regular simplex is the best initializer everywhere
    best_each = [
        INITS[int(np.argmin(mean_ranks(R, t, construction_of(t))))]
        for t in TRIGGERS
    ]
    claim(
        "spendley_full is the best initializer under every trigger",
        all(b == "spendley_full" for b in best_each),
        ", ".join(f"{t}:{b}" for t, b in zip(TRIGGERS, best_each)),
    )

    print(f"\n{'=' * 70}")
    print(f"{sum(verdicts)}/{len(verdicts)} claims hold on the extended data")
    print(f"merged records written to {out}")
    return 0 if all(verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
