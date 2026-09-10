"""
The initialization x restart factorial on the COCO/BBOB noiseless suite.

BBOB is not optional for this paper: both antecedents run on it. Wessing
(Optimization Letters 13(4), 2019) and Takenaga, Ozaki & Onishi (Optimization
Letters 17(2), 2023) both benchmark Nelder-Mead initialization on the 24
noiseless functions, so a study claiming to extend them has to be measured on
the same instrument.

BBOB f-values carry a per-instance offset and this COCO build does not expose
f_opt, so the reference value is `f_L` -- the best value any arm reached on that
problem -- which is the Moré-Wild (SIAM J. Optim. 20(1):172-191, 2009)
convention anyway. The convergence test is theirs:

    f(x0) - f(x)  >=  (1 - tau) * (f(x0) - f_L)

reported at tau in {1e-1, 1e-3, 1e-5, 1e-7}, with the budget counted in simplex
gradients (n+1 evaluations), which is what makes the profiles dimension-fair.
COCO's own `final_target_hit` is recorded alongside as a cross-check.

Usage:
    python benchmarks/scripts/run_bbob_factorial.py --quick
    python benchmarks/scripts/run_bbob_factorial.py --dims 2 5 10 --instances 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cocoex  # noqa: E402

from run_init_restart_factorial import (  # noqa: E402
    CONSTRUCTIONS,
    INITIALIZATIONS,
    TRIGGERS,
    run_one,
)


class BBOBProblem:
    """Adapts a cocoex problem to the attributes `run_one` needs."""

    def __init__(self, coco_problem):
        self._p = coco_problem
        self.name = coco_problem.id
        self.dim = coco_problem.number_of_variables
        self.bounds = (
            np.asarray(coco_problem.lower_bounds, dtype=float),
            np.asarray(coco_problem.upper_bounds, dtype=float),
        )
        self.function_id = coco_problem.id_function
        self.instance = coco_problem.id_instance

    def objective(self, x):
        return float(self._p(np.asarray(x, dtype=float)))

    @property
    def target_hit(self):
        return bool(self._p.final_target_hit)


#: BBOB's own grouping. The multimodal groups are the ones that let this paper
#: make a scoped modality claim instead of an unsupported one.
BBOB_GROUPS = {
    **{f: "separable" for f in range(1, 6)},
    **{f: "low_moderate_conditioning" for f in range(6, 10)},
    **{f: "high_conditioning_unimodal" for f in range(10, 15)},
    **{f: "multimodal_global_structure" for f in range(15, 20)},
    **{f: "multimodal_weak_structure" for f in range(20, 25)},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--functions", default="1-24")
    ap.add_argument("--budget-per-dim", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="f1-24, dim 2, 1 instance")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "results" / "bbob_factorial.json"),
    )
    args = ap.parse_args()

    dims = [2] if args.quick else args.dims
    instances = 1 if args.quick else args.instances

    suite = cocoex.Suite(
        "bbob",
        "",
        f"function_indices:{args.functions} "
        f"dimensions:{','.join(map(str, dims))} "
        f"instance_indices:1-{instances}",
    )
    # `none` has no restart and `progress` is Kelley's own detector+construction
    # pairing, so neither takes a construction level; the other four are crossed
    # with both. That keeps the construction axis honest without paying for arms
    # that would be duplicates.
    arms = []
    for init in INITIALIZATIONS:
        for trig in TRIGGERS:
            if trig in ("none", "progress"):
                arms.append((init, trig, "gaussian_best"))
            else:
                arms.extend((init, trig, c) for c in CONSTRUCTIONS)
    total = len(suite) * len(arms)
    print(
        f"{total} runs: {len(suite)} BBOB problems x {len(arms)} arms "
        f"({len(INITIALIZATIONS)} inits x {len(TRIGGERS)} triggers x construction)",
        flush=True,
    )

    records, t0 = [], time.time()
    # Iterate the suite directly: cocoex frees a problem as soon as the iterator
    # moves past it, so a pre-materialised list of wrappers is all dangling
    # handles except the last.
    for i, coco_problem in enumerate(suite):
        problem = BBOBProblem(coco_problem)
        budget = args.budget_per_dim * problem.dim
        # f(x0) for the Moré-Wild ratio: the centre of the box, the same start
        # every structured initializer is built around.
        f_x0 = problem.objective(np.zeros(problem.dim))
        for init, trigger, construction in arms:
            rec = run_one(problem, init, trigger, 101, budget, construction)
            rec.update(
                function_id=problem.function_id,
                instance=problem.instance,
                group=BBOB_GROUPS[problem.function_id],
                f_x0=f_x0,
                coco_target_hit=problem.target_hit,
                budget=budget,
            )
            records.append(rec)
        if (i + 1) % 20 == 0:
            done = (i + 1) * len(arms)
            rate = done / (time.time() - t0)
            print(
                f"  {done}/{total}  {rate:.1f} runs/s  "
                f"eta {(total - done) / rate / 60:.1f} min",
                flush=True,
            )

    # f_L per problem: best value any arm reached. Moré-Wild's reference when the
    # true optimum is unavailable -- which it is here, by construction.
    best_by_problem = {}
    for r in records:
        key = r["problem"]
        best_by_problem[key] = min(best_by_problem.get(key, np.inf), r["best"])
    for r in records:
        f_L = best_by_problem[r["problem"]]
        span = r["f_x0"] - f_L
        r["f_L"] = f_L
        # Moré-Wild: solved at tau iff f(x0) - f(x) >= (1 - tau)(f(x0) - f_L).
        # A degenerate span (every arm stuck at f(x0)) counts as solved only if
        # the arm actually sits at f_L, never by dividing by zero.
        for tau in (1e-1, 1e-3, 1e-5, 1e-7):
            r[f"solved_tau{tau:g}"] = bool(
                (r["f_x0"] - r["best"]) >= (1 - tau) * span
                if span > 0
                else r["best"] <= f_L
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "suite": "bbob",
                    "functions": args.functions,
                    "dims": dims,
                    "instances": instances,
                    "budget_per_dim": args.budget_per_dim,
                    "seed": 101,
                    "constructions": list(CONSTRUCTIONS),
                    "reference": "f_L = best over all arms (More-Wild convention)",
                },
                "records": records,
            },
            indent=1,
        )
    )
    print(f"wrote {out}  ({len(records)} records, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
