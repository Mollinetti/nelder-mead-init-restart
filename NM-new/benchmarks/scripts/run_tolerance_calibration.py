"""
Calibrate the GBNM degeneracy tolerances against dimension.

The paper claims the degeneracy trigger of Luersen & Le Riche (2004) is not
merely unhelpful but harmful at moderate dimension. That claim is only worth
making if the failure is a property of the *criterion* rather than of the
tolerances we happened to run it at -- especially since its authors say the
tolerances "may be difficult to tune". This sweep decides which it is.

Eq. (9) declares a simplex degenerate when

    min_k||e^k|| / max_k||e^k||  < eps_s3   or   |det L| / prod_k||e^k|| < eps_s4

so both tolerances are dimensionless and live in [0, 1]. Larger means fire more.
The grid is log-spaced from "essentially never fires" (1e-12) to "fires often"
(1e-2) and includes the implementation default 1e-6 on both axes, which doubles
as a correctness gate against the main experiment.

ACCEPTANCE CRITERIA -- fixed before the sweep was run, so the outcome cannot be
rationalised afterwards:

  G1 (harness gate). At (1e-6, 1e-6) the sweep must reproduce the main
     experiment's `conditioning` arm at n=10 and n=20 to within bootstrap noise.
     If it does not, the harness is wrong and nothing else here is readable.

  A1 -> claim HOLDS, strengthened. No (eps_s3, eps_s4) in the grid brings the
     degeneracy trigger up to the `size` trigger at BOTH n=10 and n=20. The
     failure is then a property of the criterion, not of its tuning.

  A2 -> claim NARROWS. Some tolerance beats `size` at both n=10 and n=20. The
     paper must then say "at the tolerances tested in practice" rather than
     indicting the criterion, and the winning tolerance becomes a
     recommendation.

  A3 -> claim SOFTENS. Some tolerance closes the gap to within 3 points at both
     dimensions without beating `size`. "Actively harmful" then becomes "not
     competitive at any tolerance tested".

Usage:
    python benchmarks/scripts/run_tolerance_calibration.py [--dims 5 10 20]
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

from run_bbob_factorial import BBOBProblem  # noqa: E402
from run_init_restart_factorial import FactorialNM  # noqa: E402

#: Log-spaced, spanning never-fires to fires-often, with the implementation
#: default included so G1 is checkable.
EPS_GRID = (1e-12, 1e-6, 1e-4, 1e-2)

#: One initializer, held at the best-performing deterministic one. The sweep is
#: about the trigger; carrying all seven would multiply cost sevenfold and add a
#: factor the question does not involve.
INIT = "spendleySimplex"


def run_cell(problem, eps3, eps4, budget, trigger):
    criteria = {
        "conditioning": ["degenerate_simplex"],
        "size": ["small_simplex"],
    }[trigger]
    alg = FactorialNM(
        objective_fn=problem.objective,
        lower_bounds=problem.bounds[0],
        upper_bounds=problem.bounds[1],
        max_fes=budget,
        init_method=INIT,
        seed=101,
        stopping_criteria=criteria,
        criterion_tolerances={
            "degenerate_edge": eps3,
            "degenerate_hadamard": eps4,
        },
        construction="gaussian_best",
        # Required: restart_log is only populated when tracing is on, so
        # without this the restart counts silently read zero.
        trace_conditioning=True,
    )
    alg.run()
    return {
        "problem": problem.name,
        "dim": problem.dim,
        "function_id": problem.function_id,
        "trigger": trigger,
        "eps3": eps3,
        "eps4": eps4,
        "best": float(np.min(alg.fitness_values)),
        "restarts": len(alg.restart_log),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--budget-per-dim", type=int, default=2000)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1] / "results" / "tolerance_calibration.json"
        ),
    )
    args = ap.parse_args()

    pairs = list(itertools.product(EPS_GRID, EPS_GRID))
    suite = cocoex.Suite(
        "bbob",
        "",
        f"function_indices:1-24 dimensions:{','.join(map(str, args.dims))} "
        f"instance_indices:1-{args.instances}",
    )
    n_problems = len(suite)
    total = n_problems * (len(pairs) + 1)  # +1 for the `size` reference arm
    print(f"{total} runs: {n_problems} problems x ({len(pairs)} tolerance pairs + 1 size ref)",
          flush=True)

    records, t0 = [], time.time()
    for i, coco_problem in enumerate(suite):
        problem = BBOBProblem(coco_problem)
        budget = args.budget_per_dim * problem.dim
        for eps3, eps4 in pairs:
            records.append(run_cell(problem, eps3, eps4, budget, "conditioning"))
        # The comparator, run in the same harness so the comparison is internal.
        records.append(run_cell(problem, np.nan, np.nan, budget, "size"))
        if (i + 1) % 20 == 0:
            done = (i + 1) * (len(pairs) + 1)
            rate = done / (time.time() - t0)
            print(f"  {done}/{total}  {rate:.1f} runs/s  "
                  f"eta {(total - done) / rate / 60:.1f} min", flush=True)

    # f_L per problem over everything run here, More-Wild convention.
    best = {}
    for r in records:
        best[r["problem"]] = min(best.get(r["problem"], np.inf), r["best"])
    for r in records:
        r["f_L"] = best[r["problem"]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {
            "eps_grid": list(EPS_GRID), "init": INIT, "dims": args.dims,
            "instances": args.instances, "budget_per_dim": args.budget_per_dim,
            "construction": "gaussian_best", "seed": 101,
            "acceptance": "G1 harness gate; A1 holds / A2 narrows / A3 softens",
        },
        "records": records,
    }, indent=1))
    print(f"wrote {out}  ({len(records)} records, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
