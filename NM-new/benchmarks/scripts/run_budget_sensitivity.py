"""
Is the dimensional story a budget artifact?

Everything so far runs at a budget of 2000n. At n=20 the no-restart arm solves
only 11.4% of problems, so that dimension may be budget-starved rather than
intrinsically hard -- and the Kelley reversal, a co-headline, is strongest
exactly there. If Kelley's advantage evaporates at 4000n, it is a budget effect
wearing a dimension effect's clothes.

This also supplies the reference arm the tolerance sweep lacked. That sweep
compared the degeneracy trigger against a size test, but never against *not
restarting*, so it could not test whether the best-tuned degeneracy trigger is
still worse than doing nothing -- which is what the paper currently claims at
the default tolerance.

ACCEPTANCE CRITERIA, fixed before running:

  B1 (gate). At 2000n with the initializer fixed, the sign of the two headline
     comparisons must match the main experiment: Kelley > none at n=20, and
     gbnm < none at n=20. If fixing the initializer flips a sign, this design is
     not comparable to the main result and nothing below is readable.

  B2 -> claim HOLDS. At 4000n, Kelley still beats none at n=20 and gbnm still
     loses to it. The dimensional story is not a budget artifact.

  B3 -> claim BECOMES BUDGET-SPECIFIC. At 4000n the Kelley advantage vanishes or
     reverses. The paper must say "at a budget of 2000n" throughout and the
     reversal is demoted.

  B4 -> partial. The advantage persists but shrinks by more than half. State the
     budget dependence explicitly.

  C1 (tolerance follow-up). Is the degeneracy trigger at its best tolerance
     (eps_s4 = 1e-40) still worse than not restarting at n=20? If yes, the
     paper's "actively harmful" stands beyond the default. If no, that sentence
     must be narrowed to the default tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cocoex  # noqa: E402

from run_bbob_factorial import BBOBProblem  # noqa: E402
from run_init_restart_factorial import FactorialKelley, FactorialNM  # noqa: E402

INIT = "spendleySimplex"

#: (label, criteria, construction, single_descent, tolerances)
ARMS = [
    ("none",            [],                                       "gaussian_best", True,  None),
    ("progress",        [],                                       "oriented",      True,  None),
    ("size",            ["small_simplex"],                        "gaussian_best", False, None),
    ("flat",            ["flat_simplex"],                         "gaussian_best", False, None),
    ("gbnm_default",    ["small_simplex", "degenerate_simplex"],  "gaussian_best", False,
     {"degenerate_edge": 1e-6, "degenerate_hadamard": 1e-6}),
    # The best tolerance found by the calibration sweep, carried forward so the
    # comparison against `none` can finally be made.
    ("conditioning_tuned", ["degenerate_simplex"],                "gaussian_best", False,
     {"degenerate_edge": 0.0, "degenerate_hadamard": 1e-40}),
]


def run_arm(problem, arm, budget):
    label, criteria, construction, single, tols = arm
    cls = FactorialKelley if label == "progress" else FactorialNM
    kw = dict(
        objective_fn=problem.objective,
        lower_bounds=problem.bounds[0],
        upper_bounds=problem.bounds[1],
        max_fes=budget,
        init_method=INIT,
        seed=101,
        stopping_criteria=criteria or ["small_simplex"],
        single_descent=single,
        construction=construction,
        trace_conditioning=True,
    )
    if tols:
        kw["criterion_tolerances"] = tols
    alg = cls(**kw)
    alg.run()
    restarts = (
        alg.num_oriented_restarts if label == "progress"
        else (0 if single else len(alg.restart_log))
    )
    return {
        "problem": problem.name, "dim": problem.dim, "arm": label,
        "budget_mult": budget // problem.dim,
        "best": float(np.min(alg.fitness_values)), "restarts": int(restarts),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2000, 4000])
    ap.add_argument("--out", default=str(
        Path(__file__).resolve().parents[1] / "results" / "budget_sensitivity.json"))
    args = ap.parse_args()

    suite = cocoex.Suite(
        "bbob", "",
        f"function_indices:1-24 dimensions:{','.join(map(str,args.dims))} "
        f"instance_indices:1-{args.instances}")
    total = len(suite) * len(ARMS) * len(args.budgets)
    print(f"{total} runs: {len(suite)} problems x {len(ARMS)} arms x {len(args.budgets)} budgets",
          flush=True)

    records, t0 = [], time.time()
    for i, cp in enumerate(suite):
        p = BBOBProblem(cp)
        for mult in args.budgets:
            for arm in ARMS:
                records.append(run_arm(p, arm, mult * p.dim))
        if (i + 1) % 30 == 0:
            done = (i + 1) * len(ARMS) * len(args.budgets)
            rate = done / (time.time() - t0)
            print(f"  {done}/{total}  {rate:.1f}/s  eta {(total-done)/rate/60:.0f} min", flush=True)

    Path(args.out).write_text(json.dumps(
        {"config": {"init": INIT, "dims": args.dims, "budgets": args.budgets,
                    "instances": args.instances, "seed": 101,
                    "arms": [a[0] for a in ARMS]},
         "records": records}, indent=1))
    print(f"wrote {args.out}  ({len(records)} records, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
