"""
Initialization x restart-trigger factorial for the Nelder-Mead method.

The experiment behind the claim that simplex *conditioning* is the shared latent
variable: initialization sets it, contraction destroys it, restart resets it.

Three published facts, never measured together:

  - Wessing (Optimization Letters 13(4):847-856, 2019): initialization changes NM
    runtime by orders of magnitude, and the effect survives Kelley's oriented
    restarts. He held the restart scheme fixed; the interaction was never
    quantified.
  - Takenaga, Ozaki & Onishi (Optimization Letters 17(2):283-297, 2023): initial
    simplex size and shape drive performance under a limited budget.
  - Luersen & Le Riche (Computers & Structures 82(23):2251-2260, 2004): GBNM
    restarts on a *conditioning* test, eq. (9) -- and its authors report the
    tolerances eps_s3, eps_s4 "may be difficult to tune, so that a simplex which
    is becoming small may be tagged as degenerated before". Nobody has ablated
    that trigger or calibrated those tolerances.

Factors:
  A  initialization   (7 levels, including two GBNM restart scales)
  B  restart trigger  (5 levels: none / size / conditioning / GBNM / flat)

Restart *construction* is held at `gaussian_best` for this pass -- it is a third
factor and multiplies the cost; vary it once the first two are understood.

Responses: final fitness, and the conditioning trace -- (edge_ratio,
hadamard_ratio) per iteration -- which is what separates this from a ranking
table. Trigger attribution records which criterion fired, so the size/degeneracy
mis-tagging the GBNM authors named becomes a number.

Usage:
    python benchmarks/scripts/run_init_restart_factorial.py --quick
    python benchmarks/scripts/run_init_restart_factorial.py --seeds 15 --budget 10000
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nelder_mead.algorithms.kelley_nm import KelleyNelderMead  # noqa: E402
from nelder_mead.algorithms.nelder_mead import NelderMead  # noqa: E402
from nelder_mead.benchmarks.unconstrained import (  # noqa: E402
    Ackley,
    Griewank,
    Rastrigin,
    Rosenbrock,
    Sphere,
)
from nelder_mead.core import simplex_operations as ops  # noqa: E402
from nelder_mead.core.simplex_operations import simplex_conditioning  # noqa: E402
from nelder_mead.initialization.simplex_init import SimplexInitializer  # noqa: E402

# --------------------------------------------------------------------------
# Factor A -- initialization
#
# `scale` shrinks the simplex about x0 after construction. GBNM draws the size
# of a restart simplex uniformly from 2-10% of the smallest domain dimension
# (Luersen & Le Riche 2004, sec. 2.2); `spendley_simplex` in this repo builds it
# at 100% of that dimension, i.e. 10-50x larger. That is a factor level, not a
# bug, and it is the axis Wessing's scale-dependence argument lives on.
# --------------------------------------------------------------------------
INITIALIZATIONS = {
    "pfeffer": ("pfefferSimplex", 1.0),        # MATLAB fminsearch default
    "spendley_full": ("spendleySimplex", 1.0),  # == GBNM eq. (A.1) at a = min(ub-lb)
    "spendley_10pct": ("spendleySimplex", 0.10),  # GBNM restart scale, upper end
    "spendley_2pct": ("spendleySimplex", 0.02),   # GBNM restart scale, lower end
    "uniform": ("uniform", 1.0),
    "gaussian": ("gaussian", 1.0),
    "positive_basis": ("minimalPositiveBasis", 1.0),  # the g-NM initializer
}

# --------------------------------------------------------------------------
# Factor B -- restart trigger. In this codebase the trigger *is* the stopping
# criterion: `run()` restarts whenever one fires. "none" instead terminates, so
# that arm is a single descent (without it NM fails every transformation after
# convergence and shrinks forever, which is not a descent and not a restart).
# --------------------------------------------------------------------------
TRIGGERS = {
    "none": [],
    "size": ["small_simplex"],
    "conditioning": ["degenerate_simplex"],
    "gbnm": ["small_simplex", "degenerate_simplex"],
    "flat": ["flat_simplex"],
    # Kelley (1999): stagnation is failure of the sufficient-decrease condition
    #     f-bar^{k+1} - f-bar^k < -alpha ||D_k f||^2
    # on the *mean* simplex value, remedied by an oriented restart rather than by
    # a fresh random simplex. This arm therefore varies construction as well as
    # trigger; that is the published method and the confound is stated in the
    # paper rather than engineered away.
    "progress": [],
}

# --------------------------------------------------------------------------
# Factor C -- restart *construction*: how to re-seed once a trigger has fired.
#
# Separating this from the trigger is what lets the study attribute an effect to
# the detector rather than to the re-seeding. `gaussian_best` draws a fresh
# simplex about the incumbent (a global move on a bounded box); `oriented` is
# Kelley eq. (8.7), a small orthogonal simplex aligned with -grad (a local move).
# Kelley's published method pairs his detector with `oriented`, so without this
# axis his arm confounds the two.
# --------------------------------------------------------------------------
CONSTRUCTIONS = ("gaussian_best", "oriented")

PROBLEMS = [Sphere, Rosenbrock, Rastrigin, Ackley, Griewank]


class FactorialNM(NelderMead):
    """NM with a scaled initial simplex and an optional terminate-on-stagnation arm."""

    def __init__(self, *args, init_scale: float = 1.0, single_descent: bool = False,
                 construction: str = "gaussian_best", **kw):
        super().__init__(*args, **kw)
        self.init_scale = init_scale
        self.single_descent = single_descent
        self.construction = construction
        self._stop_requested = False

    def initialize_population(self, x0=None):
        simplex = super().initialize_population(x0)
        if self.init_scale != 1.0:
            # Contract about the first vertex, then re-clip to the box.
            simplex = simplex[0] + self.init_scale * (simplex - simplex[0])
            simplex = np.clip(simplex, self.lower_bounds, self.upper_bounds)
        return simplex

    def _restart_simplex(self):
        if self.single_descent:
            self._stop_requested = True
            return
        if self.construction == "oriented":
            self.simplex = ops.oriented_restart(
                self.simplex, self.fitness_values,
                self.lower_bounds, self.upper_bounds,
            )
            (
                self.fitness_values,
                self.eq_violations_list,
                self.ineq_violations_list,
                self.total_violations,
            ) = self.evaluate_population(self.simplex)
            self._sort_simplex()
            return
        super()._restart_simplex()

    def should_terminate(self):
        return self._stop_requested or super().should_terminate()


class FactorialKelley(FactorialNM, KelleyNelderMead):
    """Kelley's oriented restart with the factorial's scaled-initialization hook.

    FactorialNM comes first so its ``__init__`` consumes ``init_scale`` and
    ``single_descent`` before delegating; it defines no ``_sort_simplex``, so the
    MRO still resolves Kelley's sufficient-decrease test. Kelley overrides
    ``_oriented_restart``, not ``_restart_simplex``, so the two do not collide.
    """


def run_one(problem, init_name, trigger_name, seed, budget,
            construction="gaussian_best"):
    """One (init, trigger, construction, problem, seed) cell -> a result record."""
    init_method, scale = INITIALIZATIONS[init_name]
    criteria = TRIGGERS[trigger_name]
    lower, upper = problem.bounds

    cls = FactorialKelley if trigger_name == "progress" else FactorialNM
    alg = cls(
        objective_fn=problem.objective,
        lower_bounds=lower,
        upper_bounds=upper,
        max_fes=budget,
        init_method=init_method,
        seed=seed,
        init_scale=scale,
        # "progress" disables the base random restart so that Kelley's oriented
        # restart is the only one acting, keeping one mechanism per arm.
        single_descent=trigger_name in ("none", "progress"),
        construction=construction,
        # "none" still needs a convergence test to stop on; give every arm the
        # same one so the single-descent arm is not simply run to exhaustion.
        stopping_criteria=criteria or ["small_simplex"],
        trace_conditioning=True,
    )
    alg.run()

    trace = np.asarray(alg.conditioning_trace) if alg.conditioning_trace else np.empty((0, 3))
    # In the single-descent arm the logged event is a *termination*, not a restart.
    single = trigger_name == "none"
    if trigger_name == "progress":
        triggers_fired = ["progress"] * alg.num_oriented_restarts
    else:
        triggers_fired = [] if single else [t for _, t in alg.restart_log]

    return {
        "problem": problem.name,
        "dim": problem.dim,
        "init": init_name,
        "trigger": trigger_name,
        "construction": "oriented" if trigger_name == "progress" else construction,
        "seed": seed,
        "best": float(np.min(alg.fitness_values)),
        "fes": int(alg.fes),
        "restarts": len(triggers_fired),
        "terminated_by": alg.restart_log[-1][1] if (single and alg.restart_log) else None,
        # Attribution: of the restarts GBNM's combined trigger fired, how many
        # were the degeneracy test rather than the size test? Their paper says
        # smallness gets mis-tagged as degeneracy; this is the measurement.
        "restarts_by_degeneracy": triggers_fired.count("degenerate_simplex"),
        "restarts_by_size": triggers_fired.count("small_simplex"),
        # Conditioning trace summary. hadamard_ratio is the shape term: 1 for
        # orthogonal equal edges, 0 for linearly dependent ones.
        "hadamard_median": float(np.median(trace[:, 2])) if len(trace) else None,
        "hadamard_min": float(trace[:, 2].min()) if len(trace) else None,
        "hadamard_final": float(trace[-1, 2]) if len(trace) else None,
        "edge_ratio_median": float(np.median(trace[:, 1])) if len(trace) else None,
        "frac_iters_below_1e6": (
            float((trace[:, 2] < 1e-6).mean()) if len(trace) else None
        ),
        "iterations": len(trace),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--quick", action="store_true", help="3 seeds, dims 2 and 5")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "results" / "init_restart_factorial.json"),
    )
    args = ap.parse_args()

    seeds = 3 if args.quick else args.seeds
    dims = [2, 5] if args.quick else args.dims

    cells = list(
        itertools.product(PROBLEMS, dims, INITIALIZATIONS, TRIGGERS, range(seeds))
    )
    print(
        f"{len(cells)} runs: {len(PROBLEMS)} problems x {len(dims)} dims x "
        f"{len(INITIALIZATIONS)} inits x {len(TRIGGERS)} triggers x {seeds} seeds",
        flush=True,
    )

    records, t0 = [], time.time()
    for i, (cls, dim, init, trigger, seed) in enumerate(cells):
        records.append(run_one(cls(dim=dim), init, trigger, 101 + seed, args.budget))
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(
                f"  {i + 1}/{len(cells)}  {rate:.1f} runs/s  "
                f"eta {(len(cells) - i - 1) / rate / 60:.1f} min",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "config": {
                    "seeds": seeds,
                    "budget": args.budget,
                    "dims": dims,
                    "initializations": {k: list(v) for k, v in INITIALIZATIONS.items()},
                    "triggers": TRIGGERS,
                    "restart_construction": "gaussian_best",
                },
                "records": records,
            },
            indent=1,
        )
    )
    print(f"wrote {out}  ({len(records)} records, {time.time() - t0:.0f}s)")


def demo():
    """Self-check: the instrumentation must actually distinguish the arms."""
    p = Rosenbrock(dim=5)
    single = run_one(p, "spendley_full", "none", 101, 3000)
    gbnm = run_one(p, "spendley_full", "gbnm", 101, 3000)

    assert single["restarts"] == 0, single
    assert single["iterations"] > 0, "conditioning trace never recorded"
    assert 0.0 <= single["hadamard_median"] <= 1.0, single

    # A single descent must lose conditioning: NM contracts, and contraction is
    # where 88.7-100% of simplex volume goes.
    assert single["hadamard_final"] < single["hadamard_median"], single

    # The scale factor must actually change the initial simplex.
    small = run_one(p, "spendley_2pct", "none", 101, 3000)
    assert small["best"] != single["best"], "init_scale had no effect"

    # Trigger attribution must add up.
    assert gbnm["restarts_by_degeneracy"] + gbnm["restarts_by_size"] == gbnm["restarts"]
    print("demo ok:", {k: single[k] for k in ("best", "iterations", "hadamard_final")})
    print("        gbnm:", {k: gbnm[k] for k in ("best", "restarts", "restarts_by_degeneracy", "restarts_by_size")})


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
