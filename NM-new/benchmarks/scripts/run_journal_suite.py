#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Journal benchmark suite: runs the extended experiments for the c-NM paper.

Four suites, corresponding to the new subsections of Section 4:

  rw           CEC2020 real-world constrained subset (12 problems, n = 2..38)
  gseries      CEC2006 g-series problems with known optima
  scalability  Dimension sweep over four parameterized families
  blackbox     Finite element bracket sizing, the expensive simulation case
  mopta08      MOPTA08, the 124-variable automotive benchmark (see below)
  all          Every suite above except mopta08

Usage:
    python benchmarks/scripts/run_journal_suite.py <suite> [--quick] [--jobs N]
                                                   [--runs N]

    --quick   Small budget for smoke-testing the pipeline end to end
    --jobs N  Worker processes (default: all cores). --jobs 1 disables parallelism.
    --runs N  Independent runs per cell (default 30, the paper's protocol)
    --init M  Simplex initializer for the Nelder-Mead variants (default uniform)

On the choice of initializer: the package's default, "spendleySimplex", builds a
regular simplex around the centre of the box and ignores the random seed
entirely. The seed then reaches the algorithm only through the restart strategy,
which fires only when a stopping criterion triggers. On problems where restarts
do not fire within the budget, all 30 "independent runs" are the same run
repeated, and the reported standard deviation measures floating-point noise
rather than robustness. This was observed directly: on the black-box suite at
2,000 evaluations, the adaptive variant produced one distinct result across all
30 seeds. The default here is therefore "uniform", which samples the initial
simplex from the box and so makes the runs genuinely independent. Pass
--init spendleySimplex to reproduce the package default instead.

`mopta08` is excluded from `all` deliberately. Each of its evaluations launches a
30-90 MB executable and costs roughly 0.3 s, so the full protocol is around 30
core-hours for that suite alone, against minutes for the algebraic ones. It needs
to be an explicit decision, and it needs its executable installed first:

    python benchmarks/scripts/setup_mopta08.py

Results are written to benchmarks/results/ as JSON, plus Markdown and LaTeX
tables and, where applicable, figures.

Evaluation budgets differ per suite by design. The original experiment used 1e5
function evaluations, which is retained for the algebraic suites so that the new
numbers sit on the same scale as the paper's Tables 2 and 3. The black-box suite
uses a far smaller budget because each evaluation is a finite element solve; the
whole point of that case is that evaluations are scarce.
"""

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead  # noqa: E402
from nelder_mead.algorithms.nelder_mead import NelderMead  # noqa: E402
from nelder_mead.algorithms.scipy_baselines import (  # noqa: E402
    get_baseline_algorithms,
)
from nelder_mead.benchmarks.blackbox import get_blackbox_problems  # noqa: E402
from nelder_mead.benchmarks.cec2020_rw import (  # noqa: E402
    get_cec2020_rw_problems,
)
from nelder_mead.benchmarks.constrained import get_cec_problems  # noqa: E402
from nelder_mead.benchmarks.scalable_constrained import (  # noqa: E402
    get_scalability_sweep,
)
from nelder_mead.constraints.augmented_lagrangian import (  # noqa: E402
    AugmentedLagrangian,
)
from nelder_mead.testing.batch_runner import BatchRunner  # noqa: E402
from nelder_mead.testing.result_analyzer import ResultAnalyzer  # noqa: E402


RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

# 30 runs with distinct seeds, matching the protocol of the original experiment.
NUM_RUNS = 30
SEEDS = list(range(101, 101 + NUM_RUNS))

# Budgets per suite. See the module docstring for why they differ.
BUDGETS = {
    "rw": 100_000,
    "gseries": 100_000,
    "scalability": 100_000,
    "blackbox": 2_000,
    # The budget the MOPTA08 literature uses (TuRBO, SCBO, BAxUS all report at
    # 2,000 evaluations), which also keeps the suite to a feasible wall-clock time.
    "mopta08": 2_000,
}

#: Suites run by "all". MOPTA08 is excluded: at ~0.3 s per evaluation it costs
#: roughly as much as every other suite combined, so it is opted into explicitly.
DEFAULT_SUITES = ("blackbox", "gseries", "rw", "scalability")

QUICK_RUNS = 3
QUICK_BUDGET = 2_000
QUICK_BLACKBOX_BUDGET = 300
QUICK_MOPTA08_BUDGET = 200

#: The algorithm under test, named as in the paper.
REFERENCE_ALGORITHM = "cNM"

#: Simplex initializer for the Nelder-Mead variants. See the module docstring:
#: the package default is deterministic and would make the 30 runs identical.
DEFAULT_INIT_METHOD = "uniform"

#: Total constraint violation below which a solution is reported as feasible.
#: The CEC constrained-optimization protocols (CEC2006 onwards, and Kumar et al.
#: 2020) report at 1e-4 rather than at machine precision, because an equality
#: constraint handled by a penalty method cannot be driven to 1e-8 within a
#: realistic evaluation budget. Reporting at 1e-8 would show 0% feasibility for
#: every algorithm on every equality-constrained problem, which distinguishes
#: nothing. BaseAlgorithm's internal ranking keeps its stricter 1e-8 threshold;
#: this tolerance governs reporting and the statistical comparison only.
FEASIBILITY_TOL = 1e-4


def make_barrier(problem):
    """
    Build a fresh augmented Lagrangian barrier for one run.

    Passed to BatchRunner as `barrier_factory` rather than as a `barrier` instance:
    the barrier carries multipliers, so a shared instance would couple consecutive
    seeds and make the runs non-independent.

    Args:
        problem: The problem being solved

    Returns:
        A new AugmentedLagrangian sized for that problem
    """
    return AugmentedLagrangian(
        bounds=problem.bounds,
        num_solutions=problem.dim + 1,
        m_eq=problem.num_eq_constraints,
        p_ineq=problem.num_ineq_constraints,
    )


def algorithm_configs(init_method=DEFAULT_INIT_METHOD):
    """
    Return the algorithms compared in every suite.

    c-NM and its adaptive variant are the methods under test; the rest are the
    library baselines. All of them route their evaluations through the same
    BaseAlgorithm machinery, so the penalty, the feasibility rule and the
    evaluation counter are identical across the comparison.

    Args:
        init_method: Simplex initializer for the Nelder-Mead variants

    Returns:
        List of BatchRunner algorithm configuration dicts
    """
    configs = [
        {"class": NelderMead, "name": REFERENCE_ALGORITHM, "init_method": init_method},
        {"class": AdaptiveNelderMead, "name": "aNM", "init_method": init_method},
    ]
    configs.extend(get_baseline_algorithms())
    return configs


def suite_problems(suite):
    """
    Return the problems for one suite.

    Args:
        suite: One of "rw", "gseries", "scalability", "blackbox"

    Returns:
        List of OptimizationProblem instances

    Raises:
        ValueError: If the suite name is not recognised
    """
    if suite == "rw":
        return list(get_cec2020_rw_problems().values())
    if suite == "gseries":
        return list(get_cec_problems().values())
    if suite == "scalability":
        return [
            problem
            for family in get_scalability_sweep().values()
            for _, problem in family
        ]
    if suite == "blackbox":
        # MOPTA08 is excluded here; it has its own suite so that its very
        # different evaluation cost is an explicit choice rather than a surprise.
        return list(get_blackbox_problems(include_mopta08=False).values())
    if suite == "mopta08":
        from nelder_mead.benchmarks.mopta08 import MOPTA08, is_available

        if not is_available():
            raise RuntimeError(
                "The MOPTA08 executable is not installed. Run\n"
                "    python benchmarks/scripts/setup_mopta08.py\n"
                "to download and verify it for this machine."
            )
        return [MOPTA08(auto_download=False)]

    raise ValueError(f"Unknown suite '{suite}'. Choose from {sorted(BUDGETS)}.")


def _run_one(task):
    """
    Execute one (algorithm, problem, seed) cell. Runs in a worker process.

    Args:
        task: Tuple of (config index, suite, problem index, seed, max_fes,
              init_method)

    Returns:
        Tuple of (config index, problem index, serialized ExperimentResult)
    """
    config_index, suite, problem_index, seed, max_fes, init_method = task

    # Some benchmark objectives are undefined at parts of their own box (G08 has a
    # pole at x1 = 0, for instance). BaseAlgorithm already converts those to a large
    # penalty, which is the correct behaviour, so the warnings are expected rather
    # than diagnostic. They are captured and counted instead of printed, because at
    # the full protocol they would otherwise emit millions of lines.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result_tuple = _execute(task)
    result_tuple[2]["num_warnings"] = len(captured)
    return result_tuple


def _execute(task):
    """Run a single cell. Separated so warnings can be captured around it."""
    config_index, suite, problem_index, seed, max_fes, init_method = task

    config = algorithm_configs(init_method)[config_index]
    problem = suite_problems(suite)[problem_index]
    params = {k: v for k, v in config.items() if k not in ("class", "name")}

    results = BatchRunner(verbose=False).run_experiment(
        algorithm_class=config["class"],
        problem=problem,
        num_runs=1,
        seeds=[seed],
        max_fes=max_fes,
        barrier_factory=make_barrier,
        **params,
    )
    result = results[0]

    return (
        config_index,
        problem_index,
        {
            "algorithm_name": config["name"],
            "problem_name": problem.name,
            "seed": result.seed,
            "best_solution": np.asarray(result.best_solution).tolist(),
            "best_fitness": float(result.best_fitness),
            "best_violation": float(result.best_violation),
            "fes_used": int(result.fes_used),
            "wall_time": float(result.wall_time),
            "success": bool(result.success),
            "convergence_history": [
                (int(fes), float(value)) for fes, value in result.convergence_history
            ],
        },
    )


def run_suite(suite, quick=False, jobs=None, num_runs=NUM_RUNS,
              init_method=DEFAULT_INIT_METHOD):
    """
    Run one suite and write its results.

    Every (algorithm, problem, seed) cell is independent, so the whole grid is
    distributed across worker processes. At the paper's protocol a single suite is
    tens of thousands of runs of up to 1e5 evaluations, which is hours of
    single-core Python.

    Args:
        suite: Suite name
        quick: If True, use a small budget and few runs to smoke-test the pipeline
        jobs: Number of worker processes; None means all cores, 1 means serial
        num_runs: Independent runs per (algorithm, problem) cell
        init_method: Simplex initializer for the Nelder-Mead variants

    Returns:
        Path to the JSON results file
    """
    problems = suite_problems(suite)
    configs = algorithm_configs(init_method)

    max_fes = BUDGETS[suite]
    seeds = list(range(101, 101 + num_runs))
    if quick:
        max_fes = {
            "blackbox": QUICK_BLACKBOX_BUDGET,
            "mopta08": QUICK_MOPTA08_BUDGET,
        }.get(suite, QUICK_BUDGET)
        seeds = seeds[:QUICK_RUNS]

    tasks = [
        (config_index, suite, problem_index, seed, max_fes, init_method)
        for config_index in range(len(configs))
        for problem_index in range(len(problems))
        for seed in seeds
    ]

    print(f"\n{'=' * 70}")
    print(f"Suite: {suite}{'  [QUICK]' if quick else ''}")
    print(f"{'=' * 70}")
    print(f"Algorithms : {[c['name'] for c in configs]}")
    print(f"Problems   : {len(problems)}")
    print(f"Runs/cell  : {len(seeds)}   Budget: {max_fes:,} FES")
    print(f"Initializer: {init_method}")
    print(f"Total runs : {len(tasks):,}")
    print(f"{'=' * 70}\n")

    started = time.time()
    raw = {}

    if jobs == 1:
        completed_iter = (_run_one(task) for task in tasks)
        for done, (config_index, problem_index, record) in enumerate(
            completed_iter, 1
        ):
            raw.setdefault(config_index, {}).setdefault(problem_index, []).append(
                record
            )
            _progress(done, len(tasks), started)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_run_one, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), 1):
                config_index, problem_index, record = future.result()
                raw.setdefault(config_index, {}).setdefault(
                    problem_index, []
                ).append(record)
                _progress(done, len(tasks), started)

    print()

    payload = _assemble(
        suite, configs, problems, raw, max_fes, seeds, started, init_method
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""
    json_path = RESULTS_DIR / f"journal_{suite}{suffix}_results.json"
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    _write_tables(suite, payload, suffix)
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")

    return json_path


def _progress(done, total, started):
    """Print a progress indicator with an ETA.

    Rewrites one line on a terminal; when the output is redirected to a file or a
    pipe, carriage returns produce an unreadable wall of text, so it falls back to
    printing at every 5% instead.
    """
    interactive = sys.stdout.isatty()
    if not interactive:
        step = max(1, total // 20)
        if done % step and done != total:
            return

    elapsed = time.time() - started
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    line = (
        f"  {done:,}/{total:,} runs  "
        f"({100 * done / total:5.1f}%)  "
        f"elapsed {elapsed / 60:6.1f} min  "
        f"eta {remaining / 60:6.1f} min"
    )
    print(f"\r{line}" if interactive else line, end="" if interactive else "\n",
          flush=True)


def _assemble(suite, configs, problems, raw, max_fes, seeds, started,
              init_method):
    """Turn the raw per-cell records into the results payload."""
    payload = {
        "suite": suite,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_runs": len(seeds),
        "seeds": seeds,
        "max_fes": max_fes,
        "init_method": init_method,
        "feasibility_tol": FEASIBILITY_TOL,
        "reference_algorithm": REFERENCE_ALGORITHM,
        "wall_time_seconds": time.time() - started,
        "algorithms": [c["name"] for c in configs],
        "num_warnings": sum(
            record.get("num_warnings", 0)
            for by_problem in raw.values()
            for records in by_problem.values()
            for record in records
        ),
        "problems": {},
        "statistics": {},
    }

    for problem_index, problem in enumerate(problems):
        payload["problems"][problem.name] = {
            "dimension": problem.dim,
            "num_eq_constraints": problem.num_eq_constraints,
            "num_ineq_constraints": problem.num_ineq_constraints,
            "optimum_value": problem.optimum_value,
        }

        payload["statistics"][problem.name] = {}
        for config_index, config in enumerate(configs):
            records = raw.get(config_index, {}).get(problem_index, [])
            records.sort(key=lambda r: r["seed"])
            fitness = np.array(
                [r["best_fitness"] for r in records if np.isfinite(r["best_fitness"])]
            )
            feasible = [r for r in records if r["best_violation"] < FEASIBILITY_TOL]

            payload["statistics"][problem.name][config["name"]] = {
                "mean": float(np.mean(fitness)) if fitness.size else None,
                "median": float(np.median(fitness)) if fitness.size else None,
                "std": float(np.std(fitness, ddof=1)) if fitness.size > 1 else 0.0,
                "best": float(np.min(fitness)) if fitness.size else None,
                "worst": float(np.max(fitness)) if fitness.size else None,
                "feasibility_rate": len(feasible) / len(records) if records else 0.0,
                "mean_fes": float(np.mean([r["fes_used"] for r in records]))
                if records
                else 0.0,
                "mean_wall_time": float(np.mean([r["wall_time"] for r in records]))
                if records
                else 0.0,
                "runs": records,
            }

    payload["statistical_tests"] = _statistical_tests(payload, configs)
    return payload


def _statistical_tests(payload, configs):
    """Run the Friedman omnibus test and the pairwise Mann-Whitney comparisons."""
    from nelder_mead.testing.batch_runner import ExperimentResult

    def as_results(problem_name, algorithm_name):
        stats = payload["statistics"][problem_name][algorithm_name]
        return [
            ExperimentResult(
                algorithm_name=algorithm_name,
                problem_name=problem_name,
                seed=r["seed"],
                best_solution=np.array(r["best_solution"]),
                best_fitness=r["best_fitness"],
                best_violation=r["best_violation"],
                fes_used=r["fes_used"],
            )
            for r in stats["runs"]
        ]

    names = [c["name"] for c in configs]
    problem_names = list(payload["problems"])

    tests = {"friedman": None, "mann_whitney": {}, "mann_whitney_holm": {}}

    by_algorithm = {
        name: {problem: as_results(problem, name) for problem in problem_names}
        for name in names
    }

    try:
        tests["friedman"] = ResultAnalyzer.friedman_test(
            by_algorithm, feasibility_tol=FEASIBILITY_TOL
        )
    except ValueError as exc:
        tests["friedman"] = {"error": str(exc)}

    for problem in problem_names:
        by_problem = {name: as_results(problem, name) for name in names}
        try:
            comparisons = ResultAnalyzer.mannwhitney_vs_reference(
                by_problem,
                reference=REFERENCE_ALGORITHM,
                feasibility_tol=FEASIBILITY_TOL,
            )
        except ValueError as exc:
            tests["mann_whitney"][problem] = {"error": str(exc)}
            continue

        tests["mann_whitney"][problem] = comparisons
        tests["mann_whitney_holm"][problem] = ResultAnalyzer.holm_correction(
            {alg: c["p_value"] for alg, c in comparisons.items()}
        )

    return tests


def _write_tables(suite, payload, suffix):
    """Write the Markdown and LaTeX summary tables for one suite."""
    algorithms = payload["algorithms"]
    lines_md = [
        f"# Journal suite results: {suite}",
        "",
        f"Generated {payload['timestamp']}",
        "",
        f"- Runs per cell: {payload['num_runs']}",
        f"- Budget: {payload['max_fes']:,} function evaluations",
        f"- Reference algorithm: {payload['reference_algorithm']}",
        f"- Feasibility tolerance: {FEASIBILITY_TOL:g} (CEC reporting convention)",
        f"- Simplex initializer: {payload.get('init_method', 'default')}",
        f"- Suppressed numerical warnings: {payload.get('num_warnings', 0):,} "
        "(objectives evaluated outside their domain, penalized by the barrier)",
        "",
        "Objectives and constraint violations are compared with Deb's feasibility "
        "rules: a feasible run always outranks an infeasible one, so an algorithm "
        "that never reaches the feasible region cannot win by reporting an "
        "objective below the constrained optimum.",
        "",
        "## Statistics",
        "",
        "| Problem | n | Algorithm | Mean | Median | Std | Best | Worst | Feas. |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(value):
        if value is None:
            return "-"
        return f"{value:.6g}"

    for problem_name, meta in payload["problems"].items():
        for algorithm in algorithms:
            stats = payload["statistics"][problem_name][algorithm]
            lines_md.append(
                f"| {problem_name} | {meta['dimension']} | {algorithm} | "
                f"{fmt(stats['mean'])} | {fmt(stats['median'])} | "
                f"{fmt(stats['std'])} | {fmt(stats['best'])} | "
                f"{fmt(stats['worst'])} | {stats['feasibility_rate']:.2f} |"
            )

    tests = payload["statistical_tests"]
    friedman = tests.get("friedman") or {}
    lines_md += ["", "## Friedman omnibus test", ""]
    if "error" in friedman:
        lines_md.append(f"Not run: {friedman['error']}")
    else:
        lines_md += [
            f"- chi-squared = {friedman['statistic']:.6g}",
            f"- p = {friedman['p_value']:.6g}",
            f"- problems = {friedman['num_problems']}",
            "",
            "| Algorithm | Mean rank |",
            "|---|---|",
        ]
        for algorithm, rank in sorted(
            friedman["mean_ranks"].items(), key=lambda kv: kv[1]
        ):
            lines_md.append(f"| {algorithm} | {rank:.3f} |")

    lines_md += [
        "",
        f"## Mann-Whitney U against {payload['reference_algorithm']}",
        "",
        "Holm-adjusted p-values in parentheses. Bold indicates p < 0.05 after "
        "adjustment.",
        "",
        "| Problem | " + " | ".join(a for a in algorithms if a != REFERENCE_ALGORITHM)
        + " |",
        "|---" * (1 + len(algorithms) - 1) + "|",
    ]
    for problem_name in payload["problems"]:
        comparisons = tests["mann_whitney"].get(problem_name, {})
        adjusted = tests["mann_whitney_holm"].get(problem_name, {})
        cells = []
        for algorithm in algorithms:
            if algorithm == REFERENCE_ALGORITHM:
                continue
            entry = comparisons.get(algorithm)
            if not isinstance(entry, dict) or "p_value" not in entry:
                cells.append("-")
                continue
            raw_p = entry["p_value"]
            adj_p = adjusted.get(algorithm, raw_p)
            cell = f"{raw_p:.3g} ({adj_p:.3g})"
            cells.append(f"**{cell}**" if adj_p < 0.05 else cell)
        lines_md.append(f"| {problem_name} | " + " | ".join(cells) + " |")

    md_path = RESULTS_DIR / f"journal_{suite}{suffix}_summary.md"
    md_path.write_text("\n".join(lines_md) + "\n")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")

    _write_latex(suite, payload, suffix)


def _write_latex(suite, payload, suffix):
    """Write a LaTeX table in the layout of the paper's Tables 2 and 3."""
    rows = [
        r"\begin{tabular}{llrrrrr}",
        r"\hline",
        r"Problem & Algorithm & Mean & Median & Std. Dev & Best & Worst \\",
        r"\hline",
    ]

    def fmt(value):
        return "-" if value is None else f"{value:.6g}"

    for problem_name in payload["problems"]:
        stats_by_alg = payload["statistics"][problem_name]
        finite = {
            a: s["mean"] for a, s in stats_by_alg.items() if s["mean"] is not None
        }
        best_algorithm = min(finite, key=finite.get) if finite else None

        for index, algorithm in enumerate(payload["algorithms"]):
            stats = stats_by_alg[algorithm]
            label = problem_name if index == 0 else ""
            cells = [
                fmt(stats["mean"]),
                fmt(stats["median"]),
                fmt(stats["std"]),
                fmt(stats["best"]),
                fmt(stats["worst"]),
            ]
            name = algorithm
            if algorithm == best_algorithm:
                name = rf"\textbf{{{algorithm}}}"
                cells = [rf"\textbf{{{c}}}" for c in cells]
            rows.append(f"{label} & {name} & " + " & ".join(cells) + r" \\")
        rows.append(r"\hline")

    rows.append(r"\end{tabular}")

    tex_path = RESULTS_DIR / f"journal_{suite}{suffix}_table.tex"
    tex_path.write_text("\n".join(rows) + "\n")
    print(f"Wrote {tex_path.relative_to(REPO_ROOT)}")


def main():
    """Parse arguments and run the requested suite."""
    parser = argparse.ArgumentParser(
        description="Run the journal benchmark suites for the c-NM paper."
    )
    parser.add_argument(
        "suite", choices=sorted(BUDGETS) + ["all"], help="Which suite to run"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=NUM_RUNS,
        help=f"Independent runs per cell (default {NUM_RUNS}, the paper's protocol)",
    )
    parser.add_argument(
        "--init",
        default=DEFAULT_INIT_METHOD,
        choices=[
            "uniform", "gaussian", "spendleySimplex", "pfefferSimplex",
            "adaptiveSimplex",
        ],
        help=(
            f"Simplex initializer for the Nelder-Mead variants "
            f"(default {DEFAULT_INIT_METHOD}; spendleySimplex ignores the seed)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small budget and few runs, to smoke-test the pipeline",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Worker processes (default: all cores; 1 disables parallelism)",
    )
    args = parser.parse_args()

    suites = list(DEFAULT_SUITES) if args.suite == "all" else [args.suite]

    for suite in suites:
        run_suite(
            suite,
            quick=args.quick,
            jobs=args.jobs,
            num_runs=args.runs,
            init_method=args.init,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
