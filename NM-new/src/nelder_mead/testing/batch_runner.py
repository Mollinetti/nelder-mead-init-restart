#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Testing Infrastructure for Optimization Algorithms

This module provides infrastructure for running batch experiments with
optimization algorithms on benchmark problems. It supports:
- Running algorithms multiple times with different random seeds
- Comparing multiple algorithms on multiple problems
- Recording results in structured format for analysis

References:
    - Standard practice for benchmarking optimization algorithms
    - Statistical analysis of stochastic optimization algorithms
"""

from typing import Callable, List, Dict, Any, Type, Optional
import numpy as np
import time
from dataclasses import dataclass, field

from ..algorithms.base_algorithm import BaseAlgorithm
from ..benchmarks.problem_suite import OptimizationProblem
from ..constraints.barrier_base import Barrier


@dataclass
class ExperimentResult:
    """
    Result from a single algorithm run on a problem.

    Attributes:
        algorithm_name (str): Name of the algorithm
        problem_name (str): Name of the problem
        seed (int): Random seed used for this run
        best_solution (np.ndarray): Best solution found
        best_fitness (float): Best objective value found
        best_violation (float): Total constraint violation of best solution
        fes_used (int): Number of function evaluations used
        convergence_history (List[tuple]): List of (fes, best_fitness) tuples
        wall_time (float): Wall clock time in seconds
        success (bool): Whether optimum was reached within tolerance
        final_simplex (Optional[np.ndarray]): Final simplex state (if applicable)
    """

    algorithm_name: str
    problem_name: str
    seed: int
    best_solution: np.ndarray
    best_fitness: float
    best_violation: float
    fes_used: int
    convergence_history: List[tuple] = field(default_factory=list)
    wall_time: float = 0.0
    success: bool = False
    final_simplex: Optional[np.ndarray] = None


class BatchRunner:
    """
    Infrastructure for running batch experiments with optimization algorithms.

    This class provides methods for:
    1. Running a single algorithm on a single problem multiple times with different seeds
    2. Comparing multiple algorithms on multiple problems
    3. Recording all results in structured format for statistical analysis

    Example:
        >>> from nelder_mead.algorithms.nelder_mead import NelderMead
        >>> from nelder_mead.benchmarks.unconstrained import Sphere
        >>>
        >>> runner = BatchRunner()
        >>> problem = Sphere(dim=10)
        >>> results = runner.run_experiment(
        ...     algorithm_class=NelderMead,
        ...     problem=problem,
        ...     num_runs=30,
        ...     seeds=range(1, 31),
        ...     max_fes=5000
        ... )
        >>> print(f"Mean fitness: {np.mean([r.best_fitness for r in results])}")

    Validates: Requirements 15.1, 15.2, 15.4
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the batch runner.

        Args:
            verbose: If True, print progress messages during execution
        """
        self.verbose = verbose

    def run_experiment(
        self,
        algorithm_class: Type[BaseAlgorithm],
        problem: OptimizationProblem,
        num_runs: int,
        seeds: Optional[List[int]] = None,
        success_tolerance: float = 1e-2,
        barrier_factory: Optional[Callable[[OptimizationProblem], Barrier]] = None,
        **algorithm_params,
    ) -> List[ExperimentResult]:
        """
        Run an algorithm on a problem multiple times with different seeds.

        This method executes the specified algorithm on the given problem
        multiple times, each with a different random seed, and records
        detailed results for each run.

        Args:
            algorithm_class: Class of the algorithm to run (must inherit from BaseAlgorithm)
            problem: Optimization problem to solve
            num_runs: Number of independent runs to perform
            seeds: List of random seeds (if None, uses range(1, num_runs+1))
            success_tolerance: Tolerance for determining if optimum was reached
            barrier_factory: Callable building a fresh Barrier for each run. Barriers
                             are stateful (AugmentedLagrangian carries lambda, mu, rho
                             and beta across iterations), so passing a single instance
                             via ``barrier=`` leaks multipliers from one seed into the
                             next. Use this instead whenever a barrier is needed.
            **algorithm_params: Additional parameters to pass to algorithm constructor
                               (e.g., max_fes, init_method, delta_r, etc.)

        Returns:
            List of ExperimentResult objects, one for each run

        Raises:
            ValueError: If num_runs <= 0 or seeds length doesn't match num_runs
            TypeError: If algorithm_class doesn't inherit from BaseAlgorithm

        Example:
            >>> results = runner.run_experiment(
            ...     algorithm_class=NelderMead,
            ...     problem=Sphere(dim=10),
            ...     num_runs=30,
            ...     max_fes=5000,
            ...     init_method="spendleySimplex",
            ...     delta_r=1.0
            ... )

        Validates: Requirements 15.1, 15.2
        """
        # Validate inputs
        if num_runs <= 0:
            raise ValueError("num_runs must be positive")

        if not issubclass(algorithm_class, BaseAlgorithm):
            raise TypeError("algorithm_class must inherit from BaseAlgorithm")

        # Generate seeds if not provided
        if seeds is None:
            seeds = list(range(1, num_runs + 1))

        if len(seeds) != num_runs:
            raise ValueError(
                f"Length of seeds ({len(seeds)}) must match num_runs ({num_runs})"
            )

        # Print experiment info
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Running Experiment: {algorithm_class.__name__} on {problem.name}")
            print(f"{'='*70}")
            print(f"Problem dimension: {problem.dim}")
            print(f"Number of runs: {num_runs}")
            print(f"Seeds: {seeds[:5]}{'...' if len(seeds) > 5 else ''}")
            if problem.optimum_value is not None:
                print(f"Known optimum: {problem.optimum_value}")
            print(f"{'='*70}\n")

        # Run experiments
        results = []
        for run_idx, seed in enumerate(seeds):
            if self.verbose:
                print(
                    f"Run {run_idx + 1}/{num_runs} (seed={seed})...",
                    end=" ",
                    flush=True,
                )

            # Build a fresh barrier per run so multipliers never leak across seeds
            run_params = dict(algorithm_params)
            if barrier_factory is not None:
                run_params["barrier"] = barrier_factory(problem)

            # Create algorithm instance
            alg = algorithm_class(
                objective_fn=problem.objective,
                constraint_eq_fn=(
                    problem.constraint_eq if problem.num_eq_constraints > 0 else None
                ),
                constraint_ineq_fn=(
                    problem.constraint_ineq
                    if problem.num_ineq_constraints > 0
                    else None
                ),
                lower_bounds=problem.bounds[0],
                upper_bounds=problem.bounds[1],
                seed=seed,
                **run_params,
            )

            # Run algorithm and measure time
            start_time = time.time()
            try:
                alg.run()
                wall_time = time.time() - start_time

                # Get results
                (
                    best_solution,
                    best_fitness,
                    best_eq_vio,
                    best_ineq_vio,
                    best_total_vio,
                ) = alg.get_best_solution()

                # Check success (if optimum is known)
                success = False
                if problem.optimum_value is not None:
                    # Success if within tolerance of known optimum and feasible
                    fitness_error = abs(best_fitness - problem.optimum_value)
                    success = bool(
                        fitness_error <= success_tolerance and best_total_vio < 1e-6
                    )

                # Get final simplex if available (for Nelder-Mead algorithms)
                final_simplex = None
                if hasattr(alg, "simplex"):
                    final_simplex = alg.simplex.copy()

                # Create result object
                result = ExperimentResult(
                    algorithm_name=algorithm_class.__name__,
                    problem_name=problem.name,
                    seed=seed,
                    best_solution=best_solution.copy(),
                    best_fitness=best_fitness,
                    best_violation=best_total_vio,
                    fes_used=alg.fes,
                    convergence_history=alg.memory_global_best.copy(),
                    wall_time=wall_time,
                    success=success,
                    final_simplex=final_simplex,
                )

                results.append(result)

                if self.verbose:
                    print(
                        f"fitness={best_fitness:.6e}, fes={alg.fes}, "
                        f"time={wall_time:.2f}s, success={success}"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"FAILED: {str(e)}")
                # Record failed run with placeholder values
                result = ExperimentResult(
                    algorithm_name=algorithm_class.__name__,
                    problem_name=problem.name,
                    seed=seed,
                    best_solution=np.full(problem.dim, np.nan),
                    best_fitness=np.inf,
                    best_violation=np.inf,
                    fes_used=0,
                    convergence_history=[],
                    wall_time=time.time() - start_time,
                    success=False,
                    final_simplex=None,
                )
                results.append(result)

        # Print summary
        if self.verbose:
            self._print_experiment_summary(results, problem)

        return results

    def compare_algorithms(
        self,
        algorithm_configs: List[Dict[str, Any]],
        problems: List[OptimizationProblem],
        num_runs: int,
        seeds: Optional[List[int]] = None,
        success_tolerance: float = 1e-2,
    ) -> Dict[str, Dict[str, List[ExperimentResult]]]:
        """
        Compare multiple algorithms on multiple problems.

        This method runs each algorithm on each problem multiple times and
        organizes the results for easy comparison and statistical analysis.

        Args:
            algorithm_configs: List of algorithm configuration dictionaries.
                              Each dict must have 'class' key with algorithm class,
                              and can have additional parameters as keys.
                              Example: [
                                  {'class': NelderMead, 'name': 'NM', 'delta_r': 1.0},
                                  {'class': AdaptiveNelderMead, 'name': 'ANM'}
                              ]
            problems: List of optimization problems to test on
            num_runs: Number of independent runs per algorithm-problem pair
            seeds: List of random seeds (if None, uses range(1, num_runs+1))
            success_tolerance: Tolerance for determining if optimum was reached

        Returns:
            Nested dictionary: {algorithm_name: {problem_name: [results]}}
            where results is a list of ExperimentResult objects

        Raises:
            ValueError: If algorithm_configs is empty or problems is empty
            KeyError: If algorithm config doesn't have 'class' key

        Example:
            >>> results = runner.compare_algorithms(
            ...     algorithm_configs=[
            ...         {'class': NelderMead, 'name': 'NM', 'max_fes': 5000},
            ...         {'class': AdaptiveNelderMead, 'name': 'ANM', 'max_fes': 5000}
            ...     ],
            ...     problems=[Sphere(dim=10), Rosenbrock(dim=10)],
            ...     num_runs=30
            ... )
            >>> # Access results: results['NM']['Sphere'] gives list of 30 ExperimentResult objects

        Validates: Requirements 15.1, 15.2, 15.4
        """
        # Validate inputs
        if not algorithm_configs:
            raise ValueError("algorithm_configs cannot be empty")

        if not problems:
            raise ValueError("problems cannot be empty")

        for config in algorithm_configs:
            if "class" not in config:
                raise KeyError("Each algorithm config must have 'class' key")

        # Generate seeds if not provided
        if seeds is None:
            seeds = list(range(1, num_runs + 1))

        # Print comparison info
        if self.verbose:
            print(f"\n{'='*70}")
            print("Algorithm Comparison")
            print(f"{'='*70}")
            print(
                f"Algorithms: {[cfg.get('name', cfg['class'].__name__) for cfg in algorithm_configs]}"
            )
            print(f"Problems: {[p.name for p in problems]}")
            print(f"Runs per combination: {num_runs}")
            print(
                f"Total experiments: {len(algorithm_configs) * len(problems) * num_runs}"
            )
            print(f"{'='*70}\n")

        # Run all experiments
        all_results = {}

        for alg_config in algorithm_configs:
            # Extract algorithm class and parameters
            alg_class = alg_config["class"]
            alg_name = alg_config.get("name", alg_class.__name__)

            # Get algorithm-specific parameters (exclude 'class' and 'name')
            alg_params = {
                k: v for k, v in alg_config.items() if k not in ["class", "name"]
            }

            # Initialize results dictionary for this algorithm
            all_results[alg_name] = {}

            # Run on each problem
            for problem in problems:
                if self.verbose:
                    print(f"\n--- {alg_name} on {problem.name} ---")

                # Run experiment
                results = self.run_experiment(
                    algorithm_class=alg_class,
                    problem=problem,
                    num_runs=num_runs,
                    seeds=seeds,
                    success_tolerance=success_tolerance,
                    **alg_params,
                )

                # Store results
                all_results[alg_name][problem.name] = results

        # Print overall summary
        if self.verbose:
            self._print_comparison_summary(all_results, problems)

        return all_results

    def _print_experiment_summary(
        self, results: List[ExperimentResult], problem: OptimizationProblem
    ) -> None:
        """
        Print summary statistics for a single experiment.

        Args:
            results: List of experiment results
            problem: The problem that was solved
        """
        # Extract fitness values (excluding failed runs)
        fitness_values = [
            r.best_fitness for r in results if np.isfinite(r.best_fitness)
        ]

        if not fitness_values:
            print("\nAll runs failed!")
            return

        # Compute statistics
        mean_fitness = np.mean(fitness_values)
        median_fitness = np.median(fitness_values)
        std_fitness = np.std(fitness_values)
        best_fitness = np.min(fitness_values)
        worst_fitness = np.max(fitness_values)

        # Success rate
        num_success = sum(r.success for r in results)
        success_rate = num_success / len(results) * 100

        # Average FES and time
        avg_fes = np.mean([r.fes_used for r in results])
        avg_time = np.mean([r.wall_time for r in results])

        # Print summary
        print(f"\n{'-'*70}")
        print("Experiment Summary")
        print(f"{'-'*70}")
        print("Fitness Statistics:")
        print(f"  Mean:   {mean_fitness:.6e}")
        print(f"  Median: {median_fitness:.6e}")
        print(f"  Std:    {std_fitness:.6e}")
        print(f"  Best:   {best_fitness:.6e}")
        print(f"  Worst:  {worst_fitness:.6e}")

        if problem.optimum_value is not None:
            error = abs(best_fitness - problem.optimum_value)
            print(f"  Error from optimum: {error:.6e}")

        print(f"\nSuccess Rate: {num_success}/{len(results)} ({success_rate:.1f}%)")
        print(f"Average FES: {avg_fes:.0f}")
        print(f"Average Time: {avg_time:.2f}s")
        print(f"{'-'*70}\n")

    def _print_comparison_summary(
        self,
        all_results: Dict[str, Dict[str, List[ExperimentResult]]],
        problems: List[OptimizationProblem],
    ) -> None:
        """
        Print summary table comparing all algorithms on all problems.

        Args:
            all_results: Nested dictionary of results
            problems: List of problems
        """
        print(f"\n{'='*70}")
        print("Comparison Summary")
        print(f"{'='*70}\n")

        # Create summary table
        print(
            f"{'Algorithm':<20} {'Problem':<20} {'Mean Fitness':<15} {'Success Rate':<15}"
        )
        print(f"{'-'*70}")

        for alg_name, problem_results in all_results.items():
            for problem in problems:
                if problem.name in problem_results:
                    results = problem_results[problem.name]

                    # Compute statistics
                    fitness_values = [
                        r.best_fitness for r in results if np.isfinite(r.best_fitness)
                    ]
                    mean_fitness = np.mean(fitness_values) if fitness_values else np.inf

                    num_success = sum(r.success for r in results)
                    success_rate = num_success / len(results) * 100

                    print(
                        f"{alg_name:<20} {problem.name:<20} {mean_fitness:<15.6e} "
                        f"{success_rate:<15.1f}%"
                    )

        print(f"{'='*70}\n")

    def export_results_to_dict(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """
        Export experiment results to a dictionary format suitable for JSON/CSV export.

        Args:
            results: List of experiment results

        Returns:
            Dictionary with structured results including statistics

        Validates: Requirements 15.2, 15.3, 15.5
        """
        # Extract fitness values
        fitness_values = [
            r.best_fitness for r in results if np.isfinite(r.best_fitness)
        ]

        if not fitness_values:
            return {
                "algorithm": results[0].algorithm_name if results else "Unknown",
                "problem": results[0].problem_name if results else "Unknown",
                "num_runs": len(results),
                "statistics": {},
                "runs": [],
            }

        # Compute statistics
        statistics = {
            "mean": float(np.mean(fitness_values)),
            "median": float(np.median(fitness_values)),
            "std": float(np.std(fitness_values)),
            "best": float(np.min(fitness_values)),
            "worst": float(np.max(fitness_values)),
            "success_rate": sum(r.success for r in results) / len(results),
            "avg_fes": float(np.mean([r.fes_used for r in results])),
            "avg_time": float(np.mean([r.wall_time for r in results])),
        }

        # Export individual runs
        runs = []
        for r in results:
            run_data = {
                "seed": r.seed,
                "best_fitness": (
                    float(r.best_fitness) if np.isfinite(r.best_fitness) else None
                ),
                "best_violation": (
                    float(r.best_violation) if np.isfinite(r.best_violation) else None
                ),
                "fes_used": r.fes_used,
                "wall_time": r.wall_time,
                "success": r.success,
                "best_solution": (
                    r.best_solution.tolist() if r.best_solution is not None else None
                ),
                "convergence_history": [
                    (int(fes), float(fit)) for fes, fit in r.convergence_history
                ],
            }
            runs.append(run_data)

        return {
            "algorithm": results[0].algorithm_name,
            "problem": results[0].problem_name,
            "num_runs": len(results),
            "statistics": statistics,
            "runs": runs,
        }
