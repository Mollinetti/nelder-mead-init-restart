#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result Analysis Infrastructure for Optimization Experiments

This module provides tools for analyzing and visualizing results from
batch optimization experiments. It supports:
- Computing statistical summaries (mean, median, std, best, worst)
- Generating convergence plots for algorithm comparison
- Generating publication-ready LaTeX comparison tables

References:
    - Standard practice for reporting optimization algorithm results
    - Statistical analysis of stochastic optimization algorithms
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .batch_runner import ExperimentResult


class ResultAnalyzer:
    """
    Statistical analysis and visualization tools for optimization experiment results.

    This class provides methods for:
    1. Computing statistical summaries across multiple runs
    2. Generating convergence plots for visual comparison
    3. Generating LaTeX tables for publication

    Example:
        >>> from nelder_mead.testing import BatchRunner, ResultAnalyzer
        >>> from nelder_mead.algorithms.nelder_mead import NelderMead
        >>> from nelder_mead.benchmarks.unconstrained import Sphere
        >>>
        >>> runner = BatchRunner()
        >>> problem = Sphere(dim=10)
        >>> results = runner.run_experiment(
        ...     algorithm_class=NelderMead,
        ...     problem=problem,
        ...     num_runs=30,
        ...     max_fes=5000
        ... )
        >>>
        >>> analyzer = ResultAnalyzer()
        >>> stats = analyzer.compute_statistics(results)
        >>> print(f"Mean: {stats['mean']:.6e}, Std: {stats['std']:.6e}")

    Validates: Requirements 15.3, 15.5
    """

    @staticmethod
    def compute_statistics(
        results: List[ExperimentResult], exclude_failed: bool = True
    ) -> Dict[str, float]:
        """
        Compute statistical summaries across multiple experiment runs.

        This method computes standard statistical measures for the best fitness
        values obtained across multiple independent runs of an algorithm.

        Args:
            results: List of ExperimentResult objects from multiple runs
            exclude_failed: If True, exclude runs with infinite or NaN fitness

        Returns:
            Dictionary containing:
                - mean: Mean of best fitness values
                - median: Median of best fitness values
                - std: Standard deviation of best fitness values
                - best: Best (minimum) fitness value found
                - worst: Worst (maximum) fitness value found
                - success_rate: Fraction of successful runs (0.0 to 1.0)
                - avg_fes: Average number of function evaluations used
                - avg_time: Average wall clock time in seconds
                - num_runs: Total number of runs
                - num_valid: Number of valid (non-failed) runs

        Raises:
            ValueError: If results list is empty
            ValueError: If all runs failed and exclude_failed is True

        Example:
            >>> stats = analyzer.compute_statistics(results)
            >>> print(f"Mean ± Std: {stats['mean']:.6e} ± {stats['std']:.6e}")
            >>> print(f"Best: {stats['best']:.6e}, Worst: {stats['worst']:.6e}")
            >>> print(f"Success rate: {stats['success_rate']*100:.1f}%")

        Validates: Requirements 15.3, 15.5
        """
        if not results:
            raise ValueError("Results list cannot be empty")

        # Extract fitness values
        if exclude_failed:
            fitness_values = [
                r.best_fitness for r in results if np.isfinite(r.best_fitness)
            ]
        else:
            fitness_values = [r.best_fitness for r in results]

        if not fitness_values:
            raise ValueError("All runs failed - no valid fitness values to analyze")

        # Compute fitness statistics
        mean_fitness = float(np.mean(fitness_values))
        median_fitness = float(np.median(fitness_values))

        # Handle standard deviation for single value case
        if len(fitness_values) > 1:
            std_fitness = float(np.std(fitness_values, ddof=1))  # Sample std deviation
        else:
            std_fitness = 0.0  # Single value has zero standard deviation

        best_fitness = float(np.min(fitness_values))
        worst_fitness = float(np.max(fitness_values))

        # Compute success rate
        num_success = sum(r.success for r in results)
        success_rate = float(num_success) / len(results)

        # Compute average FES and time (across all runs, including failed)
        avg_fes = float(np.mean([r.fes_used for r in results]))
        avg_time = float(np.mean([r.wall_time for r in results]))

        return {
            "mean": mean_fitness,
            "median": median_fitness,
            "std": std_fitness,
            "best": best_fitness,
            "worst": worst_fitness,
            "success_rate": success_rate,
            "avg_fes": avg_fes,
            "avg_time": avg_time,
            "num_runs": len(results),
            "num_valid": len(fitness_values),
        }

    @staticmethod
    def deb_comparison_values(
        results_by_algorithm: Dict[str, List[ExperimentResult]],
        feasibility_tol: float = 1e-8,
    ) -> Dict[str, np.ndarray]:
        """
        Map runs to rank-equivalent scalars that honour feasibility.

        Comparing constrained-optimization runs on the raw objective is wrong, and
        wrong in a way that flatters bad results: a run that never reached the
        feasible region can report an objective below the true constrained optimum,
        because it is minimizing over a larger set. Ranking on that number would
        declare the most infeasible algorithm the winner.

        This applies Deb's rules, the standard for constrained benchmarking:

        1. A feasible run always beats an infeasible one.
        2. Two feasible runs are ordered by objective.
        3. Two infeasible runs are ordered by constraint violation.

        A fourth category is added for runs that failed outright and recorded a
        non-finite objective. These rank strictly worst. Dropping them instead,
        which is the obvious alternative, silently flatters an algorithm that
        crashes: its surviving runs are its lucky ones, and its sample shrinks to
        just those.

        The rules are folded into a single scalar per run, by stacking the
        categories above one another:

            value = objective                      if feasible and finite
            value = offset + violation             if infeasible and finite
            value = offset + max_violation + 1     if non-finite

        where ``offset`` exceeds the largest feasible objective in the comparison.
        The result is not meaningful as a number; it is meaningful only as a rank,
        which is exactly what Mann-Whitney U and Friedman consume.

        Args:
            results_by_algorithm: Dict {algorithm: [results]} for a single problem
            feasibility_tol: Total violation below which a run counts as feasible

        Returns:
            Dict {algorithm: array of comparison values}, in run order

        Example:
            >>> values = ResultAnalyzer.deb_comparison_values(results)  # doctest: +SKIP
        """
        feasible_objectives = [
            r.best_fitness
            for runs in results_by_algorithm.values()
            for r in runs
            if r.best_violation <= feasibility_tol and np.isfinite(r.best_fitness)
        ]

        # The offset must sit strictly above every feasible objective so that no
        # infeasible run can rank ahead of a feasible one.
        offset = max(feasible_objectives) + 1.0 if feasible_objectives else 0.0

        # Failed runs rank above every completed run, feasible or not.
        finite_violations = [
            max(r.best_violation, 0.0)
            for runs in results_by_algorithm.values()
            for r in runs
            if np.isfinite(r.best_fitness) and np.isfinite(r.best_violation)
        ]
        failure_value = offset + (max(finite_violations) if finite_violations else 0.0) + 1.0

        values = {}
        for algorithm, runs in results_by_algorithm.items():
            column = []
            for run in runs:
                if not np.isfinite(run.best_fitness) or not np.isfinite(
                    run.best_violation
                ):
                    column.append(failure_value)
                elif run.best_violation <= feasibility_tol:
                    column.append(run.best_fitness)
                else:
                    column.append(offset + max(run.best_violation, 0.0))
            values[algorithm] = np.array(column, dtype=float)

        return values

    @staticmethod
    def friedman_test(
        results_by_algorithm: Dict[str, Dict[str, List[ExperimentResult]]],
        statistic: str = "mean",
        feasibility_aware: bool = True,
        feasibility_tol: float = 1e-8,
    ) -> Dict[str, Any]:
        """
        Friedman omnibus test for differences among algorithms across problems.

        The Friedman test asks a single question: over the whole problem set, do
        the algorithms differ at all? It ranks the algorithms within each problem
        and tests whether the mean ranks depart from what random ordering would
        produce. It is the non-parametric counterpart of a repeated-measures
        ANOVA, which matters here because optimization results are neither normal
        nor homoscedastic across problems.

        It is an omnibus test, so a significant result says only that *some*
        difference exists. Use :meth:`mannwhitney_vs_reference` afterwards to say
        which pairs differ, and correct those p-values with :meth:`holm_correction`.

        Args:
            results_by_algorithm: Nested dict {algorithm: {problem: [results]}},
                                  as returned by BatchRunner.compare_algorithms
            statistic: Per-problem summary to rank, "mean" or "median"
            feasibility_aware: If True (the default), rank on Deb comparison values
                               rather than on the raw objective, so that an
                               infeasible run cannot outrank a feasible one. See
                               :meth:`deb_comparison_values`.
            feasibility_tol: Total violation below which a run counts as feasible.
                             The CEC constrained-optimization protocols report at
                             1e-4, because driving equality constraints to machine
                             precision with a penalty method is not achievable
                             within a realistic evaluation budget.

        Returns:
            Dict with:
                - statistic: Friedman chi-squared statistic
                - p_value: p-value of the test
                - mean_ranks: {algorithm: mean rank across problems}, lower is better
                - num_problems: number of problems the test ran over
                - algorithms: algorithm names in the order they were ranked

        Raises:
            ValueError: If fewer than three algorithms are given (the test is
                        undefined for two; use Mann-Whitney U directly), if the
                        algorithms do not share a common problem set, or if
                        `statistic` is not "mean" or "median"

        Example:
            >>> outcome = ResultAnalyzer.friedman_test(results)   # doctest: +SKIP
            >>> outcome["p_value"] < 0.05                          # doctest: +SKIP
            True
        """
        if statistic not in ("mean", "median"):
            raise ValueError(
                f"statistic must be 'mean' or 'median', got '{statistic}'"
            )

        algorithms = list(results_by_algorithm.keys())
        if len(algorithms) < 3:
            raise ValueError(
                "Friedman test requires at least 3 algorithms, got "
                f"{len(algorithms)}. For two algorithms use mannwhitney_vs_reference."
            )

        problems = list(results_by_algorithm[algorithms[0]].keys())
        for alg in algorithms:
            if set(results_by_algorithm[alg].keys()) != set(problems):
                raise ValueError(
                    f"Algorithm '{alg}' was not run on the same problem set as "
                    f"'{algorithms[0]}'. The Friedman test requires complete blocks."
                )

        if len(problems) < 2:
            raise ValueError(
                f"Friedman test requires at least 2 problems, got {len(problems)}"
            )

        reduce = np.mean if statistic == "mean" else np.median

        # Rows are problems (blocks), columns are algorithms (treatments). Each
        # problem is reduced independently, which is what makes the feasibility
        # offset safe: values are only ever compared within a row.
        rows = []
        for problem in problems:
            per_problem = {
                alg: results_by_algorithm[alg][problem] for alg in algorithms
            }
            if feasibility_aware:
                columns = ResultAnalyzer.deb_comparison_values(
                    per_problem, feasibility_tol=feasibility_tol
                )
                rows.append([reduce(columns[alg]) for alg in algorithms])
            else:
                rows.append(
                    [
                        reduce(
                            [
                                r.best_fitness
                                for r in per_problem[alg]
                                if np.isfinite(r.best_fitness)
                            ]
                            or [np.inf]
                        )
                        for alg in algorithms
                    ]
                )
        matrix = np.array(rows)

        from scipy.stats import friedmanchisquare, rankdata

        chi2, p_value = friedmanchisquare(*matrix.T)

        # Rank within each problem; ties share the average rank, and rank 1 is best
        ranks = np.array([rankdata(row) for row in matrix])
        mean_ranks = dict(zip(algorithms, ranks.mean(axis=0)))

        return {
            "statistic": float(chi2),
            "p_value": float(p_value),
            "mean_ranks": mean_ranks,
            "num_problems": len(problems),
            "algorithms": algorithms,
        }

    @staticmethod
    def mannwhitney_vs_reference(
        results_by_problem: Dict[str, List[ExperimentResult]],
        reference: str,
        alternative: str = "two-sided",
        feasibility_aware: bool = True,
        feasibility_tol: float = 1e-8,
    ) -> Dict[str, Dict[str, float]]:
        """
        Pairwise Mann-Whitney U tests of each algorithm against a reference.

        For one problem, this compares the distribution of final objective values
        the reference algorithm produced over its independent runs against each
        competitor's distribution. Mann-Whitney U is a rank test on two independent
        samples, which is the right choice here: separate runs of two algorithms are
        independent samples, and their objective values are not normally distributed.

        This produces the layout of Table 4 in the paper: one column per competitor,
        each cell a p-value against the reference algorithm.

        Args:
            results_by_problem: Dict {algorithm: [results]} for a single problem
            reference: Name of the reference algorithm to compare everything against
            alternative: "two-sided", "less" or "greater". "two-sided" asks whether
                         the distributions differ at all; "less" asks whether the
                         reference achieves lower (better) values than the competitor
            feasibility_aware: If True (the default), compare Deb comparison values
                               rather than raw objectives, so an infeasible run
                               cannot rank ahead of a feasible one. See
                               :meth:`deb_comparison_values`.
            feasibility_tol: Total violation below which a run counts as feasible.
                             The CEC constrained-optimization protocols report at
                             1e-4.

        Returns:
            Dict {algorithm: {"statistic": U, "p_value": p, "n_reference": int,
            "n_other": int}} for every algorithm except the reference

        Raises:
            ValueError: If the reference is absent, if it is the only algorithm
                        present, or if any algorithm has no finite results

        Example:
            >>> pvals = ResultAnalyzer.mannwhitney_vs_reference(  # doctest: +SKIP
            ...     results["TBT"], reference="cNM"
            ... )
        """
        if reference not in results_by_problem:
            raise ValueError(
                f"Reference algorithm '{reference}' not found. Available: "
                f"{list(results_by_problem.keys())}"
            )

        others = [a for a in results_by_problem if a != reference]
        if not others:
            raise ValueError(
                f"Only the reference algorithm '{reference}' was provided; "
                "there is nothing to compare it against."
            )

        comparison_columns = (
            ResultAnalyzer.deb_comparison_values(
                results_by_problem, feasibility_tol=feasibility_tol
            )
            if feasibility_aware
            else None
        )

        def finite_fitness(algorithm: str) -> np.ndarray:
            if comparison_columns is not None:
                values = comparison_columns[algorithm]
                values = values[np.isfinite(values)]
            else:
                values = np.array(
                    [
                        r.best_fitness
                        for r in results_by_problem[algorithm]
                        if np.isfinite(r.best_fitness)
                    ]
                )
            if values.size == 0:
                raise ValueError(
                    f"Algorithm '{algorithm}' has no finite results to test."
                )
            return values

        from scipy.stats import mannwhitneyu

        reference_values = finite_fitness(reference)
        comparisons = {}

        for algorithm in others:
            other_values = finite_fitness(algorithm)

            # Identical constant samples (e.g. every run hit the same optimum) make
            # the U statistic degenerate; scipy raises rather than returning p=1.
            if np.all(reference_values == reference_values[0]) and np.all(
                other_values == reference_values[0]
            ):
                comparisons[algorithm] = {
                    "statistic": float(
                        reference_values.size * other_values.size / 2.0
                    ),
                    "p_value": 1.0,
                    "n_reference": int(reference_values.size),
                    "n_other": int(other_values.size),
                }
                continue

            u_statistic, p_value = mannwhitneyu(
                reference_values, other_values, alternative=alternative
            )
            comparisons[algorithm] = {
                "statistic": float(u_statistic),
                "p_value": float(p_value),
                "n_reference": int(reference_values.size),
                "n_other": int(other_values.size),
            }

        return comparisons

    @staticmethod
    def holm_correction(pvalues: Dict[str, float]) -> Dict[str, float]:
        """
        Holm-Bonferroni correction for a family of pairwise p-values.

        Testing one reference algorithm against k competitors runs k hypotheses at
        once, so the chance of at least one spurious "significant" result grows with
        k. Holm's step-down procedure controls the family-wise error rate: sort the
        p-values ascending and multiply the i-th by (k - i), then enforce
        monotonicity so an adjusted value never falls below the one before it.

        Holm is uniformly more powerful than plain Bonferroni and makes no
        assumption about dependence between the tests, so it is the appropriate
        default for reporting a table of pairwise comparisons.

        Args:
            pvalues: Dict {algorithm: raw p-value}

        Returns:
            Dict {algorithm: adjusted p-value}, each clipped to at most 1.0

        Raises:
            ValueError: If any p-value lies outside [0, 1]

        Example:
            >>> raw = {"a": 0.01, "b": 0.04, "c": 0.03}
            >>> adjusted = ResultAnalyzer.holm_correction(raw)
            >>> round(adjusted["a"], 4)
            0.03
            >>> round(adjusted["b"], 4)
            0.06
        """
        if not pvalues:
            return {}

        for name, value in pvalues.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"p-value for '{name}' is {value}, outside the range [0, 1]"
                )

        names = list(pvalues.keys())
        raw = np.array([pvalues[n] for n in names], dtype=float)
        num_tests = raw.size

        order = np.argsort(raw)
        adjusted_sorted = raw[order] * (num_tests - np.arange(num_tests))

        # Step-down: an adjusted p-value can never decrease along the sorted order
        adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted

        return dict(zip(names, adjusted.tolist()))

    @staticmethod
    def generate_convergence_plots(
        results_dict: Dict[str, List[ExperimentResult]],
        problem_name: str,
        output_path: Optional[str] = None,
        show_plot: bool = True,
        log_scale: bool = True,
        plot_median: bool = True,
        plot_quartiles: bool = True,
        figsize: Tuple[int, int] = (10, 6),
    ) -> Optional[Any]:
        """
        Generate convergence plots comparing multiple algorithms.

        This method creates a plot showing how the best fitness value evolves
        over function evaluations for different algorithms. It can show median
        convergence curves with quartile bands for statistical robustness.

        Args:
            results_dict: Dictionary mapping algorithm names to lists of ExperimentResult
                         Example: {'NM': [result1, result2, ...], 'ANM': [...]}
            problem_name: Name of the problem (used in plot title)
            output_path: If provided, save plot to this file path (e.g., 'plot.png', 'plot.pdf')
            show_plot: If True, display the plot interactively (requires display)
            log_scale: If True, use logarithmic scale for y-axis (fitness)
            plot_median: If True, plot median convergence curve
            plot_quartiles: If True, plot 25th and 75th percentile bands
            figsize: Figure size as (width, height) in inches

        Returns:
            matplotlib Figure object if matplotlib is available, None otherwise

        Raises:
            ValueError: If results_dict is empty
            ImportError: If matplotlib is not installed (only when called)

        Example:
            >>> results_dict = {
            ...     'NM': nm_results,
            ...     'ANM': anm_results
            ... }
            >>> analyzer.generate_convergence_plots(
            ...     results_dict=results_dict,
            ...     problem_name='Sphere',
            ...     output_path='convergence_sphere.pdf',
            ...     log_scale=True
            ... )

        Note:
            This method requires matplotlib to be installed. If matplotlib is not
            available, it will raise an ImportError with installation instructions.

        Validates: Requirements 15.5
        """
        if not results_dict:
            raise ValueError("results_dict cannot be empty")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install it with: pip install matplotlib"
            )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot convergence for each algorithm
        for alg_name, results in results_dict.items():
            # Extract convergence histories
            convergence_histories = []
            for result in results:
                if result.convergence_history:
                    # Convert to arrays: fes and fitness
                    fes = np.array([entry[0] for entry in result.convergence_history])
                    fitness = np.array(
                        [entry[1] for entry in result.convergence_history]
                    )
                    convergence_histories.append((fes, fitness))

            if not convergence_histories:
                continue

            # Find common FES grid (use the maximum range)
            max_fes = max(fes[-1] for fes, _ in convergence_histories)
            min_fes = min(fes[0] for fes, _ in convergence_histories)

            # Create a common FES grid for interpolation
            num_points = 100
            common_fes = np.linspace(min_fes, max_fes, num_points)

            # Interpolate all histories to common grid
            interpolated_fitness = []
            for fes, fitness in convergence_histories:
                # Use linear interpolation with forward fill for monotonic best fitness
                interp_fitness = np.interp(common_fes, fes, fitness)
                interpolated_fitness.append(interp_fitness)

            interpolated_fitness = np.array(interpolated_fitness)

            # Compute statistics across runs
            if plot_median:
                median_fitness = np.median(interpolated_fitness, axis=0)
                ax.plot(common_fes, median_fitness, label=alg_name, linewidth=2)
            else:
                mean_fitness = np.mean(interpolated_fitness, axis=0)
                ax.plot(common_fes, mean_fitness, label=alg_name, linewidth=2)

            # Plot quartile bands
            if plot_quartiles and len(interpolated_fitness) > 1:
                q25 = np.percentile(interpolated_fitness, 25, axis=0)
                q75 = np.percentile(interpolated_fitness, 75, axis=0)
                ax.fill_between(common_fes, q25, q75, alpha=0.2)

        # Configure plot
        ax.set_xlabel("Function Evaluations", fontsize=12)
        ax.set_ylabel("Best Fitness Value", fontsize=12)
        ax.set_title(f"Convergence on {problem_name}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if log_scale:
            ax.set_yscale("log")

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_path}")

        # Show if requested
        if show_plot:
            plt.show()

        return fig

    @staticmethod
    def generate_comparison_table(
        results_dict: Dict[str, Dict[str, List[ExperimentResult]]],
        problem_names: List[str],
        output_path: Optional[str] = None,
        format: str = "latex",
        include_std: bool = True,
        include_success_rate: bool = True,
        scientific_notation: bool = True,
        precision: int = 2,
    ) -> str:
        """
        Generate a comparison table in LaTeX or Markdown format for publication.

        This method creates a formatted table comparing multiple algorithms across
        multiple problems, showing mean fitness, standard deviation, and success rates.
        The table is suitable for inclusion in academic papers.

        Args:
            results_dict: Nested dictionary: {algorithm_name: {problem_name: [results]}}
                         Example: {'NM': {'Sphere': [r1, r2, ...], 'Rosenbrock': [...]}}
            problem_names: List of problem names to include in table (in order)
            output_path: If provided, save table to this file path
            format: Output format - 'latex' or 'markdown'
            include_std: If True, show standard deviation alongside mean
            include_success_rate: If True, include success rate column
            scientific_notation: If True, use scientific notation for fitness values
            precision: Number of decimal places for fitness values

        Returns:
            String containing the formatted table

        Raises:
            ValueError: If results_dict is empty or problem_names is empty
            ValueError: If format is not 'latex' or 'markdown'

        Example:
            >>> results = {
            ...     'NM': {'Sphere': nm_sphere_results, 'Rosenbrock': nm_rosen_results},
            ...     'ANM': {'Sphere': anm_sphere_results, 'Rosenbrock': anm_rosen_results}
            ... }
            >>> table = analyzer.generate_comparison_table(
            ...     results_dict=results,
            ...     problem_names=['Sphere', 'Rosenbrock'],
            ...     output_path='comparison_table.tex',
            ...     format='latex'
            ... )
            >>> print(table)

        LaTeX Example Output:
            \\begin{table}[h]
            \\centering
            \\caption{Algorithm Comparison Results}
            \\begin{tabular}{lllll}
            \\hline
            Algorithm & Problem & Mean ± Std & Best & Success Rate \\\\
            \\hline
            NM & Sphere & 1.23e-05 ± 4.56e-06 & 3.21e-06 & 100.0\\% \\\\
            ...
            \\hline
            \\end{tabular}
            \\end{table}

        Validates: Requirements 15.5
        """
        if not results_dict:
            raise ValueError("results_dict cannot be empty")

        if not problem_names:
            raise ValueError("problem_names cannot be empty")

        if format not in ["latex", "markdown"]:
            raise ValueError("format must be 'latex' or 'markdown'")

        # Build table content
        table_lines = []

        if format == "latex":
            # LaTeX table header
            table_lines.append("\\begin{table}[h]")
            table_lines.append("\\centering")
            table_lines.append("\\caption{Algorithm Comparison Results}")

            # Determine columns
            columns = "lll"  # Algorithm, Problem, Mean±Std
            if include_success_rate:
                columns += "l"  # Success Rate

            table_lines.append(f"\\begin{{tabular}}{{{columns}}}")
            table_lines.append("\\hline")

            # Header row
            header = "Algorithm & Problem & "
            if include_std:
                header += "Mean $\\pm$ Std & Best"
            else:
                header += "Mean & Best"
            if include_success_rate:
                header += " & Success Rate"
            header += " \\\\"
            table_lines.append(header)
            table_lines.append("\\hline")

        else:  # markdown
            # Markdown table header
            header = "| Algorithm | Problem | "
            if include_std:
                header += "Mean ± Std | Best"
            else:
                header += "Mean | Best"
            separator = "|-----------|---------|"
            if include_std:
                separator += "------------|------"
            else:
                separator += "------|------"
            if include_success_rate:
                header += " | Success Rate"
                separator += "|-------------"
            header += " |"
            separator += "|"
            table_lines.append(header)
            table_lines.append(separator)

        # Compute statistics for each algorithm-problem combination
        analyzer = ResultAnalyzer()

        for alg_name in sorted(results_dict.keys()):
            problem_results = results_dict[alg_name]

            for problem_name in problem_names:
                if problem_name not in problem_results:
                    continue

                results = problem_results[problem_name]

                try:
                    stats = analyzer.compute_statistics(results)

                    # Format fitness values
                    if scientific_notation:
                        mean_str = f"{stats['mean']:.{precision}e}"
                        std_str = f"{stats['std']:.{precision}e}"
                        best_str = f"{stats['best']:.{precision}e}"
                    else:
                        mean_str = f"{stats['mean']:.{precision}f}"
                        std_str = f"{stats['std']:.{precision}f}"
                        best_str = f"{stats['best']:.{precision}f}"

                    if include_std:
                        fitness_str = (
                            f"{mean_str} $\\pm$ {std_str}"
                            if format == "latex"
                            else f"{mean_str} ± {std_str}"
                        )
                    else:
                        fitness_str = mean_str

                    # Format success rate
                    success_str = (
                        f"{stats['success_rate']*100:.1f}\\%"
                        if format == "latex"
                        else f"{stats['success_rate']*100:.1f}%"
                    )

                    # Build row
                    if format == "latex":
                        row = (
                            f"{alg_name} & {problem_name} & {fitness_str} & {best_str}"
                        )
                        if include_success_rate:
                            row += f" & {success_str}"
                        row += " \\\\"
                    else:  # markdown
                        row = f"| {alg_name} | {problem_name} | {fitness_str} | {best_str}"
                        if include_success_rate:
                            row += f" | {success_str}"
                        row += " |"

                    table_lines.append(row)

                except ValueError:
                    # Handle case where all runs failed
                    if format == "latex":
                        row = f"{alg_name} & {problem_name} & Failed & Failed"
                        if include_success_rate:
                            row += " & 0.0\\%"
                        row += " \\\\"
                    else:
                        row = f"| {alg_name} | {problem_name} | Failed | Failed"
                        if include_success_rate:
                            row += " | 0.0%"
                        row += " |"
                    table_lines.append(row)

        # Table footer
        if format == "latex":
            table_lines.append("\\hline")
            table_lines.append("\\end{tabular}")
            table_lines.append("\\label{tab:comparison}")
            table_lines.append("\\end{table}")

        # Join lines
        table_str = "\n".join(table_lines)

        # Save to file if requested
        if output_path:
            with open(output_path, "w") as f:
                f.write(table_str)
            print(f"Table saved to: {output_path}")

        return table_str
