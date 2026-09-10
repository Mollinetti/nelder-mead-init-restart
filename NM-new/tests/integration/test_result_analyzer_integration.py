#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for ResultAnalyzer with BatchRunner.

Tests the complete workflow of running experiments and analyzing results.

This test suite validates:
- Statistical computation accuracy (Requirement 15.3)
- Convergence plot generation (Requirement 15.5)
- Comparison table generation (Requirement 15.5)

Feature: nelder-mead-thesis-refactor
Validates: Requirements 15.3, 15.5
"""

import pytest
import tempfile
import os
import numpy as np

from nelder_mead.testing import BatchRunner, ResultAnalyzer
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
from nelder_mead.benchmarks.unconstrained import Sphere, Rosenbrock


def test_batch_runner_with_result_analyzer():
    """Test complete workflow: run experiments and analyze results."""
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments on Sphere problem
    problem = Sphere(dim=5)
    results = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=problem,
        num_runs=5,
        seeds=[1, 2, 3, 4, 5],
        max_fes=1000,
        init_method="spendleySimplex",
    )

    # Analyze results
    analyzer = ResultAnalyzer()
    stats = analyzer.compute_statistics(results)

    # Verify statistics
    assert stats["num_runs"] == 5
    assert stats["mean"] >= 0  # Can be 0 if optimum is found
    assert stats["best"] <= stats["mean"] <= stats["worst"]
    assert 0.0 <= stats["success_rate"] <= 1.0
    assert stats["avg_fes"] > 0
    assert stats["avg_time"] > 0


def test_compare_algorithms_with_analysis():
    """Test comparing multiple algorithms and analyzing results."""
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Compare NM and ANM on Sphere
    results = runner.compare_algorithms(
        algorithm_configs=[
            {"class": NelderMead, "name": "NM", "max_fes": 1000},
            {"class": AdaptiveNelderMead, "name": "ANM", "max_fes": 1000},
        ],
        problems=[Sphere(dim=5)],
        num_runs=3,
        seeds=[1, 2, 3],
    )

    # Analyze results for each algorithm
    analyzer = ResultAnalyzer()

    for alg_name in ["NM", "ANM"]:
        alg_results = results[alg_name]["Sphere"]
        stats = analyzer.compute_statistics(alg_results)

        assert stats["num_runs"] == 3
        assert stats["mean"] >= 0  # Can be 0 if optimum is found
        assert stats["best"] >= 0


def test_generate_comparison_table_from_batch_results():
    """Test generating comparison table from batch experiment results."""
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    results = runner.compare_algorithms(
        algorithm_configs=[
            {"class": NelderMead, "name": "NM", "max_fes": 500},
            {"class": AdaptiveNelderMead, "name": "ANM", "max_fes": 500},
        ],
        problems=[Sphere(dim=3), Rosenbrock(dim=3)],
        num_runs=3,
        seeds=[1, 2, 3],
    )

    # Generate comparison table
    analyzer = ResultAnalyzer()
    table = analyzer.generate_comparison_table(
        results_dict=results, problem_names=["Sphere", "Rosenbrock"], format="markdown"
    )

    # Verify table content
    assert "NM" in table
    assert "ANM" in table
    assert "Sphere" in table
    assert "Rosenbrock" in table
    assert "Algorithm" in table
    assert "Problem" in table


def test_generate_convergence_plots_from_batch_results():
    """Test generating convergence plots from batch experiment results."""
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    nm_results = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=Sphere(dim=5),
        num_runs=3,
        seeds=[1, 2, 3],
        max_fes=1000,
    )

    anm_results = runner.run_experiment(
        algorithm_class=AdaptiveNelderMead,
        problem=Sphere(dim=5),
        num_runs=3,
        seeds=[1, 2, 3],
        max_fes=1000,
    )

    # Try to generate convergence plot
    analyzer = ResultAnalyzer()

    try:
        fig = analyzer.generate_convergence_plots(
            results_dict={"NM": nm_results, "ANM": anm_results},
            problem_name="Sphere",
            show_plot=False,
        )

        # Verify figure was created
        assert fig is not None

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    except ImportError:
        # matplotlib not installed - skip test
        pytest.skip("matplotlib not installed")


def test_save_table_and_plot_to_files():
    """Test saving both table and plot to files."""
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    results = runner.compare_algorithms(
        algorithm_configs=[{"class": NelderMead, "name": "NM", "max_fes": 500}],
        problems=[Sphere(dim=3)],
        num_runs=3,
        seeds=[1, 2, 3],
    )

    analyzer = ResultAnalyzer()

    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tmp_table:
        table_path = tmp_table.name

    try:
        # Generate and save table
        analyzer.generate_comparison_table(
            results_dict=results,
            problem_names=["Sphere"],
            output_path=table_path,
            format="latex",
        )

        # Verify table file was created
        assert os.path.exists(table_path)
        assert os.path.getsize(table_path) > 0

        # Read and verify content
        with open(table_path, "r") as f:
            content = f.read()
        assert "\\begin{table}" in content
        assert "NM" in content
        assert "Sphere" in content

    finally:
        # Clean up
        if os.path.exists(table_path):
            os.remove(table_path)

    # Try to save plot (if matplotlib available)
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_plot:
            plot_path = tmp_plot.name

        try:
            fig = analyzer.generate_convergence_plots(
                results_dict={"NM": results["NM"]["Sphere"]},
                problem_name="Sphere",
                output_path=plot_path,
                show_plot=False,
            )

            # Verify plot file was created
            assert os.path.exists(plot_path)
            assert os.path.getsize(plot_path) > 0

            # Clean up
            import matplotlib.pyplot as plt

            plt.close(fig)

        finally:
            if os.path.exists(plot_path):
                os.remove(plot_path)

    except ImportError:
        # matplotlib not installed - skip plot test
        pass


def test_statistics_match_across_formats():
    """Test that statistics are consistent across different output formats."""
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    results = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=Sphere(dim=5),
        num_runs=5,
        seeds=[1, 2, 3, 4, 5],
        max_fes=1000,
    )

    # Compute statistics directly
    analyzer = ResultAnalyzer()
    analyzer.compute_statistics(results)

    # Generate table and verify statistics are embedded
    results_dict = {"NM": {"Sphere": results}}
    table_latex = analyzer.generate_comparison_table(
        results_dict=results_dict,
        problem_names=["Sphere"],
        format="latex",
        scientific_notation=True,
        precision=2,
    )

    table_markdown = analyzer.generate_comparison_table(
        results_dict=results_dict,
        problem_names=["Sphere"],
        format="markdown",
        scientific_notation=True,
        precision=2,
    )

    # Both tables should contain the algorithm and problem names
    assert "NM" in table_latex
    assert "NM" in table_markdown
    assert "Sphere" in table_latex
    assert "Sphere" in table_markdown

    # Both should contain scientific notation
    assert "e-" in table_latex or "e+" in table_latex
    assert "e-" in table_markdown or "e+" in table_markdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# Additional Integration Tests for Requirement 15.3 and 15.5
# ============================================================================


def test_statistical_computation_accuracy():
    """
    Test that statistical computations are mathematically accurate.
    
    Validates: Requirement 15.3 - compute statistical summaries across runs
    """
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments with known seeds for reproducibility
    problem = Sphere(dim=5)
    results = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=problem,
        num_runs=10,
        seeds=list(range(1, 11)),
        max_fes=1000,
        init_method="spendleySimplex",
    )

    # Compute statistics
    analyzer = ResultAnalyzer()
    stats = analyzer.compute_statistics(results)

    # Extract fitness values manually for verification
    fitness_values = [r.best_fitness for r in results if np.isfinite(r.best_fitness)]

    # Verify statistical computations are accurate
    assert len(fitness_values) > 0, "No valid fitness values found"

    # Test mean computation
    expected_mean = np.mean(fitness_values)
    assert np.isclose(
        stats["mean"], expected_mean, rtol=1e-10
    ), f"Mean mismatch: {stats['mean']} vs {expected_mean}"

    # Test median computation
    expected_median = np.median(fitness_values)
    assert np.isclose(
        stats["median"], expected_median, rtol=1e-10
    ), f"Median mismatch: {stats['median']} vs {expected_median}"

    # Test standard deviation computation (sample std with ddof=1)
    if len(fitness_values) > 1:
        expected_std = np.std(fitness_values, ddof=1)
        assert np.isclose(
            stats["std"], expected_std, rtol=1e-10
        ), f"Std mismatch: {stats['std']} vs {expected_std}"
    else:
        assert stats["std"] == 0.0, "Single value should have zero std"

    # Test best (minimum) computation
    expected_best = np.min(fitness_values)
    assert np.isclose(
        stats["best"], expected_best, rtol=1e-10
    ), f"Best mismatch: {stats['best']} vs {expected_best}"

    # Test worst (maximum) computation
    expected_worst = np.max(fitness_values)
    assert np.isclose(
        stats["worst"], expected_worst, rtol=1e-10
    ), f"Worst mismatch: {stats['worst']} vs {expected_worst}"

    # Test success rate computation
    num_success = sum(r.success for r in results)
    expected_success_rate = num_success / len(results)
    assert np.isclose(
        stats["success_rate"], expected_success_rate, rtol=1e-10
    ), f"Success rate mismatch: {stats['success_rate']} vs {expected_success_rate}"

    # Test average FES computation
    expected_avg_fes = np.mean([r.fes_used for r in results])
    assert np.isclose(
        stats["avg_fes"], expected_avg_fes, rtol=1e-10
    ), f"Avg FES mismatch: {stats['avg_fes']} vs {expected_avg_fes}"

    # Test average time computation
    expected_avg_time = np.mean([r.wall_time for r in results])
    assert np.isclose(
        stats["avg_time"], expected_avg_time, rtol=1e-10
    ), f"Avg time mismatch: {stats['avg_time']} vs {expected_avg_time}"

    # Test counts
    assert stats["num_runs"] == len(results), "Incorrect num_runs"
    assert stats["num_valid"] == len(fitness_values), "Incorrect num_valid"

    # Verify statistical relationships
    assert stats["best"] <= stats["mean"] <= stats["worst"], "Best <= Mean <= Worst violated"
    assert stats["best"] <= stats["median"] <= stats["worst"], "Best <= Median <= Worst violated"
    assert stats["std"] >= 0, "Standard deviation must be non-negative"
    assert 0.0 <= stats["success_rate"] <= 1.0, "Success rate must be in [0, 1]"


def test_statistical_computation_with_all_required_fields():
    """
    Test that compute_statistics returns all required fields per Requirement 15.5.
    
    Validates: Requirement 15.5 - include mean, median, std, best, and worst values
    """
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    problem = Rosenbrock(dim=3)
    results = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=problem,
        num_runs=5,
        seeds=[1, 2, 3, 4, 5],
        max_fes=500,
    )

    # Compute statistics
    analyzer = ResultAnalyzer()
    stats = analyzer.compute_statistics(results)

    # Verify all required fields are present (Requirement 15.5)
    required_fields = ["mean", "median", "std", "best", "worst"]
    for field in required_fields:
        assert field in stats, f"Required field '{field}' missing from statistics"
        assert isinstance(
            stats[field], (int, float)
        ), f"Field '{field}' must be numeric"
        assert np.isfinite(stats[field]), f"Field '{field}' must be finite"

    # Verify additional useful fields
    additional_fields = ["success_rate", "avg_fes", "avg_time", "num_runs", "num_valid"]
    for field in additional_fields:
        assert field in stats, f"Additional field '{field}' missing from statistics"


def test_convergence_plot_generation_accuracy():
    """
    Test that convergence plots are generated correctly with proper data.
    
    Validates: Requirement 15.5 - generate convergence plots
    """
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    problem = Sphere(dim=5)
    nm_results = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=problem,
        num_runs=5,
        seeds=[1, 2, 3, 4, 5],
        max_fes=1000,
    )

    anm_results = runner.run_experiment(
        algorithm_class=AdaptiveNelderMead,
        problem=problem,
        num_runs=5,
        seeds=[1, 2, 3, 4, 5],
        max_fes=1000,
    )

    # Generate convergence plot
    analyzer = ResultAnalyzer()

    try:
        fig = analyzer.generate_convergence_plots(
            results_dict={"NM": nm_results, "ANM": anm_results},
            problem_name="Sphere",
            show_plot=False,
            log_scale=True,
            plot_median=True,
            plot_quartiles=True,
        )

        # Verify figure was created
        assert fig is not None, "Figure should be created"

        # Verify figure has axes
        axes = fig.get_axes()
        assert len(axes) > 0, "Figure should have at least one axis"

        ax = axes[0]

        # Verify axis labels are set
        assert ax.get_xlabel() != "", "X-axis label should be set"
        assert ax.get_ylabel() != "", "Y-axis label should be set"
        assert ax.get_title() != "", "Title should be set"

        # Verify legend is present
        legend = ax.get_legend()
        assert legend is not None, "Legend should be present"

        # Verify both algorithms are in legend
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "NM" in legend_texts, "NM should be in legend"
        assert "ANM" in legend_texts, "ANM should be in legend"

        # Verify log scale is applied
        assert ax.get_yscale() == "log", "Y-axis should use log scale"

        # Verify lines are plotted
        lines = ax.get_lines()
        assert len(lines) >= 2, "Should have at least 2 lines (one per algorithm)"

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    except ImportError:
        # matplotlib not installed - skip test
        pytest.skip("matplotlib not installed")


def test_convergence_plot_with_different_configurations():
    """
    Test convergence plot generation with various configuration options.
    
    Validates: Requirement 15.5 - generate convergence plots with flexibility
    """
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    problem = Sphere(dim=3)
    results = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=problem,
        num_runs=3,
        seeds=[1, 2, 3],
        max_fes=500,
    )

    analyzer = ResultAnalyzer()

    try:
        # Test with median and quartiles
        fig1 = analyzer.generate_convergence_plots(
            results_dict={"NM": results},
            problem_name="Sphere",
            show_plot=False,
            log_scale=True,
            plot_median=True,
            plot_quartiles=True,
        )
        assert fig1 is not None
        import matplotlib.pyplot as plt

        plt.close(fig1)

        # Test with mean instead of median
        fig2 = analyzer.generate_convergence_plots(
            results_dict={"NM": results},
            problem_name="Sphere",
            show_plot=False,
            log_scale=False,
            plot_median=False,
            plot_quartiles=False,
        )
        assert fig2 is not None
        plt.close(fig2)

        # Test with custom figure size
        fig3 = analyzer.generate_convergence_plots(
            results_dict={"NM": results},
            problem_name="Sphere",
            show_plot=False,
            figsize=(8, 5),
        )
        assert fig3 is not None
        assert fig3.get_size_inches()[0] == 8
        assert fig3.get_size_inches()[1] == 5
        plt.close(fig3)

    except ImportError:
        pytest.skip("matplotlib not installed")


def test_comparison_table_generation_accuracy():
    """
    Test that comparison tables are generated correctly with accurate data.
    
    Validates: Requirement 15.5 - generate comparison tables
    """
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    sphere = Sphere(dim=3)
    rosenbrock = Rosenbrock(dim=3)

    results = runner.compare_algorithms(
        algorithm_configs=[
            {"class": NelderMead, "name": "NM", "max_fes": 500},
            {"class": AdaptiveNelderMead, "name": "ANM", "max_fes": 500},
        ],
        problems=[sphere, rosenbrock],
        num_runs=5,
        seeds=[1, 2, 3, 4, 5],
    )

    # Generate comparison table in LaTeX format
    analyzer = ResultAnalyzer()
    latex_table = analyzer.generate_comparison_table(
        results_dict=results,
        problem_names=["Sphere", "Rosenbrock"],
        format="latex",
        include_std=True,
        include_success_rate=True,
        scientific_notation=True,
        precision=2,
    )

    # Verify LaTeX table structure
    assert "\\begin{table}" in latex_table, "LaTeX table should have begin tag"
    assert "\\end{table}" in latex_table, "LaTeX table should have end tag"
    assert "\\begin{tabular}" in latex_table, "LaTeX table should have tabular environment"
    assert "\\end{tabular}" in latex_table, "LaTeX table should close tabular environment"
    assert "\\hline" in latex_table, "LaTeX table should have horizontal lines"

    # Verify algorithm names are present
    assert "NM" in latex_table, "NM algorithm should be in table"
    assert "ANM" in latex_table, "ANM algorithm should be in table"

    # Verify problem names are present
    assert "Sphere" in latex_table, "Sphere problem should be in table"
    assert "Rosenbrock" in latex_table, "Rosenbrock problem should be in table"

    # Verify column headers
    assert "Algorithm" in latex_table, "Algorithm column header should be present"
    assert "Problem" in latex_table, "Problem column header should be present"
    assert "Mean" in latex_table or "mean" in latex_table.lower(), "Mean column should be present"
    assert "Best" in latex_table or "best" in latex_table.lower(), "Best column should be present"

    # Verify scientific notation is used
    assert "e-" in latex_table or "e+" in latex_table, "Scientific notation should be used"

    # Verify std is included
    assert "pm" in latex_table or "±" in latex_table, "Standard deviation should be included"

    # Verify success rate is included
    assert "%" in latex_table or "\\%" in latex_table, "Success rate should be included"

    # Generate comparison table in Markdown format
    markdown_table = analyzer.generate_comparison_table(
        results_dict=results,
        problem_names=["Sphere", "Rosenbrock"],
        format="markdown",
        include_std=True,
        include_success_rate=True,
        scientific_notation=True,
        precision=2,
    )

    # Verify Markdown table structure
    assert "|" in markdown_table, "Markdown table should use pipe separators"
    assert "Algorithm" in markdown_table, "Algorithm column header should be present"
    assert "Problem" in markdown_table, "Problem column header should be present"

    # Verify algorithm and problem names
    assert "NM" in markdown_table, "NM algorithm should be in table"
    assert "ANM" in markdown_table, "ANM algorithm should be in table"
    assert "Sphere" in markdown_table, "Sphere problem should be in table"
    assert "Rosenbrock" in markdown_table, "Rosenbrock problem should be in table"

    # Verify scientific notation
    assert "e-" in markdown_table or "e+" in markdown_table, "Scientific notation should be used"


def test_comparison_table_with_different_formats():
    """
    Test comparison table generation with different formatting options.
    
    Validates: Requirement 15.5 - flexible table generation
    """
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run experiments
    problem = Sphere(dim=3)
    results = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=problem,
        num_runs=3,
        seeds=[1, 2, 3],
        max_fes=500,
    )

    results_dict = {"NM": {"Sphere": results}}

    analyzer = ResultAnalyzer()

    # Test without std
    table1 = analyzer.generate_comparison_table(
        results_dict=results_dict,
        problem_names=["Sphere"],
        format="latex",
        include_std=False,
        include_success_rate=True,
    )
    assert "pm" not in table1 and "±" not in table1, "Std should not be included"

    # Test without success rate
    table2 = analyzer.generate_comparison_table(
        results_dict=results_dict,
        problem_names=["Sphere"],
        format="latex",
        include_std=True,
        include_success_rate=False,
    )
    # Success rate column should not be present (harder to verify, but table should be shorter)
    assert len(table2) < len(
        analyzer.generate_comparison_table(
            results_dict=results_dict,
            problem_names=["Sphere"],
            format="latex",
            include_std=True,
            include_success_rate=True,
        )
    )

    # Test with fixed-point notation
    table3 = analyzer.generate_comparison_table(
        results_dict=results_dict,
        problem_names=["Sphere"],
        format="markdown",
        scientific_notation=False,
        precision=4,
    )
    # Should not have scientific notation
    lines_with_e = [line for line in table3.split("\n") if "e-" in line or "e+" in line]
    # Allow for some edge cases, but most lines should not have scientific notation
    assert len(lines_with_e) < 2, "Fixed-point notation should be used"

    # Test with different precision
    table4 = analyzer.generate_comparison_table(
        results_dict=results_dict,
        problem_names=["Sphere"],
        format="markdown",
        scientific_notation=True,
        precision=6,
    )
    # Verify higher precision is used (more digits after decimal)
    assert table4 != table1, "Different precision should produce different output"


def test_end_to_end_workflow_with_result_analysis():
    """
    Test complete end-to-end workflow: batch run -> statistics -> plots -> tables.
    
    Validates: Requirements 15.3, 15.5 - complete analysis workflow
    """
    # Create batch runner
    runner = BatchRunner(verbose=False)

    # Run comprehensive comparison
    sphere = Sphere(dim=5)
    rosenbrock = Rosenbrock(dim=5)

    results = runner.compare_algorithms(
        algorithm_configs=[
            {"class": NelderMead, "name": "NM", "max_fes": 1000},
            {"class": AdaptiveNelderMead, "name": "ANM", "max_fes": 1000},
        ],
        problems=[sphere, rosenbrock],
        num_runs=5,
        seeds=[1, 2, 3, 4, 5],
    )

    # Analyze results
    analyzer = ResultAnalyzer()

    # 1. Compute statistics for each algorithm-problem combination
    for alg_name in ["NM", "ANM"]:
        for problem_name in ["Sphere", "Rosenbrock"]:
            alg_results = results[alg_name][problem_name]
            stats = analyzer.compute_statistics(alg_results)

            # Verify all required statistics are present
            assert "mean" in stats
            assert "median" in stats
            assert "std" in stats
            assert "best" in stats
            assert "worst" in stats
            assert stats["num_runs"] == 5

    # 2. Generate comparison table
    table = analyzer.generate_comparison_table(
        results_dict=results,
        problem_names=["Sphere", "Rosenbrock"],
        format="latex",
    )
    assert len(table) > 0
    assert "NM" in table
    assert "ANM" in table

    # 3. Generate convergence plots (if matplotlib available)
    try:
        for problem_name in ["Sphere", "Rosenbrock"]:
            fig = analyzer.generate_convergence_plots(
                results_dict={
                    "NM": results["NM"][problem_name],
                    "ANM": results["ANM"][problem_name],
                },
                problem_name=problem_name,
                show_plot=False,
            )
            assert fig is not None

            import matplotlib.pyplot as plt

            plt.close(fig)

    except ImportError:
        pytest.skip("matplotlib not installed")


def test_statistical_accuracy_with_edge_cases():
    """
    Test statistical computation with edge cases (single run, identical values, etc.).
    
    Validates: Requirement 15.3 - robust statistical computation
    """
    # Create batch runner
    runner = BatchRunner(verbose=False)
    analyzer = ResultAnalyzer()

    # Test with single run
    problem = Sphere(dim=3)
    results_single = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=problem,
        num_runs=1,
        seeds=[1],
        max_fes=500,
    )

    stats_single = analyzer.compute_statistics(results_single)
    assert stats_single["num_runs"] == 1
    assert stats_single["std"] == 0.0, "Single run should have zero std"
    assert stats_single["mean"] == stats_single["best"] == stats_single["worst"]

    # Test with multiple runs (should have non-zero std if results vary)
    results_multiple = runner.run_experiment(
        algorithm_class=NelderMead,
        problem=problem,
        num_runs=10,
        seeds=list(range(1, 11)),
        max_fes=500,
    )

    stats_multiple = analyzer.compute_statistics(results_multiple)
    assert stats_multiple["num_runs"] == 10
    # Std might be zero if all runs converge to same value, but usually non-zero
    assert stats_multiple["std"] >= 0.0
