#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for ResultAnalyzer class.

Tests statistical computation, convergence plotting, and table generation
for optimization experiment results.
"""

import pytest
import numpy as np
from typing import List
import tempfile
import os

from nelder_mead.testing.batch_runner import ExperimentResult
from nelder_mead.testing.result_analyzer import ResultAnalyzer

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_results() -> List[ExperimentResult]:
    """Create sample experiment results for testing."""
    results = []

    # Create 5 successful runs with varying fitness values
    for i in range(5):
        result = ExperimentResult(
            algorithm_name="TestAlgorithm",
            problem_name="TestProblem",
            seed=i + 1,
            best_solution=np.array([0.1 * i, 0.2 * i]),
            best_fitness=1e-5 * (i + 1),  # 1e-5, 2e-5, 3e-5, 4e-5, 5e-5
            best_violation=0.0,
            fes_used=1000 + i * 100,
            convergence_history=[(100, 1e-3), (500, 1e-4), (1000, 1e-5 * (i + 1))],
            wall_time=1.0 + i * 0.1,
            success=True,
        )
        results.append(result)

    return results


@pytest.fixture
def sample_results_with_failures() -> List[ExperimentResult]:
    """Create sample results including some failed runs."""
    results = []

    # 3 successful runs
    for i in range(3):
        result = ExperimentResult(
            algorithm_name="TestAlgorithm",
            problem_name="TestProblem",
            seed=i + 1,
            best_solution=np.array([0.1 * i, 0.2 * i]),
            best_fitness=1e-5 * (i + 1),
            best_violation=0.0,
            fes_used=1000,
            convergence_history=[(100, 1e-3), (500, 1e-4), (1000, 1e-5 * (i + 1))],
            wall_time=1.0,
            success=True,
        )
        results.append(result)

    # 2 failed runs
    for i in range(2):
        result = ExperimentResult(
            algorithm_name="TestAlgorithm",
            problem_name="TestProblem",
            seed=i + 4,
            best_solution=np.array([np.nan, np.nan]),
            best_fitness=np.inf,
            best_violation=np.inf,
            fes_used=0,
            convergence_history=[],
            wall_time=0.1,
            success=False,
        )
        results.append(result)

    return results


@pytest.fixture
def multi_algorithm_results() -> dict:
    """Create results for multiple algorithms on multiple problems."""
    results = {}

    # Algorithm 1
    results["Alg1"] = {}
    results["Alg1"]["Problem1"] = [
        ExperimentResult(
            algorithm_name="Alg1",
            problem_name="Problem1",
            seed=i + 1,
            best_solution=np.array([0.0, 0.0]),
            best_fitness=1e-6 * (i + 1),
            best_violation=0.0,
            fes_used=1000,
            convergence_history=[(100, 1e-4), (500, 1e-5), (1000, 1e-6 * (i + 1))],
            wall_time=1.0,
            success=True,
        )
        for i in range(3)
    ]

    # Algorithm 2
    results["Alg2"] = {}
    results["Alg2"]["Problem1"] = [
        ExperimentResult(
            algorithm_name="Alg2",
            problem_name="Problem1",
            seed=i + 1,
            best_solution=np.array([0.0, 0.0]),
            best_fitness=1e-5 * (i + 1),
            best_violation=0.0,
            fes_used=1500,
            convergence_history=[(100, 1e-3), (500, 1e-4), (1500, 1e-5 * (i + 1))],
            wall_time=1.5,
            success=True,
        )
        for i in range(3)
    ]

    return results


# ============================================================================
# Test compute_statistics
# ============================================================================


def test_compute_statistics_basic(sample_results):
    """Test basic statistical computation."""
    analyzer = ResultAnalyzer()
    stats = analyzer.compute_statistics(sample_results)

    # Check that all expected keys are present
    expected_keys = [
        "mean",
        "median",
        "std",
        "best",
        "worst",
        "success_rate",
        "avg_fes",
        "avg_time",
        "num_runs",
        "num_valid",
    ]
    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"

    # Check values
    fitness_values = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
    assert stats["mean"] == pytest.approx(np.mean(fitness_values))
    assert stats["median"] == pytest.approx(np.median(fitness_values))
    assert stats["std"] == pytest.approx(np.std(fitness_values, ddof=1))
    assert stats["best"] == pytest.approx(1e-5)
    assert stats["worst"] == pytest.approx(5e-5)
    assert stats["success_rate"] == 1.0
    assert stats["num_runs"] == 5
    assert stats["num_valid"] == 5


def test_compute_statistics_with_failures(sample_results_with_failures):
    """Test statistics computation with failed runs."""
    analyzer = ResultAnalyzer()
    stats = analyzer.compute_statistics(
        sample_results_with_failures, exclude_failed=True
    )

    # Should only consider the 3 successful runs
    fitness_values = [1e-5, 2e-5, 3e-5]
    assert stats["mean"] == pytest.approx(np.mean(fitness_values))
    assert stats["median"] == pytest.approx(np.median(fitness_values))
    assert stats["best"] == pytest.approx(1e-5)
    assert stats["worst"] == pytest.approx(3e-5)
    assert stats["success_rate"] == 0.6  # 3 out of 5
    assert stats["num_runs"] == 5
    assert stats["num_valid"] == 3


def test_compute_statistics_empty_list():
    """Test that empty results list raises ValueError."""
    analyzer = ResultAnalyzer()
    with pytest.raises(ValueError, match="Results list cannot be empty"):
        analyzer.compute_statistics([])


def test_compute_statistics_all_failed():
    """Test that all failed runs raises ValueError."""
    analyzer = ResultAnalyzer()

    failed_results = [
        ExperimentResult(
            algorithm_name="Test",
            problem_name="Test",
            seed=i,
            best_solution=np.array([np.nan]),
            best_fitness=np.inf,
            best_violation=np.inf,
            fes_used=0,
            convergence_history=[],
            wall_time=0.0,
            success=False,
        )
        for i in range(3)
    ]

    with pytest.raises(ValueError, match="All runs failed"):
        analyzer.compute_statistics(failed_results, exclude_failed=True)


def test_compute_statistics_single_run():
    """Test statistics with a single run."""
    analyzer = ResultAnalyzer()

    single_result = [
        ExperimentResult(
            algorithm_name="Test",
            problem_name="Test",
            seed=1,
            best_solution=np.array([0.0]),
            best_fitness=1e-5,
            best_violation=0.0,
            fes_used=1000,
            convergence_history=[(1000, 1e-5)],
            wall_time=1.0,
            success=True,
        )
    ]

    stats = analyzer.compute_statistics(single_result)

    assert stats["mean"] == pytest.approx(1e-5)
    assert stats["median"] == pytest.approx(1e-5)
    assert stats["best"] == pytest.approx(1e-5)
    assert stats["worst"] == pytest.approx(1e-5)
    assert stats["std"] == 0.0  # Single value has zero std with ddof=1 gives 0
    assert stats["success_rate"] == 1.0


def test_compute_statistics_avg_fes_and_time(sample_results):
    """Test that average FES and time are computed correctly."""
    analyzer = ResultAnalyzer()
    stats = analyzer.compute_statistics(sample_results)

    expected_avg_fes = np.mean([1000, 1100, 1200, 1300, 1400])
    expected_avg_time = np.mean([1.0, 1.1, 1.2, 1.3, 1.4])

    assert stats["avg_fes"] == pytest.approx(expected_avg_fes)
    assert stats["avg_time"] == pytest.approx(expected_avg_time)


# ============================================================================
# Test generate_convergence_plots
# ============================================================================


def test_generate_convergence_plots_basic(multi_algorithm_results):
    """Test basic convergence plot generation."""
    analyzer = ResultAnalyzer()

    # Extract results for plotting
    plot_data = {
        "Alg1": multi_algorithm_results["Alg1"]["Problem1"],
        "Alg2": multi_algorithm_results["Alg2"]["Problem1"],
    }

    # Test without showing or saving (just create the figure)
    try:
        fig = analyzer.generate_convergence_plots(
            results_dict=plot_data,
            problem_name="Problem1",
            show_plot=False,
            output_path=None,
        )

        # Check that figure was created
        assert fig is not None

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    except ImportError:
        # matplotlib not installed - skip test
        pytest.skip("matplotlib not installed")


def test_generate_convergence_plots_save_to_file(multi_algorithm_results):
    """Test saving convergence plot to file."""
    analyzer = ResultAnalyzer()

    plot_data = {"Alg1": multi_algorithm_results["Alg1"]["Problem1"]}

    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            fig = analyzer.generate_convergence_plots(
                results_dict=plot_data,
                problem_name="Problem1",
                output_path=tmp_path,
                show_plot=False,
            )

            # Check that file was created
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0

            # Clean up
            import matplotlib.pyplot as plt

            plt.close(fig)

        finally:
            # Remove temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except ImportError:
        pytest.skip("matplotlib not installed")


def test_generate_convergence_plots_empty_dict():
    """Test that empty results dict raises ValueError."""
    analyzer = ResultAnalyzer()

    try:
        with pytest.raises(ValueError, match="results_dict cannot be empty"):
            analyzer.generate_convergence_plots(
                results_dict={}, problem_name="Test", show_plot=False
            )
    except ImportError:
        pytest.skip("matplotlib not installed")


def test_generate_convergence_plots_log_scale(multi_algorithm_results):
    """Test convergence plot with logarithmic scale."""
    analyzer = ResultAnalyzer()

    plot_data = {"Alg1": multi_algorithm_results["Alg1"]["Problem1"]}

    try:
        fig = analyzer.generate_convergence_plots(
            results_dict=plot_data,
            problem_name="Problem1",
            log_scale=True,
            show_plot=False,
        )

        # Check that y-axis is logarithmic
        ax = fig.axes[0]
        assert ax.get_yscale() == "log"

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    except ImportError:
        pytest.skip("matplotlib not installed")


def test_generate_convergence_plots_custom_figsize(multi_algorithm_results):
    """Test convergence plot with custom figure size."""
    analyzer = ResultAnalyzer()

    plot_data = {"Alg1": multi_algorithm_results["Alg1"]["Problem1"]}

    try:
        fig = analyzer.generate_convergence_plots(
            results_dict=plot_data,
            problem_name="Problem1",
            figsize=(12, 8),
            show_plot=False,
        )

        # Check figure size
        assert fig.get_figwidth() == pytest.approx(12)
        assert fig.get_figheight() == pytest.approx(8)

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    except ImportError:
        pytest.skip("matplotlib not installed")


# ============================================================================
# Test generate_comparison_table
# ============================================================================


def test_generate_comparison_table_latex(multi_algorithm_results):
    """Test LaTeX table generation."""
    analyzer = ResultAnalyzer()

    table = analyzer.generate_comparison_table(
        results_dict=multi_algorithm_results, problem_names=["Problem1"], format="latex"
    )

    # Check that table contains expected LaTeX commands
    assert "\\begin{table}" in table
    assert "\\end{table}" in table
    assert "\\begin{tabular}" in table
    assert "\\end{tabular}" in table
    assert "\\hline" in table
    assert "Alg1" in table
    assert "Alg2" in table
    assert "Problem1" in table


def test_generate_comparison_table_markdown(multi_algorithm_results):
    """Test Markdown table generation."""
    analyzer = ResultAnalyzer()

    table = analyzer.generate_comparison_table(
        results_dict=multi_algorithm_results,
        problem_names=["Problem1"],
        format="markdown",
    )

    # Check that table contains expected Markdown syntax
    assert "|" in table
    assert "Algorithm" in table
    assert "Problem" in table
    assert "Alg1" in table
    assert "Alg2" in table
    assert "Problem1" in table


def test_generate_comparison_table_save_to_file(multi_algorithm_results):
    """Test saving comparison table to file."""
    analyzer = ResultAnalyzer()

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        table = analyzer.generate_comparison_table(
            results_dict=multi_algorithm_results,
            problem_names=["Problem1"],
            output_path=tmp_path,
            format="latex",
        )

        # Check that file was created
        assert os.path.exists(tmp_path)

        # Read file and verify content
        with open(tmp_path, "r") as f:
            file_content = f.read()

        assert file_content == table
        assert "\\begin{table}" in file_content

    finally:
        # Remove temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_generate_comparison_table_empty_dict():
    """Test that empty results dict raises ValueError."""
    analyzer = ResultAnalyzer()

    with pytest.raises(ValueError, match="results_dict cannot be empty"):
        analyzer.generate_comparison_table(
            results_dict={}, problem_names=["Test"], format="latex"
        )


def test_generate_comparison_table_empty_problems():
    """Test that empty problem names raises ValueError."""
    analyzer = ResultAnalyzer()

    with pytest.raises(ValueError, match="problem_names cannot be empty"):
        analyzer.generate_comparison_table(
            results_dict={"Alg1": {}}, problem_names=[], format="latex"
        )


def test_generate_comparison_table_invalid_format():
    """Test that invalid format raises ValueError."""
    analyzer = ResultAnalyzer()

    with pytest.raises(ValueError, match="format must be 'latex' or 'markdown'"):
        analyzer.generate_comparison_table(
            results_dict={"Alg1": {"Problem1": []}},
            problem_names=["Problem1"],
            format="invalid",
        )


def test_generate_comparison_table_scientific_notation(multi_algorithm_results):
    """Test table with scientific notation."""
    analyzer = ResultAnalyzer()

    table = analyzer.generate_comparison_table(
        results_dict=multi_algorithm_results,
        problem_names=["Problem1"],
        format="latex",
        scientific_notation=True,
        precision=2,
    )

    # Check for scientific notation (e.g., 1.23e-05)
    assert "e-" in table or "e+" in table


def test_generate_comparison_table_without_std(multi_algorithm_results):
    """Test table without standard deviation."""
    analyzer = ResultAnalyzer()

    table = analyzer.generate_comparison_table(
        results_dict=multi_algorithm_results,
        problem_names=["Problem1"],
        format="markdown",
        include_std=False,
    )

    # Should not contain ± symbol
    assert "±" not in table


def test_generate_comparison_table_without_success_rate(multi_algorithm_results):
    """Test table without success rate column."""
    analyzer = ResultAnalyzer()

    table = analyzer.generate_comparison_table(
        results_dict=multi_algorithm_results,
        problem_names=["Problem1"],
        format="markdown",
        include_success_rate=False,
    )

    # Should not contain Success Rate header
    assert "Success Rate" not in table


def test_generate_comparison_table_with_failed_runs():
    """Test table generation with failed runs."""
    analyzer = ResultAnalyzer()

    # Create results with all failed runs
    results = {
        "FailedAlg": {
            "Problem1": [
                ExperimentResult(
                    algorithm_name="FailedAlg",
                    problem_name="Problem1",
                    seed=i,
                    best_solution=np.array([np.nan]),
                    best_fitness=np.inf,
                    best_violation=np.inf,
                    fes_used=0,
                    convergence_history=[],
                    wall_time=0.0,
                    success=False,
                )
                for i in range(3)
            ]
        }
    }

    table = analyzer.generate_comparison_table(
        results_dict=results, problem_names=["Problem1"], format="markdown"
    )

    # Should contain "Failed" indicator
    assert "Failed" in table


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_workflow_statistics_and_table(sample_results):
    """Test complete workflow: compute statistics and generate table."""
    analyzer = ResultAnalyzer()

    # Compute statistics
    stats = analyzer.compute_statistics(sample_results)
    assert stats["mean"] > 0
    assert stats["success_rate"] == 1.0

    # Create results dict for table
    results_dict = {"TestAlgorithm": {"TestProblem": sample_results}}

    # Generate table
    table = analyzer.generate_comparison_table(
        results_dict=results_dict, problem_names=["TestProblem"], format="markdown"
    )

    assert "TestAlgorithm" in table
    assert "TestProblem" in table


def test_analyzer_with_real_convergence_data():
    """Test analyzer with realistic convergence data."""
    analyzer = ResultAnalyzer()

    # Create results with realistic convergence histories
    results = []
    for seed in range(5):
        # Simulate convergence: exponential decay
        fes_points = np.linspace(0, 5000, 50)
        fitness_points = 1e2 * np.exp(-fes_points / 1000) + 1e-6
        convergence_history = [
            (int(fes), float(fit)) for fes, fit in zip(fes_points, fitness_points)
        ]

        result = ExperimentResult(
            algorithm_name="RealAlg",
            problem_name="RealProblem",
            seed=seed,
            best_solution=np.zeros(10),
            best_fitness=fitness_points[-1],
            best_violation=0.0,
            fes_used=5000,
            convergence_history=convergence_history,
            wall_time=2.5,
            success=True,
        )
        results.append(result)

    # Compute statistics
    stats = analyzer.compute_statistics(results)
    assert stats["mean"] > 0
    assert stats["best"] <= stats["mean"] <= stats["worst"]

    # Try to generate plot (if matplotlib available)
    try:
        fig = analyzer.generate_convergence_plots(
            results_dict={"RealAlg": results},
            problem_name="RealProblem",
            show_plot=False,
        )

        import matplotlib.pyplot as plt

        plt.close(fig)
    except ImportError:
        pass  # matplotlib not available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
