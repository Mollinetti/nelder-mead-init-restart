#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for BatchRunner

Tests the batch testing infrastructure including:
- Running single experiments with multiple seeds
- Comparing multiple algorithms on multiple problems
- Recording results in structured format
- Computing statistics correctly
"""

import pytest
import numpy as np

from nelder_mead.testing.batch_runner import BatchRunner, ExperimentResult
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
from nelder_mead.benchmarks.unconstrained import Sphere
from nelder_mead.benchmarks.problem_suite import OptimizationProblem


class SimpleProblem(OptimizationProblem):
    """Simple test problem for unit testing."""

    def __init__(self, dim=2):
        super().__init__(
            name="SimpleProblem",
            dim=dim,
            bounds=(np.full(dim, -10.0), np.full(dim, 10.0)),
            optimum_value=0.0,
            optimum_location=np.zeros(dim),
        )

    def objective(self, x: np.ndarray) -> float:
        """Simple quadratic function."""
        return np.sum(x**2)


class TestBatchRunner:
    """Test suite for BatchRunner class."""

    def test_initialization(self):
        """Test BatchRunner initialization."""
        runner = BatchRunner(verbose=False)
        assert runner.verbose is False

        runner_verbose = BatchRunner(verbose=True)
        assert runner_verbose.verbose is True

    def test_run_experiment_basic(self):
        """Test running a basic experiment with multiple seeds."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=3,
            seeds=[1, 2, 3],
            max_fes=500,
        )

        # Check we got correct number of results
        assert len(results) == 3

        # Check each result has correct structure
        for i, result in enumerate(results):
            assert isinstance(result, ExperimentResult)
            assert result.algorithm_name == "NelderMead"
            assert result.problem_name == "SimpleProblem"
            assert result.seed == i + 1
            assert result.best_solution.shape == (2,)
            assert isinstance(result.best_fitness, float)
            assert isinstance(result.best_violation, float)
            assert result.fes_used > 0
            assert result.fes_used <= 500
            assert isinstance(result.convergence_history, list)
            assert result.wall_time > 0
            assert isinstance(result.success, bool)

    def test_run_experiment_with_default_seeds(self):
        """Test that default seeds are generated correctly."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=5,
            seeds=None,  # Should generate seeds [1, 2, 3, 4, 5]
            max_fes=300,
        )

        assert len(results) == 5
        assert [r.seed for r in results] == [1, 2, 3, 4, 5]

    def test_run_experiment_reproducibility(self):
        """Test that same seed produces same results."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        # Run twice with same seed
        results1 = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=1,
            seeds=[42],
            max_fes=500,
            init_method="spendleySimplex",
        )

        results2 = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=1,
            seeds=[42],
            max_fes=500,
            init_method="spendleySimplex",
        )

        # Results should be identical
        assert results1[0].best_fitness == results2[0].best_fitness
        assert np.allclose(results1[0].best_solution, results2[0].best_solution)
        assert results1[0].fes_used == results2[0].fes_used

    def test_run_experiment_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=3,
            seeds=[1, 2, 3],
            max_fes=500,
        )

        # Results should be different (with high probability)
        fitness_values = [r.best_fitness for r in results]
        # Not all fitness values should be identical
        assert len(set(fitness_values)) > 1 or all(f < 1e-6 for f in fitness_values)

    def test_run_experiment_success_detection(self):
        """Test that success is correctly detected when optimum is reached."""
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=2)  # Known optimum at 0

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=2,
            seeds=[1, 2],
            max_fes=1000,
            success_tolerance=1e-2,
        )

        # At least check that success field is set
        for result in results:
            assert isinstance(result.success, bool)
            # If fitness is very close to optimum, should be marked as success
            if abs(result.best_fitness - 0.0) < 1e-2:
                assert result.success

    def test_run_experiment_convergence_history(self):
        """Test that convergence history is recorded correctly."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=1,
            seeds=[1],
            max_fes=500,
        )

        result = results[0]

        # Check convergence history structure
        assert len(result.convergence_history) > 0

        for fes, fitness in result.convergence_history:
            assert isinstance(fes, int)
            assert isinstance(fitness, float)
            assert fes > 0
            assert fes <= 500

        # Check that fitness is non-increasing (best solution tracking)
        fitness_values = [f for _, f in result.convergence_history]
        for i in range(1, len(fitness_values)):
            assert (
                fitness_values[i] <= fitness_values[i - 1] + 1e-10
            )  # Allow small numerical errors

    def test_run_experiment_algorithm_parameters(self):
        """Test that algorithm parameters are correctly passed."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        # Run with custom parameters
        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=1,
            seeds=[1],
            max_fes=300,
            init_method="uniform",
            delta_r=1.5,
            delta_e=2.5,
        )

        # Should complete without errors
        assert len(results) == 1
        assert results[0].fes_used <= 300

    def test_run_experiment_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        # Test num_runs <= 0
        with pytest.raises(ValueError, match="num_runs must be positive"):
            runner.run_experiment(
                algorithm_class=NelderMead, problem=problem, num_runs=0, seeds=[1]
            )

        # Test seeds length mismatch
        with pytest.raises(ValueError, match="Length of seeds"):
            runner.run_experiment(
                algorithm_class=NelderMead,
                problem=problem,
                num_runs=3,
                seeds=[1, 2],  # Only 2 seeds for 3 runs
            )

        # Test non-algorithm class
        with pytest.raises(TypeError, match="must inherit from BaseAlgorithm"):
            runner.run_experiment(
                algorithm_class=str,  # Not an algorithm class
                problem=problem,
                num_runs=1,
            )

    def test_compare_algorithms_basic(self):
        """Test comparing multiple algorithms on multiple problems."""
        runner = BatchRunner(verbose=False)

        algorithm_configs = [
            {"class": NelderMead, "name": "NM", "max_fes": 500},
            {"class": AdaptiveNelderMead, "name": "ANM", "max_fes": 500},
        ]

        problems = [SimpleProblem(dim=2), Sphere(dim=2)]

        results = runner.compare_algorithms(
            algorithm_configs=algorithm_configs,
            problems=problems,
            num_runs=2,
            seeds=[1, 2],
        )

        # Check structure
        assert "NM" in results
        assert "ANM" in results

        for alg_name in ["NM", "ANM"]:
            assert "SimpleProblem" in results[alg_name]
            assert "Sphere" in results[alg_name]

            for problem_name in ["SimpleProblem", "Sphere"]:
                problem_results = results[alg_name][problem_name]
                assert len(problem_results) == 2

                for result in problem_results:
                    assert isinstance(result, ExperimentResult)
                    assert result.algorithm_name in ["NelderMead", "AdaptiveNelderMead"]

    def test_compare_algorithms_with_default_names(self):
        """Test that algorithm names default to class names if not provided."""
        runner = BatchRunner(verbose=False)

        algorithm_configs = [
            {"class": NelderMead, "max_fes": 300},  # No 'name' key
        ]

        problems = [SimpleProblem(dim=2)]

        results = runner.compare_algorithms(
            algorithm_configs=algorithm_configs,
            problems=problems,
            num_runs=1,
            seeds=[1],
        )

        # Should use class name as key
        assert "NelderMead" in results

    def test_compare_algorithms_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        runner = BatchRunner(verbose=False)

        # Test empty algorithm configs
        with pytest.raises(ValueError, match="algorithm_configs cannot be empty"):
            runner.compare_algorithms(
                algorithm_configs=[], problems=[SimpleProblem(dim=2)], num_runs=1
            )

        # Test empty problems
        with pytest.raises(ValueError, match="problems cannot be empty"):
            runner.compare_algorithms(
                algorithm_configs=[{"class": NelderMead}], problems=[], num_runs=1
            )

        # Test missing 'class' key
        with pytest.raises(KeyError, match="must have 'class' key"):
            runner.compare_algorithms(
                algorithm_configs=[{"name": "Test"}],  # Missing 'class'
                problems=[SimpleProblem(dim=2)],
                num_runs=1,
            )

    def test_export_results_to_dict(self):
        """Test exporting results to dictionary format."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=3,
            seeds=[1, 2, 3],
            max_fes=500,
        )

        # Export to dict
        export_dict = runner.export_results_to_dict(results)

        # Check structure
        assert "algorithm" in export_dict
        assert "problem" in export_dict
        assert "num_runs" in export_dict
        assert "statistics" in export_dict
        assert "runs" in export_dict

        # Check values
        assert export_dict["algorithm"] == "NelderMead"
        assert export_dict["problem"] == "SimpleProblem"
        assert export_dict["num_runs"] == 3

        # Check statistics
        stats = export_dict["statistics"]
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "best" in stats
        assert "worst" in stats
        assert "success_rate" in stats
        assert "avg_fes" in stats
        assert "avg_time" in stats

        # Check runs
        assert len(export_dict["runs"]) == 3
        for run in export_dict["runs"]:
            assert "seed" in run
            assert "best_fitness" in run
            assert "best_violation" in run
            assert "fes_used" in run
            assert "wall_time" in run
            assert "success" in run
            assert "best_solution" in run
            assert "convergence_history" in run

    def test_export_results_statistics_correctness(self):
        """Test that exported statistics are computed correctly."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=2)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=5,
            seeds=[1, 2, 3, 4, 5],
            max_fes=500,
        )

        export_dict = runner.export_results_to_dict(results)
        stats = export_dict["statistics"]

        # Manually compute statistics
        fitness_values = [r.best_fitness for r in results]

        assert np.isclose(stats["mean"], np.mean(fitness_values))
        assert np.isclose(stats["median"], np.median(fitness_values))
        assert np.isclose(stats["std"], np.std(fitness_values))
        assert np.isclose(stats["best"], np.min(fitness_values))
        assert np.isclose(stats["worst"], np.max(fitness_values))

        success_count = sum(r.success for r in results)
        assert np.isclose(stats["success_rate"], success_count / len(results))

    def test_experiment_result_dataclass(self):
        """Test ExperimentResult dataclass structure."""
        result = ExperimentResult(
            algorithm_name="TestAlg",
            problem_name="TestProblem",
            seed=42,
            best_solution=np.array([1.0, 2.0]),
            best_fitness=3.5,
            best_violation=0.0,
            fes_used=100,
            convergence_history=[(10, 5.0), (20, 4.0)],
            wall_time=1.5,
            success=True,
            final_simplex=np.array([[1, 2], [3, 4]]),
        )

        assert result.algorithm_name == "TestAlg"
        assert result.problem_name == "TestProblem"
        assert result.seed == 42
        assert np.array_equal(result.best_solution, np.array([1.0, 2.0]))
        assert result.best_fitness == 3.5
        assert result.best_violation == 0.0
        assert result.fes_used == 100
        assert result.convergence_history == [(10, 5.0), (20, 4.0)]
        assert result.wall_time == 1.5
        assert result.success is True
        assert np.array_equal(result.final_simplex, np.array([[1, 2], [3, 4]]))

    def test_run_experiment_with_constrained_problem(self):
        """Test running experiments on constrained problems."""
        from nelder_mead.benchmarks.constrained import G01
        from nelder_mead.constraints.deb_barrier import DebBarrier

        runner = BatchRunner(verbose=False)
        problem = G01()

        # Create barrier for constraint handling
        barrier = DebBarrier(
            bounds=(problem.bounds[0], problem.bounds[1]), num_solutions=problem.dim + 1
        )

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=2,
            seeds=[1, 2],
            max_fes=1000,
            barrier=barrier,
        )

        assert len(results) == 2

        for result in results:
            assert result.problem_name == "G01"
            # Check that violation is tracked
            assert isinstance(result.best_violation, float)
            assert result.best_violation >= 0.0

    def test_multiple_runs_show_variation(self):
        """Test that multiple runs with different seeds show statistical variation."""
        runner = BatchRunner(verbose=False)
        problem = SimpleProblem(dim=5)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=10,
            seeds=list(range(1, 11)),
            max_fes=500,
        )

        fitness_values = [r.best_fitness for r in results]

        # Should have some variation (std > 0)
        # Unless all runs converged perfectly to optimum
        if not all(f < 1e-10 for f in fitness_values):
            assert np.std(fitness_values) > 0

    def test_compare_algorithms_consistent_seeds(self):
        """Test that compare_algorithms uses consistent seeds across algorithms."""
        runner = BatchRunner(verbose=False)

        algorithm_configs = [
            {"class": NelderMead, "name": "NM", "max_fes": 500},
            {"class": AdaptiveNelderMead, "name": "ANM", "max_fes": 500},
        ]

        problems = [SimpleProblem(dim=2)]
        seeds = [10, 20, 30]

        results = runner.compare_algorithms(
            algorithm_configs=algorithm_configs,
            problems=problems,
            num_runs=3,
            seeds=seeds,
        )

        # Check that both algorithms used the same seeds
        nm_seeds = [r.seed for r in results["NM"]["SimpleProblem"]]
        anm_seeds = [r.seed for r in results["ANM"]["SimpleProblem"]]

        assert nm_seeds == seeds
        assert anm_seeds == seeds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
