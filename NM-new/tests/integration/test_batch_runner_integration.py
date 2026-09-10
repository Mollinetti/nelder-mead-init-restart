#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests for BatchRunner

Tests the BatchRunner with real algorithms and benchmark problems
to ensure end-to-end functionality works correctly.
"""

import pytest
import numpy as np

from nelder_mead.testing.batch_runner import BatchRunner
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
from nelder_mead.benchmarks.unconstrained import Sphere, Rosenbrock
from nelder_mead.benchmarks.constrained import G01
from nelder_mead.constraints.deb_barrier import DebBarrier


class TestBatchRunnerIntegration:
    """Integration tests for BatchRunner with real problems."""

    def test_single_algorithm_unconstrained_problem(self):
        """Test running a single algorithm on an unconstrained problem."""
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=5)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=5,
            max_fes=2000,
            init_method="spendleySimplex",
        )

        # All runs should complete
        assert len(results) == 5

        # All runs should find good solutions (Sphere is easy)
        for result in results:
            assert result.best_fitness < 1e-4  # Should get very close to optimum
            assert result.fes_used <= 2000
            assert result.success  # Should reach optimum within tolerance

    def test_adaptive_vs_classic_nelder_mead(self):
        """Test comparing Adaptive NM vs Classic NM on multiple problems."""
        runner = BatchRunner(verbose=False)

        algorithm_configs = [
            {"class": NelderMead, "name": "Classic-NM", "max_fes": 3000},
            {"class": AdaptiveNelderMead, "name": "Adaptive-NM", "max_fes": 3000},
        ]

        problems = [Sphere(dim=5), Rosenbrock(dim=5)]

        results = runner.compare_algorithms(
            algorithm_configs=algorithm_configs,
            problems=problems,
            num_runs=3,
            seeds=[1, 2, 3],
        )

        # Check structure
        assert "Classic-NM" in results
        assert "Adaptive-NM" in results

        # Check both algorithms ran on both problems
        for alg_name in ["Classic-NM", "Adaptive-NM"]:
            assert "Sphere" in results[alg_name]
            assert "Rosenbrock" in results[alg_name]

            # Each problem should have 3 runs
            assert len(results[alg_name]["Sphere"]) == 3
            assert len(results[alg_name]["Rosenbrock"]) == 3

        # Sphere should be solved successfully by both
        for alg_name in ["Classic-NM", "Adaptive-NM"]:
            sphere_results = results[alg_name]["Sphere"]
            success_count = sum(r.success for r in sphere_results)
            assert success_count >= 2  # At least 2 out of 3 should succeed

    def test_constrained_problem_with_barrier(self):
        """Test running on a constrained problem with barrier method."""
        runner = BatchRunner(verbose=False)
        problem = G01()

        # Create barrier for constraint handling
        barrier = DebBarrier(
            bounds=(problem.bounds[0], problem.bounds[1]), num_solutions=problem.dim + 1
        )

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=3,
            seeds=[1, 2, 3],
            max_fes=5000,
            barrier=barrier,
        )

        # All runs should complete
        assert len(results) == 3

        # Check that constraint violations are tracked
        for result in results:
            assert isinstance(result.best_violation, float)
            assert result.best_violation >= 0.0
            # G01 is challenging, so we just check runs completed
            assert result.fes_used > 0

    def test_export_and_statistics(self):
        """Test exporting results and computing statistics."""
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=3)

        results = runner.run_experiment(
            algorithm_class=NelderMead, problem=problem, num_runs=10, max_fes=1500
        )

        # Export to dictionary
        export_dict = runner.export_results_to_dict(results)

        # Verify statistics are reasonable
        stats = export_dict["statistics"]

        # Mean should be close to optimum (0.0) for Sphere
        assert stats["mean"] < 1e-3
        assert stats["best"] <= stats["mean"]
        assert stats["mean"] <= stats["worst"]
        assert stats["std"] >= 0

        # Success rate should be high for easy problem
        assert stats["success_rate"] > 0.5

        # All runs should be recorded
        assert len(export_dict["runs"]) == 10

    def test_convergence_history_tracking(self):
        """Test that convergence history is properly tracked."""
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=3)

        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=1,
            seeds=[42],
            max_fes=1000,
        )

        result = results[0]

        # Should have convergence history
        assert len(result.convergence_history) > 0

        # History should show improvement (non-increasing fitness)
        fitness_values = [f for _, f in result.convergence_history]
        for i in range(1, len(fitness_values)):
            # Allow small numerical tolerance
            assert fitness_values[i] <= fitness_values[i - 1] + 1e-10

        # Final fitness in history should match best fitness
        assert abs(fitness_values[-1] - result.best_fitness) < 1e-10

    def test_different_initialization_methods(self):
        """Test that different initialization methods work with BatchRunner."""
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=4)

        init_methods = ["uniform", "gaussian", "spendleySimplex", "pfefferSimplex"]

        for init_method in init_methods:
            results = runner.run_experiment(
                algorithm_class=NelderMead,
                problem=problem,
                num_runs=2,
                seeds=[1, 2],
                max_fes=1000,
                init_method=init_method,
            )

            # Should complete successfully
            assert len(results) == 2

            # Should find reasonable solutions
            for result in results:
                assert result.best_fitness < 10.0  # Should improve from random start

    def test_reproducibility_across_runs(self):
        """Test that results are reproducible with same configuration."""
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=3)

        config = {
            "algorithm_class": NelderMead,
            "problem": problem,
            "num_runs": 3,
            "seeds": [10, 20, 30],
            "max_fes": 1000,
            "init_method": "spendleySimplex",
        }

        # Run twice with same configuration
        results1 = runner.run_experiment(**config)
        results2 = runner.run_experiment(**config)

        # Results should be identical
        for r1, r2 in zip(results1, results2):
            assert r1.seed == r2.seed
            assert r1.best_fitness == r2.best_fitness
            assert np.allclose(r1.best_solution, r2.best_solution)
            assert r1.fes_used == r2.fes_used

    def test_performance_on_rosenbrock(self):
        """Test algorithm performance on Rosenbrock function."""
        runner = BatchRunner(verbose=False)
        problem = Rosenbrock(dim=5)

        results = runner.run_experiment(
            algorithm_class=AdaptiveNelderMead,
            problem=problem,
            num_runs=5,
            max_fes=5000,
        )

        # Rosenbrock is harder, but should still make progress
        for result in results:
            # Should improve significantly from random initialization
            # (random start typically has fitness > 1000 for dim=5)
            assert result.best_fitness < 100.0
            assert result.fes_used <= 5000

    def test_wall_time_tracking(self):
        """Test that wall time is tracked correctly."""
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=3)

        results = runner.run_experiment(
            algorithm_class=NelderMead, problem=problem, num_runs=3, max_fes=500
        )

        # All runs should have positive wall time
        for result in results:
            assert result.wall_time > 0
            # Should complete in reasonable time (< 1 second for small problem)
            assert result.wall_time < 5.0

    def test_batch_runner_executes_multiple_runs_correctly(self):
        """
        Integration test: Verify batch runner executes multiple runs correctly.
        
        Validates: Requirement 15.1 - Execute each algorithm on each problem 
        multiple times with different seeds
        """
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=4)
        
        # Run with 10 different seeds
        num_runs = 10
        seeds = list(range(100, 100 + num_runs))
        
        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=num_runs,
            seeds=seeds,
            max_fes=1500,
            init_method="spendleySimplex",
        )
        
        # Verify correct number of runs
        assert len(results) == num_runs
        
        # Verify each run has correct seed
        result_seeds = [r.seed for r in results]
        assert result_seeds == seeds
        
        # Verify all runs completed successfully
        for result in results:
            assert result.fes_used > 0
            # FES can exceed max_fes by simplex size (d+1) due to final iteration
            assert result.fes_used <= 1500 + problem.dim + 1
            assert result.best_fitness < np.inf
            assert result.wall_time > 0
            
        # Verify runs produced different results (stochastic behavior)
        fitness_values = [r.best_fitness for r in results]
        # Not all should be identical (unless all converged to exact optimum)
        if not all(f < 1e-12 for f in fitness_values):
            assert len(set(fitness_values)) > 1

    def test_results_recorded_in_correct_format(self):
        """
        Integration test: Verify results are recorded in correct structured format.
        
        Validates: Requirement 15.2 - Record results in a structured format
        """
        runner = BatchRunner(verbose=False)
        problem = Rosenbrock(dim=3)
        
        results = runner.run_experiment(
            algorithm_class=AdaptiveNelderMead,
            problem=problem,
            num_runs=5,
            seeds=[1, 2, 3, 4, 5],
            max_fes=2000,
        )
        
        # Verify each result has all required fields
        required_fields = [
            'algorithm_name', 'problem_name', 'seed', 'best_solution',
            'best_fitness', 'best_violation', 'fes_used', 'convergence_history',
            'wall_time', 'success', 'final_simplex'
        ]
        
        for result in results:
            for field in required_fields:
                assert hasattr(result, field), f"Missing field: {field}"
            
            # Verify field types
            assert isinstance(result.algorithm_name, str)
            assert isinstance(result.problem_name, str)
            assert isinstance(result.seed, int)
            assert isinstance(result.best_solution, np.ndarray)
            assert isinstance(result.best_fitness, float)
            assert isinstance(result.best_violation, float)
            assert isinstance(result.fes_used, int)
            assert isinstance(result.convergence_history, list)
            assert isinstance(result.wall_time, float)
            assert isinstance(result.success, bool)
            
            # Verify convergence history format
            for entry in result.convergence_history:
                assert isinstance(entry, tuple)
                assert len(entry) == 2
                assert isinstance(entry[0], int)  # FES
                assert isinstance(entry[1], float)  # Fitness
        
        # Verify export format
        export_dict = runner.export_results_to_dict(results)
        
        assert 'algorithm' in export_dict
        assert 'problem' in export_dict
        assert 'num_runs' in export_dict
        assert 'statistics' in export_dict
        assert 'runs' in export_dict
        
        assert export_dict['algorithm'] == 'AdaptiveNelderMead'
        assert export_dict['problem'] == 'Rosenbrock'
        assert export_dict['num_runs'] == 5
        assert len(export_dict['runs']) == 5

    def test_statistical_summaries_computed_correctly(self):
        """
        Integration test: Verify statistical summaries are computed correctly.
        
        Validates: Requirement 15.3 - Compute statistical summaries across runs
        Validates: Requirement 15.5 - Include mean, median, std, best, worst values
        """
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=5)
        
        # Run multiple times to get statistical variation
        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=20,
            max_fes=1000,
        )
        
        # Export and get statistics
        export_dict = runner.export_results_to_dict(results)
        stats = export_dict['statistics']
        
        # Verify all required statistics are present
        required_stats = ['mean', 'median', 'std', 'best', 'worst', 
                         'success_rate', 'avg_fes', 'avg_time']
        for stat in required_stats:
            assert stat in stats, f"Missing statistic: {stat}"
        
        # Manually compute statistics and verify correctness
        fitness_values = [r.best_fitness for r in results]
        
        # Verify mean
        expected_mean = np.mean(fitness_values)
        assert np.isclose(stats['mean'], expected_mean), \
            f"Mean mismatch: {stats['mean']} vs {expected_mean}"
        
        # Verify median
        expected_median = np.median(fitness_values)
        assert np.isclose(stats['median'], expected_median), \
            f"Median mismatch: {stats['median']} vs {expected_median}"
        
        # Verify std
        expected_std = np.std(fitness_values)
        assert np.isclose(stats['std'], expected_std), \
            f"Std mismatch: {stats['std']} vs {expected_std}"
        
        # Verify best (minimum)
        expected_best = np.min(fitness_values)
        assert np.isclose(stats['best'], expected_best), \
            f"Best mismatch: {stats['best']} vs {expected_best}"
        
        # Verify worst (maximum)
        expected_worst = np.max(fitness_values)
        assert np.isclose(stats['worst'], expected_worst), \
            f"Worst mismatch: {stats['worst']} vs {expected_worst}"
        
        # Verify success rate
        expected_success_rate = sum(r.success for r in results) / len(results)
        assert np.isclose(stats['success_rate'], expected_success_rate), \
            f"Success rate mismatch: {stats['success_rate']} vs {expected_success_rate}"
        
        # Verify avg_fes
        expected_avg_fes = np.mean([r.fes_used for r in results])
        assert np.isclose(stats['avg_fes'], expected_avg_fes), \
            f"Avg FES mismatch: {stats['avg_fes']} vs {expected_avg_fes}"
        
        # Verify avg_time
        expected_avg_time = np.mean([r.wall_time for r in results])
        assert np.isclose(stats['avg_time'], expected_avg_time), \
            f"Avg time mismatch: {stats['avg_time']} vs {expected_avg_time}"
        
        # Verify statistical relationships
        assert stats['best'] <= stats['mean'] <= stats['worst']
        assert stats['std'] >= 0
        assert 0 <= stats['success_rate'] <= 1

    def test_algorithm_comparison_functionality(self):
        """
        Integration test: Verify algorithm comparison functionality works correctly.
        
        Validates: Requirement 15.4 - Use consistent problem configurations 
        when comparing algorithms
        """
        runner = BatchRunner(verbose=False)
        
        # Define multiple algorithms with different configurations
        algorithm_configs = [
            {
                'class': NelderMead,
                'name': 'NM-Classic',
                'max_fes': 2000,
                'init_method': 'spendleySimplex',
                'delta_r': 1.0,
                'delta_e': 2.0,
            },
            {
                'class': AdaptiveNelderMead,
                'name': 'NM-Adaptive',
                'max_fes': 2000,
                'init_method': 'spendleySimplex',
            },
            {
                'class': NelderMead,
                'name': 'NM-Modified',
                'max_fes': 2000,
                'init_method': 'uniform',
                'delta_r': 1.5,
                'delta_e': 2.5,
            },
        ]
        
        # Define multiple problems
        problems = [
            Sphere(dim=5),
            Rosenbrock(dim=5),
        ]
        
        # Run comparison with consistent seeds
        num_runs = 5
        seeds = [10, 20, 30, 40, 50]
        
        results = runner.compare_algorithms(
            algorithm_configs=algorithm_configs,
            problems=problems,
            num_runs=num_runs,
            seeds=seeds,
        )
        
        # Verify structure: all algorithms present
        assert 'NM-Classic' in results
        assert 'NM-Adaptive' in results
        assert 'NM-Modified' in results
        
        # Verify all problems tested for each algorithm
        for alg_name in ['NM-Classic', 'NM-Adaptive', 'NM-Modified']:
            assert 'Sphere' in results[alg_name]
            assert 'Rosenbrock' in results[alg_name]
            
            # Verify correct number of runs
            assert len(results[alg_name]['Sphere']) == num_runs
            assert len(results[alg_name]['Rosenbrock']) == num_runs
        
        # Verify consistent seeds across algorithms (Requirement 15.4)
        for problem_name in ['Sphere', 'Rosenbrock']:
            # Get seeds from first algorithm
            reference_seeds = [r.seed for r in results['NM-Classic'][problem_name]]
            assert reference_seeds == seeds
            
            # Verify all other algorithms used same seeds
            for alg_name in ['NM-Adaptive', 'NM-Modified']:
                alg_seeds = [r.seed for r in results[alg_name][problem_name]]
                assert alg_seeds == seeds, \
                    f"Seeds mismatch for {alg_name} on {problem_name}"
        
        # Verify all runs completed successfully
        for alg_name in results:
            for problem_name in results[alg_name]:
                for result in results[alg_name][problem_name]:
                    assert result.fes_used > 0
                    # FES can exceed max_fes by simplex size due to final iteration
                    assert result.fes_used <= 2000 + 6  # dim=5, so simplex size = 6
                    assert result.best_fitness < np.inf
        
        # Verify algorithms can be compared (all tested on same problems)
        # Extract mean fitness for each algorithm on Sphere
        sphere_means = {}
        for alg_name in results:
            fitness_values = [r.best_fitness for r in results[alg_name]['Sphere']]
            sphere_means[alg_name] = np.mean(fitness_values)
        
        # All algorithms should find reasonable solutions on Sphere (easy problem)
        # Note: Some configurations might not converge perfectly with limited FES
        for alg_name, mean_fitness in sphere_means.items():
            assert mean_fitness < 100.0, \
                f"{alg_name} failed to make progress on Sphere: mean fitness = {mean_fitness}"

    def test_batch_runner_with_multiple_problem_dimensions(self):
        """
        Integration test: Verify batch runner works across different problem dimensions.
        """
        runner = BatchRunner(verbose=False)
        
        # Test on problems with different dimensions
        # Note: All Sphere problems have the same name, so we test them separately
        dimensions = [2, 5, 10]
        
        for dim in dimensions:
            problem = Sphere(dim=dim)
            
            results = runner.run_experiment(
                algorithm_class=NelderMead,
                problem=problem,
                num_runs=3,
                seeds=[1, 2, 3],
                max_fes=1500,
            )
            
            # All runs should complete
            assert len(results) == 3
            
            for result in results:
                assert result.best_solution.shape[0] == dim
                assert result.fes_used > 0
                # FES can exceed max_fes by simplex size
                assert result.fes_used <= 1500 + dim + 1

    def test_batch_runner_handles_algorithm_failures_gracefully(self):
        """
        Integration test: Verify batch runner handles algorithm failures gracefully.
        """
        runner = BatchRunner(verbose=False)
        problem = Sphere(dim=3)
        
        # Run with very low max_fes to potentially cause issues
        results = runner.run_experiment(
            algorithm_class=NelderMead,
            problem=problem,
            num_runs=3,
            seeds=[1, 2, 3],
            max_fes=10,  # Very low, might not complete properly
        )
        
        # Should still return results for all runs
        assert len(results) == 3
        
        # All results should have valid structure even if not optimal
        for result in results:
            assert result.fes_used <= 10
            assert isinstance(result.best_fitness, float)
            assert isinstance(result.best_solution, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
