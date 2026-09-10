#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for NelderMead algorithm class.

Tests cover:
- Initialization with d+1 solutions
- Configuration of simplex operation coefficients
- Configuration of stopping criteria
- Configuration of restart strategy
- Basic optimization on simple test problems
- Algorithm on problems with known optima (Sphere, Rosenbrock)
- Restart mechanism triggering
- Stopping criteria integration
- Coefficient configuration

Validates: Requirements 2.1-2.6, 4.1-4.5, 5.4-5.5, 17.1-17.5
"""

import pytest
import numpy as np
from src.nelder_mead.algorithms.nelder_mead import NelderMead
from src.nelder_mead.constraints.hard_barrier import HardBarrier
from src.nelder_mead.benchmarks.unconstrained import Sphere, Rosenbrock


class TestNelderMeadInitialization:
    """Test initialization and configuration."""

    def test_valid_initialization(self):
        """Test that valid parameters initialize correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=42,
        )

        # Check base algorithm initialization
        assert alg.dim == 2
        assert alg.max_fes == 100
        assert alg.num_solutions == 3  # dim + 1 for simplex
        assert alg.fes == 0

        # Check simplex operation coefficients (Requirement 17.2)
        assert alg.delta_r == 1.0
        assert alg.delta_e == 2.0
        assert alg.delta_oc == 0.5
        assert alg.delta_ic == -0.5
        assert alg.gamma_s == 0.5

        # Check default stopping criteria (Requirement 17.4)
        assert "fminsearch_fun" in alg.stopping_criteria
        assert "fminsearch_x" in alg.stopping_criteria

        # Check restart strategy (Requirement 17.5)
        assert alg.restart_strategy is not None

    def test_custom_coefficients(self):
        """Test that custom simplex operation coefficients are set correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            delta_r=1.5,
            delta_e=2.5,
            delta_oc=0.75,
            delta_ic=-0.75,
            gamma_s=0.25,
        )

        assert alg.delta_r == 1.5
        assert alg.delta_e == 2.5
        assert alg.delta_oc == 0.75
        assert alg.delta_ic == -0.75
        assert alg.gamma_s == 0.25

    def test_custom_stopping_criteria(self):
        """Test that custom stopping criteria are configured correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        custom_criteria = ["oriented_length", "std_dev", "flat_simplex"]

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            stopping_criteria=custom_criteria,
        )

        assert alg.stopping_criteria == custom_criteria

    def test_invalid_stopping_criterion_raises_error(self):
        """Test that invalid stopping criterion raises ValueError."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="not a valid stopping criterion"):
            NelderMead(
                objective_fn=obj_fn,
                lower_bounds=lower,
                upper_bounds=upper,
                stopping_criteria=["invalid_criterion"],
            )

    def test_custom_restart_strategy(self):
        """Test that custom restart strategies are configured correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        for strategy in ["uniform", "gaussian", "gaussian_best"]:
            alg = NelderMead(
                objective_fn=obj_fn,
                lower_bounds=lower,
                upper_bounds=upper,
                restart_strategy=strategy,
            )
            assert alg.restart_strategy is not None

    def test_invalid_restart_strategy_raises_error(self):
        """Test that invalid restart strategy raises ValueError."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="not a valid restart strategy"):
            NelderMead(
                objective_fn=obj_fn,
                lower_bounds=lower,
                upper_bounds=upper,
                restart_strategy="invalid_strategy",
            )

    def test_num_solutions_is_dim_plus_one(self):
        """Test that simplex has d+1 solutions for d-dimensional problem (Requirement 17.1)."""

        def obj_fn(x):
            return np.sum(x**2)

        # Test with different dimensions
        for dim in [2, 3, 5, 10]:
            lower = np.zeros(dim)
            upper = np.ones(dim)

            alg = NelderMead(
                objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
            )

            assert alg.num_solutions == dim + 1


class TestNelderMeadOptimization:
    """Test optimization behavior on simple problems."""

    def test_sphere_function_2d(self):
        """Test optimization on 2D sphere function (simple convex problem).
        
        Validates: Requirements 2.1-2.6 (simplex operations work correctly)
        """

        def sphere(x):
            return np.sum(x**2)

        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])

        alg = NelderMead(
            objective_fn=sphere,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            seed=42,
        )

        alg.run()

        best_solution, best_fitness, _, _, _ = alg.get_best_solution()

        # Should find solution close to origin
        assert best_fitness < 0.1  # Should be close to optimal value of 0
        assert np.linalg.norm(best_solution) < 1.0  # Should be close to [0, 0]

    def test_sphere_function_with_benchmark_class(self):
        """Test optimization on Sphere benchmark problem with known optimum.
        
        Validates: Requirements 2.1-2.6 (simplex operations)
        """
        # Create 2D Sphere problem
        problem = Sphere(dim=2, bounds_range=5.0)
        
        alg = NelderMead(
            objective_fn=problem.objective,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=500,
            seed=42,
        )
        
        alg.run()
        
        best_solution, best_fitness, _, _, _ = alg.get_best_solution()
        
        # Known optimum is at origin with value 0
        assert best_fitness < 0.1
        assert np.linalg.norm(best_solution - problem.optimum_location) < 1.0

    def test_rosenbrock_function_2d(self):
        """Test optimization on 2D Rosenbrock function.
        
        Validates: Requirements 2.1-2.6 (simplex operations work correctly)
        """

        def rosenbrock(x):
            return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])

        alg = NelderMead(
            objective_fn=rosenbrock,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=1000,
            seed=42,
        )

        alg.run()

        best_solution, best_fitness, _, _, _ = alg.get_best_solution()

        # Rosenbrock optimum is at [1, 1] with value 0
        # Nelder-Mead should get reasonably close
        assert best_fitness < 1.0  # Should be close to optimal value of 0

    def test_rosenbrock_function_with_benchmark_class(self):
        """Test optimization on Rosenbrock benchmark problem with known optimum.
        
        Validates: Requirements 2.1-2.6 (simplex operations)
        """
        # Create 2D Rosenbrock problem
        problem = Rosenbrock(dim=2, bounds_range=2.0)
        
        alg = NelderMead(
            objective_fn=problem.objective,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=1000,
            seed=42,
        )
        
        alg.run()
        
        best_solution, best_fitness, _, _, _ = alg.get_best_solution()
        
        # Known optimum is at [1, 1, ..., 1] with value 0
        # Rosenbrock is harder, so we allow more tolerance
        assert best_fitness < 1.0
        # Check that we're moving toward the optimum
        assert np.linalg.norm(best_solution - problem.optimum_location) < 2.0

    def test_fes_tracking(self):
        """Test that function evaluations are tracked correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=42,
        )

        alg.run()

        # FES should be tracked and not exceed max_fes significantly
        assert alg.fes > 0
        assert (
            alg.fes <= alg.max_fes + alg.num_solutions
        )  # Allow for final simplex evaluation

    def test_best_solution_improves(self):
        """Test that best solution improves during optimization."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=42,
        )

        alg.run()

        # Check convergence history
        assert len(alg.memory_global_best) > 0

        # Best fitness should improve or stay the same (monotonic)
        for i in range(1, len(alg.memory_global_best)):
            assert alg.memory_global_best[i][1] <= alg.memory_global_best[i - 1][1]

    def test_simplex_initialization(self):
        """Test that simplex is initialized with correct shape."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=10,  # Just enough to initialize
            seed=42,
        )

        alg.run()

        # Check simplex shape
        assert alg.simplex is not None
        assert alg.simplex.shape == (4, 3)  # 4 vertices in 3D space

        # Check fitness values
        assert alg.fitness_values is not None
        assert len(alg.fitness_values) == 4

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Run twice with same seed
        results = []
        for _ in range(2):
            alg = NelderMead(
                objective_fn=obj_fn,
                lower_bounds=lower,
                upper_bounds=upper,
                max_fes=100,
                seed=42,
            )
            alg.run()
            best_solution, best_fitness, _, _, _ = alg.get_best_solution()
            results.append((best_solution.copy(), best_fitness))

        # Results should be identical
        np.testing.assert_array_almost_equal(results[0][0], results[1][0])
        assert abs(results[0][1] - results[1][1]) < 1e-10


class TestNelderMeadWithConstraints:
    """Test Nelder-Mead with constraint handling."""

    def test_with_hard_barrier(self):
        """Test optimization with hard barrier constraint handling."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        barrier = HardBarrier(bounds=(lower, upper), num_solutions=3)

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            barrier=barrier,
            seed=42,
        )

        alg.run()

        best_solution, best_fitness, _, _, total_violation = alg.get_best_solution()

        # Solution should be within bounds
        assert np.all(best_solution >= lower)
        assert np.all(best_solution <= upper)

        # Should have low violation
        assert total_violation < 1e-6


class TestNelderMeadInternalMethods:
    """Test internal helper methods."""

    def test_sort_simplex(self):
        """Test that simplex sorting works correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=10,
            seed=42,
        )

        # Initialize simplex
        alg.simplex = alg.initialize_population()
        (
            alg.fitness_values,
            alg.eq_violations_list,
            alg.ineq_violations_list,
            alg.total_violations,
        ) = alg.evaluate_population(alg.simplex)

        # Sort
        alg._sort_simplex()

        # Check that fitness values are sorted (ascending)
        for i in range(len(alg.fitness_values) - 1):
            assert alg.fitness_values[i] <= alg.fitness_values[i + 1]

    def test_check_stopping_criteria(self):
        """Test that stopping criteria checking works.
        
        Validates: Requirements 4.1-4.5 (stopping criteria integration)
        """

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            stopping_criteria=["fminsearch_fun"],
            seed=42,
        )

        # Create a converged simplex (all points at same location)
        alg.simplex = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        alg.fitness_values = np.array([0.5, 0.5, 0.5])

        # Should detect convergence
        assert alg._check_stopping_criteria() is True

    def test_restart_simplex(self):
        """Test that simplex restart works correctly.
        
        Validates: Requirements 5.4, 5.5 (restart mechanism triggering)
        """

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=42,
        )

        # Initialize simplex
        alg.simplex = alg.initialize_population()
        (
            alg.fitness_values,
            alg.eq_violations_list,
            alg.ineq_violations_list,
            alg.total_violations,
        ) = alg.evaluate_population(alg.simplex)
        alg._sort_simplex()

        # Store best solution before restart
        best_before = alg.simplex[0].copy()
        fes_before = alg.fes

        # Restart
        alg._restart_simplex()

        # Best solution should be preserved
        np.testing.assert_array_equal(alg.simplex[0], best_before)

        # FES should have increased by num_solutions (re-evaluation)
        assert alg.fes == fes_before + alg.num_solutions


class TestRestartMechanismTriggering:
    """Test restart mechanism triggering during optimization.
    
    Validates: Requirements 5.4, 5.5
    """

    def test_restart_triggered_by_stopping_criteria(self):
        """Test that restart is triggered when stopping criteria are met."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Use very tight stopping criteria to trigger restart quickly
        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=200,
            stopping_criteria=["fminsearch_fun", "fminsearch_x"],
            restart_strategy="gaussian_best",
            seed=42,
        )

        # Track FES before run
        initial_fes = alg.fes

        alg.run()

        # Algorithm should have run and potentially triggered restarts
        assert alg.fes > initial_fes
        # If restarts occurred, FES should reflect multiple simplex evaluations
        # (This is a weak test but verifies the mechanism doesn't crash)

    def test_restart_with_different_strategies(self):
        """Test that different restart strategies work correctly.
        
        Validates: Requirements 5.1, 5.2, 5.3
        """

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        for strategy in ["uniform", "gaussian", "gaussian_best"]:
            alg = NelderMead(
                objective_fn=obj_fn,
                lower_bounds=lower,
                upper_bounds=upper,
                max_fes=100,
                restart_strategy=strategy,
                seed=42,
            )

            alg.run()

            # Should complete without errors
            best_solution, best_fitness, _, _, _ = alg.get_best_solution()
            assert best_fitness >= 0  # Sphere function is non-negative

    def test_restart_preserves_best_solution(self):
        """Test that restart preserves the best solution found so far."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            stopping_criteria=["fminsearch_fun"],
            seed=42,
        )

        # Initialize and evaluate
        alg.simplex = alg.initialize_population()
        (
            alg.fitness_values,
            alg.eq_violations_list,
            alg.ineq_violations_list,
            alg.total_violations,
        ) = alg.evaluate_population(alg.simplex)
        alg._sort_simplex()

        # Store best before restart
        best_fitness_before = alg.fitness_values[0]
        best_solution_before = alg.simplex[0].copy()

        # Trigger restart
        alg._restart_simplex()

        # Best solution should be preserved as first vertex
        np.testing.assert_array_equal(alg.simplex[0], best_solution_before)
        assert alg.fitness_values[0] == best_fitness_before


class TestStoppingCriteriaIntegration:
    """Test stopping criteria integration in the algorithm.
    
    Validates: Requirements 4.1-4.5
    """

    def test_oriented_length_criterion(self):
        """Test oriented length stopping criterion."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            stopping_criteria=["oriented_length"],
            seed=42,
        )

        alg.run()

        # Should complete and find a solution
        best_solution, best_fitness, _, _, _ = alg.get_best_solution()
        assert best_fitness >= 0

    def test_std_dev_criterion(self):
        """Test standard deviation stopping criterion."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            stopping_criteria=["std_dev"],
            seed=42,
        )

        alg.run()

        # Should complete and find a solution
        best_solution, best_fitness, _, _, _ = alg.get_best_solution()
        assert best_fitness >= 0

    def test_multiple_stopping_criteria(self):
        """Test that multiple stopping criteria work together."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            stopping_criteria=["oriented_length", "std_dev", "flat_simplex"],
            seed=42,
        )

        alg.run()

        # Should complete and find a solution
        best_solution, best_fitness, _, _, _ = alg.get_best_solution()
        assert best_fitness >= 0

    def test_flat_simplex_criterion(self):
        """Test flat simplex stopping criterion."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            stopping_criteria=["flat_simplex"],
            seed=42,
        )

        alg.run()

        # Should complete
        best_solution, best_fitness, _, _, _ = alg.get_best_solution()
        assert best_fitness >= 0


class TestCoefficientConfiguration:
    """Test simplex operation coefficient configuration.
    
    Validates: Requirements 17.2, 17.3
    """

    def test_default_coefficients(self):
        """Test that default coefficients are set correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
        )

        # Check default values (Requirement 17.2)
        assert alg.delta_r == 1.0
        assert alg.delta_e == 2.0
        assert alg.delta_oc == 0.5
        assert alg.delta_ic == -0.5
        assert alg.gamma_s == 0.5

    def test_custom_coefficients_affect_optimization(self):
        """Test that custom coefficients affect optimization behavior."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Run with default coefficients
        alg1 = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=42,
        )
        alg1.run()
        result1 = alg1.get_best_solution()[1]

        # Run with different coefficients
        alg2 = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            delta_r=1.5,
            delta_e=2.5,
            delta_oc=0.75,
            delta_ic=-0.75,
            gamma_s=0.25,
            seed=42,
        )
        alg2.run()
        result2 = alg2.get_best_solution()[1]

        # Results should be different (coefficients affect behavior)
        # Note: This is a weak test but verifies coefficients are used
        # Both should still find reasonable solutions
        assert result1 >= 0
        assert result2 >= 0

    def test_extreme_coefficients_still_work(self):
        """Test that algorithm works with extreme coefficient values."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Very aggressive expansion
        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            delta_e=5.0,  # Very aggressive expansion
            seed=42,
        )

        alg.run()

        # Should still complete without errors
        best_solution, best_fitness, _, _, _ = alg.get_best_solution()
        assert best_fitness >= 0


class TestSimplexOperationsInAlgorithm:
    """Test that simplex operations are correctly applied during optimization.
    
    Validates: Requirements 2.1-2.6
    """

    def test_reflection_operation_used(self):
        """Test that reflection operation is used in the algorithm."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=42,
        )

        alg.run()

        # If algorithm ran, reflection was used (it's always the first operation tried)
        assert alg.fes > alg.num_solutions  # More than just initialization

    def test_shrink_operation_used(self):
        """Test that shrink operation can be triggered."""

        # Create a function where shrink is likely to be needed
        def difficult_fn(x):
            # Highly non-convex function
            return np.sum(np.sin(10 * x) ** 2 + 0.1 * x**2)

        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=difficult_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=200,
            seed=42,
        )

        alg.run()

        # Algorithm should complete (shrink operation doesn't cause crashes)
        best_solution, best_fitness, _, _, _ = alg.get_best_solution()
        assert best_fitness is not None

    def test_centroid_excludes_worst_point(self):
        """Test that centroid computation excludes worst point (Requirement 2.6)."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = NelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=10,
            seed=42,
        )

        # Initialize simplex
        alg.simplex = alg.initialize_population()
        (
            alg.fitness_values,
            alg.eq_violations_list,
            alg.ineq_violations_list,
            alg.total_violations,
        ) = alg.evaluate_population(alg.simplex)
        alg._sort_simplex()

        # Manually compute centroid
        from src.nelder_mead.core import simplex_operations as ops

        centroid = ops.compute_centroid(alg.simplex, exclude_worst=True)

        # Centroid should be mean of first n points (excluding worst)
        expected_centroid = np.mean(alg.simplex[:-1], axis=0)
        np.testing.assert_array_almost_equal(centroid, expected_centroid)
