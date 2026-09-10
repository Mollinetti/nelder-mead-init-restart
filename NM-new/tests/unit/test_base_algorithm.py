#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for BaseAlgorithm class.

Tests cover:
- Initialization and input validation
- Function evaluation tracking (FES)
- Best/worst solution tracking
- Barrier method integration
- Termination checking
- Error handling
"""

import pytest
import numpy as np
from nelder_mead.algorithms.base_algorithm import BaseAlgorithm
from nelder_mead.constraints.hard_barrier import HardBarrier


# Concrete implementation for testing
class DummyAlgorithm(BaseAlgorithm):
    """Minimal concrete implementation for testing BaseAlgorithm."""

    def run(self):
        """Simple run that just evaluates initial population."""
        self.population = self.initialize_population()
        self.fitness_values, _, _, _ = self.evaluate_population(self.population)


class TestBaseAlgorithmInitialization:
    """Test initialization and input validation."""

    def test_valid_initialization(self):
        """Test that valid parameters initialize correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=42,
        )

        assert alg.dim == 2
        assert alg.max_fes == 100
        assert alg.num_solutions == 3  # dim + 1
        assert alg.fes == 0
        assert alg.best_fitness == float("inf")
        assert alg.seed == 42

    def test_non_callable_objective_raises_error(self):
        """Test that non-callable objective function raises TypeError."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(TypeError, match="objective_fn must be callable"):
            DummyAlgorithm(
                objective_fn="not a function", lower_bounds=lower, upper_bounds=upper
            )

    def test_missing_bounds_raises_error(self):
        """Test that missing bounds raise ValueError."""

        def obj_fn(x):
            return np.sum(x**2)

        with pytest.raises(
            ValueError, match="lower_bounds and upper_bounds must be provided"
        ):
            DummyAlgorithm(objective_fn=obj_fn)

    def test_non_array_bounds_raises_error(self):
        """Test that non-array bounds raise TypeError."""

        def obj_fn(x):
            return np.sum(x**2)

        with pytest.raises(TypeError, match="must be numpy arrays"):
            DummyAlgorithm(
                objective_fn=obj_fn, lower_bounds=[0.0, 0.0], upper_bounds=[1.0, 1.0]
            )

    def test_mismatched_bounds_shape_raises_error(self):
        """Test that mismatched bound shapes raise ValueError."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="same shape"):
            DummyAlgorithm(objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper)

    def test_inconsistent_bounds_raises_error(self):
        """Test that lower >= upper raises ValueError."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([1.0, 0.0])
        upper = np.array([0.0, 1.0])

        with pytest.raises(
            ValueError, match="Lower bounds must be less than upper bounds"
        ):
            DummyAlgorithm(objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper)

    def test_invalid_max_fes_raises_error(self):
        """Test that non-positive max_fes raises ValueError."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="max_fes must be positive"):
            DummyAlgorithm(
                objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, max_fes=0
            )

    def test_invalid_num_solutions_raises_error(self):
        """Test that non-positive num_solutions raises ValueError."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="num_solutions must be positive"):
            DummyAlgorithm(
                objective_fn=obj_fn,
                lower_bounds=lower,
                upper_bounds=upper,
                num_solutions=0,
            )

    def test_max_fes_less_than_num_solutions_warns(self):
        """Test that max_fes < num_solutions issues warning and adjusts."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.warns(UserWarning, match="max_fes.*num_solutions"):
            alg = DummyAlgorithm(
                objective_fn=obj_fn,
                lower_bounds=lower,
                upper_bounds=upper,
                max_fes=2,
                num_solutions=5,
            )
            assert alg.max_fes == 5  # Adjusted to num_solutions

    def test_invalid_init_method_raises_error(self):
        """Test that invalid initialization method raises ValueError."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="not a valid initialization method"):
            DummyAlgorithm(
                objective_fn=obj_fn,
                lower_bounds=lower,
                upper_bounds=upper,
                init_method="invalid_method",
            )

    def test_default_num_solutions_is_dim_plus_one(self):
        """Test that default num_solutions is dim + 1."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        assert alg.num_solutions == 4  # 3 + 1


class TestFunctionEvaluationTracking:
    """Test function evaluation counter (FES) tracking."""

    def test_fes_starts_at_zero(self):
        """Test that FES counter starts at 0."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        assert alg.fes == 0

    def test_fes_increments_on_evaluate(self):
        """Test that FES increments on each evaluation."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        solution = np.array([0.5, 0.5])
        alg.evaluate(solution)
        assert alg.fes == 1

        alg.evaluate(solution)
        assert alg.fes == 2

    def test_fes_counts_initial_population(self):
        """Test that initial population evaluations count toward FES."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, num_solutions=3
        )

        population = alg.initialize_population()
        alg.evaluate_population(population)

        assert alg.fes == 3  # 3 solutions evaluated


class TestBestSolutionTracking:
    """Test best solution tracking."""

    def test_best_solution_updates_on_improvement(self):
        """Test that best solution updates when better solution is found."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Evaluate first solution
        sol1 = np.array([0.5, 0.5])
        alg.evaluate(sol1)
        assert alg.best_fitness == 0.5  # 0.5^2 + 0.5^2
        np.testing.assert_array_equal(alg.best_solution, sol1)

        # Evaluate better solution
        sol2 = np.array([0.1, 0.1])
        alg.evaluate(sol2)
        assert alg.best_fitness == pytest.approx(0.02)  # 0.1^2 + 0.1^2
        np.testing.assert_array_equal(alg.best_solution, sol2)

    def test_best_solution_does_not_update_on_worse(self):
        """Test that best solution doesn't update when worse solution is found."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Evaluate first solution
        sol1 = np.array([0.1, 0.1])
        alg.evaluate(sol1)
        best_fitness_before = alg.best_fitness
        best_solution_before = alg.best_solution.copy()

        # Evaluate worse solution
        sol2 = np.array([0.9, 0.9])
        alg.evaluate(sol2)

        assert alg.best_fitness == best_fitness_before
        np.testing.assert_array_equal(alg.best_solution, best_solution_before)

    def test_worst_fitness_tracking(self):
        """Test that worst fitness is tracked correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Evaluate solutions
        alg.evaluate(np.array([0.1, 0.1]))
        alg.evaluate(np.array([0.9, 0.9]))
        alg.evaluate(np.array([0.5, 0.5]))

        # Worst should be 0.9^2 + 0.9^2 = 1.62
        assert alg.worst_fitness == pytest.approx(1.62)


class TestConstraintHandling:
    """Test constraint handling and barrier integration."""

    def test_feasible_solution_comparison(self):
        """Test that feasible solutions are compared by fitness."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Both feasible: prefer lower fitness
        assert alg._is_better_solution(1.0, 0.0, 2.0, 0.0) is True
        assert alg._is_better_solution(2.0, 0.0, 1.0, 0.0) is False

    def test_feasible_vs_infeasible_comparison(self):
        """Test that feasible solution is preferred over infeasible."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Feasible vs infeasible: prefer feasible
        assert alg._is_better_solution(10.0, 0.0, 1.0, 5.0) is True
        assert alg._is_better_solution(1.0, 5.0, 10.0, 0.0) is False

    def test_infeasible_solution_comparison(self):
        """Test that infeasible solutions are compared by violation."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Both infeasible: prefer lower violation
        assert alg._is_better_solution(10.0, 1.0, 1.0, 5.0) is True
        assert alg._is_better_solution(1.0, 5.0, 10.0, 1.0) is False

    def test_barrier_integration(self):
        """Test that barrier method is integrated correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        barrier = HardBarrier(bounds=(lower, upper), num_solutions=3)

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, barrier=barrier
        )

        # Evaluate solution within bounds
        sol_in = np.array([0.5, 0.5])
        fitness_in, _, _, _ = alg.evaluate(sol_in)
        assert np.isfinite(fitness_in)

        # Evaluate solution outside bounds (should get penalty)
        sol_out = np.array([1.5, 0.5])
        fitness_out, _, _, _ = alg.evaluate(sol_out)
        # Hard barrier returns sys.maxsize for violations
        assert fitness_out > fitness_in


class TestTerminationChecking:
    """Test termination condition checking."""

    def test_should_terminate_when_fes_reached(self):
        """Test that algorithm terminates when FES limit is reached."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, max_fes=5
        )

        assert alg.should_terminate() is False

        # Evaluate 5 times
        solution = np.array([0.5, 0.5])
        for _ in range(5):
            alg.evaluate(solution)

        assert alg.should_terminate() is True

    def test_should_not_terminate_before_fes_limit(self):
        """Test that algorithm doesn't terminate before FES limit."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, max_fes=10
        )

        solution = np.array([0.5, 0.5])
        for _ in range(5):
            alg.evaluate(solution)

        assert alg.should_terminate() is False


class TestConvergenceHistory:
    """Test convergence history tracking."""

    def test_convergence_history_records_at_intervals(self):
        """Test that convergence history is recorded at intervals."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, max_fes=100
        )

        # Tracking interval should be max_fes // 100 = 1
        assert alg.tracking_interval == 1

        # Evaluate some solutions
        for i in range(10):
            alg.evaluate(np.array([0.1 * i, 0.1 * i]))

        # Should have recorded at each evaluation
        assert len(alg.memory_global_best) == 10

        # Check format: list of (fes, best_fitness) tuples
        assert all(
            isinstance(entry, tuple) and len(entry) == 2
            for entry in alg.memory_global_best
        )

    def test_convergence_history_records_first_evaluation(self):
        """Test that first evaluation is always recorded."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, max_fes=1000
        )

        # Evaluate one solution
        alg.evaluate(np.array([0.5, 0.5]))

        # Should have recorded first evaluation
        assert len(alg.memory_global_best) >= 1
        assert alg.memory_global_best[0][0] == 1  # FES = 1


class TestErrorHandling:
    """Test error handling for objective and constraint functions."""

    def test_objective_function_exception_handling(self):
        """Test that objective function exceptions are handled gracefully."""

        def bad_obj_fn(x):
            raise ValueError("Intentional error")

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=bad_obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        with pytest.warns(UserWarning, match="Objective function evaluation failed"):
            fitness, _, _, _ = alg.evaluate(np.array([0.5, 0.5]))
            assert fitness == 1e10  # Large penalty

    def test_objective_function_nan_handling(self):
        """Test that NaN objective values are handled."""

        def nan_obj_fn(x):
            return np.nan

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=nan_obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        with pytest.warns(UserWarning, match="non-finite value"):
            fitness, _, _, _ = alg.evaluate(np.array([0.5, 0.5]))
            assert fitness == 1e10  # Large penalty

    def test_objective_function_inf_handling(self):
        """Test that Inf objective values are handled."""

        def inf_obj_fn(x):
            return float("inf")

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=inf_obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        with pytest.warns(UserWarning, match="non-finite value"):
            fitness, _, _, _ = alg.evaluate(np.array([0.5, 0.5]))
            assert fitness == 1e10  # Large penalty

    def test_constraint_function_exception_handling(self):
        """Test that constraint function exceptions are handled."""

        def obj_fn(x):
            return np.sum(x**2)

        def bad_constraint_fn(x):
            raise ValueError("Intentional error")

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn,
            constraint_eq_fn=bad_constraint_fn,
            lower_bounds=lower,
            upper_bounds=upper,
        )

        with pytest.warns(UserWarning, match="Equality constraint evaluation failed"):
            fitness, eq_vio, _, _ = alg.evaluate(np.array([0.5, 0.5]))
            assert len(eq_vio) > 0
            assert eq_vio[0] == 1e6  # Maximum violation


class TestGetBestSolution:
    """Test get_best_solution method."""

    def test_get_best_solution_returns_correct_format(self):
        """Test that get_best_solution returns correct tuple format."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Evaluate a solution
        sol = np.array([0.5, 0.5])
        alg.evaluate(sol)

        # Get best solution
        best_sol, best_fit, eq_vio, ineq_vio, total_vio = alg.get_best_solution()

        assert isinstance(best_sol, np.ndarray)
        assert isinstance(best_fit, float)
        assert isinstance(eq_vio, np.ndarray)
        assert isinstance(ineq_vio, np.ndarray)
        assert isinstance(total_vio, float)

        np.testing.assert_array_equal(best_sol, sol)
        assert best_fit == pytest.approx(0.5)


class TestPopulationInitialization:
    """Test population initialization."""

    def test_initialize_population_returns_correct_shape(self):
        """Test that initialize_population returns correct shape."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            num_solutions=3,  # Must be dim+1 for spendleySimplex
            init_method="spendleySimplex",
        )

        population = alg.initialize_population()

        assert population.shape == (3, 2)

    def test_initialize_population_within_bounds(self):
        """Test that initialized population is within bounds."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            init_method="uniform",
        )

        population = alg.initialize_population()

        assert np.all(population >= lower)
        assert np.all(population <= upper)

    def test_evaluate_population_returns_correct_format(self):
        """Test that evaluate_population returns correct format."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = DummyAlgorithm(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, num_solutions=3
        )

        population = alg.initialize_population()
        fitness, eq_vio_list, ineq_vio_list, total_vio = alg.evaluate_population(
            population
        )

        assert fitness.shape == (3,)
        assert len(eq_vio_list) == 3
        assert len(ineq_vio_list) == 3
        assert total_vio.shape == (3,)


class TestReproducibility:
    """Test reproducibility with fixed seed."""

    def test_same_seed_produces_same_initialization(self):
        """Test that same seed produces identical initialization."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg1 = DummyAlgorithm(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            init_method="uniform",
            seed=42,
        )

        alg2 = DummyAlgorithm(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            init_method="uniform",
            seed=42,
        )

        pop1 = alg1.initialize_population()
        pop2 = alg2.initialize_population()

        np.testing.assert_array_almost_equal(pop1, pop2)

    def test_different_seed_produces_different_initialization(self):
        """Test that different seeds produce different initialization."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg1 = DummyAlgorithm(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            init_method="uniform",
            seed=42,
        )

        alg2 = DummyAlgorithm(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            init_method="uniform",
            seed=123,
        )

        pop1 = alg1.initialize_population()
        pop2 = alg2.initialize_population()

        # Should be different (with very high probability)
        assert not np.allclose(pop1, pop2)
