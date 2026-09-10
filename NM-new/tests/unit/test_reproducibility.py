#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for reproducibility and comprehensive edge cases.

This module tests:
- Requirement 18.1: Reproducibility with fixed seeds
- Requirement 19.1-19.5: Error handling for invalid inputs and edge cases
- Known simplex configurations with expected outputs
- Numerical edge cases

Task 8: Implement reproducibility and add comprehensive unit tests
"""

import unittest
import numpy as np
import warnings

from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
from nelder_mead.constraints.hard_barrier import HardBarrier
from nelder_mead.initialization.simplex_init import SimplexInitializer
from nelder_mead.core.restart_strategies import create_restart_strategy


class TestReproducibilityWithSeed(unittest.TestCase):
    """
    Test that algorithms produce identical results with the same seed.

    Validates: Requirement 18.1
    """

    def test_nelder_mead_reproducibility_simple_function(self):
        """Test that NelderMead produces identical results with same seed."""

        # Simple quadratic function
        def objective(x):
            return np.sum(x**2)

        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])

        # Run 1
        alg1 = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=42,
            init_method="uniform",
        )
        alg1.run()
        result1 = alg1.get_best_solution()

        # Run 2 with same seed
        alg2 = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=42,
            init_method="uniform",
        )
        alg2.run()
        result2 = alg2.get_best_solution()

        # Results should be identical
        np.testing.assert_array_equal(
            result1[0], result2[0], err_msg="Solutions differ with same seed"
        )
        self.assertEqual(result1[1], result2[1], "Fitness values differ with same seed")
        self.assertEqual(alg1.fes, alg2.fes, "FES counts differ with same seed")

    def test_adaptive_nm_reproducibility(self):
        """Test that AdaptiveNelderMead produces identical results with same seed."""

        def objective(x):
            return np.sum((x - 1.0) ** 2)

        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([2.0, 2.0, 2.0])

        # Run 1
        alg1 = AdaptiveNelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=150,
            seed=123,
            init_method="gaussian",
        )
        alg1.run()
        result1 = alg1.get_best_solution()

        # Run 2 with same seed
        alg2 = AdaptiveNelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=150,
            seed=123,
            init_method="gaussian",
        )
        alg2.run()
        result2 = alg2.get_best_solution()

        # Results should be identical
        np.testing.assert_array_equal(result1[0], result2[0])
        self.assertEqual(result1[1], result2[1])

    def test_reproducibility_with_restart(self):
        """Test reproducibility when restart mechanism is triggered."""

        # Function that triggers restart (flat simplex)
        def objective(x):
            return 0.0  # Constant function triggers flat simplex criterion

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Run 1
        alg1 = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=999,
            stopping_criteria=["flat_simplex"],
            restart_strategy="gaussian_best",
        )
        alg1.run()

        # Run 2 with same seed
        alg2 = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=999,
            stopping_criteria=["flat_simplex"],
            restart_strategy="gaussian_best",
        )
        alg2.run()

        # Should have same FES count (same number of restarts)
        self.assertEqual(alg1.fes, alg2.fes)

    def test_reproducibility_with_constraints(self):
        """Test reproducibility with constraint handling."""

        def objective(x):
            return x[0] ** 2 + x[1] ** 2

        def constraint_ineq(x):
            # x[0] + x[1] <= 1
            return np.array([x[0] + x[1] - 1.0])

        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 2.0])
        barrier = HardBarrier(bounds=(lower, upper), num_solutions=3)

        # Run 1
        alg1 = NelderMead(
            objective_fn=objective,
            constraint_ineq_fn=constraint_ineq,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=777,
            barrier=barrier,
        )
        alg1.run()
        result1 = alg1.get_best_solution()

        # Run 2 with same seed
        barrier2 = HardBarrier(bounds=(lower, upper), num_solutions=3)
        alg2 = NelderMead(
            objective_fn=objective,
            constraint_ineq_fn=constraint_ineq,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=777,
            barrier=barrier2,
        )
        alg2.run()
        result2 = alg2.get_best_solution()

        # Results should be identical
        np.testing.assert_array_almost_equal(result1[0], result2[0])
        self.assertAlmostEqual(result1[1], result2[1])

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])

        # Run with seed 1
        alg1 = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=1,
            init_method="uniform",
        )
        alg1.run()
        result1 = alg1.get_best_solution()

        # Run with seed 2
        alg2 = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=2,
            init_method="uniform",
        )
        alg2.run()
        result2 = alg2.get_best_solution()

        # Results should be different (with high probability)
        # At least the initial population should differ
        self.assertFalse(
            np.array_equal(result1[0], result2[0]),
            "Different seeds should produce different results",
        )

    def test_convergence_history_reproducibility(self):
        """Test that convergence history is identical with same seed."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Run 1
        alg1 = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=555,
        )
        alg1.run()
        history1 = alg1.memory_global_best

        # Run 2 with same seed
        alg2 = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            seed=555,
        )
        alg2.run()
        history2 = alg2.memory_global_best

        # Histories should be identical
        self.assertEqual(len(history1), len(history2))
        for (fes1, fitness1), (fes2, fitness2) in zip(history1, history2):
            self.assertEqual(fes1, fes2)
            self.assertEqual(fitness1, fitness2)


class TestKnownSimplexConfigurations(unittest.TestCase):
    """
    Test specific simplex configurations with known expected outputs.

    Validates: Requirement 19.1-19.5 (specific examples)
    """

    def test_sphere_function_at_origin(self):
        """Test that sphere function at origin returns 0."""

        def sphere(x):
            return np.sum(x**2)

        origin = np.array([0.0, 0.0, 0.0])
        self.assertEqual(sphere(origin), 0.0)

    def test_rosenbrock_at_optimum(self):
        """Test that Rosenbrock function at (1,1) returns 0."""

        def rosenbrock(x):
            return sum(
                100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)
            )

        optimum = np.array([1.0, 1.0])
        self.assertAlmostEqual(rosenbrock(optimum), 0.0, places=10)

    def test_regular_simplex_geometry(self):
        """Test that Spendley simplex has regular geometry."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        x0 = np.array([5.0, 5.0])

        simplex = SimplexInitializer.spendley_simplex(
            num_solutions=3, lower_bounds=lower, upper_bounds=upper, dim=2, x0=x0
        )

        # All edges from first vertex should have equal length
        first_vertex = simplex[0]
        edge_lengths = [
            np.linalg.norm(simplex[i] - first_vertex) for i in range(1, len(simplex))
        ]

        # Check all edges are approximately equal
        for length in edge_lengths[1:]:
            self.assertAlmostEqual(length, edge_lengths[0], places=10)

    def test_simplex_within_bounds(self):
        """Test that all initialization methods respect bounds."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        rng = np.random.default_rng(42)

        methods = [
            "uniform",
            "gaussian",
            "spendleySimplex",
            "pfefferSimplex",
            "adaptiveSimplex",
        ]

        for method in methods:
            with self.subTest(method=method):
                if method in ["uniform", "gaussian"]:
                    simplex = SimplexInitializer.initialize(
                        method, 3, lower, upper, 2, rng=rng
                    )
                else:
                    simplex = SimplexInitializer.initialize(
                        method, 3, lower, upper, 2, x0=np.array([0.5, 0.5])
                    )

                # Check all points are within bounds
                self.assertTrue(np.all(simplex >= lower))
                self.assertTrue(np.all(simplex <= upper))


class TestErrorHandling(unittest.TestCase):
    """
    Test error handling for invalid inputs and edge cases.

    Validates: Requirement 19.1-19.5
    """

    def test_invalid_initialization_method(self):
        """Test that invalid initialization method raises error."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with self.assertRaises(ValueError) as context:
            NelderMead(
                objective_fn=objective,
                lower_bounds=lower,
                upper_bounds=upper,
                init_method="invalid_method",
            )

        self.assertIn("not a valid initialization method", str(context.exception))

    def test_invalid_restart_strategy(self):
        """Test that invalid restart strategy raises error."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with self.assertRaises(ValueError) as context:
            NelderMead(
                objective_fn=objective,
                lower_bounds=lower,
                upper_bounds=upper,
                restart_strategy="invalid_strategy",
            )

        self.assertIn("not a valid restart strategy", str(context.exception))

    def test_inconsistent_bounds(self):
        """Test that inconsistent bounds raise error."""

        def objective(x):
            return np.sum(x**2)

        # Lower bound > upper bound
        lower = np.array([1.0, 1.0])
        upper = np.array([0.0, 0.0])

        with self.assertRaises(ValueError) as context:
            NelderMead(objective_fn=objective, lower_bounds=lower, upper_bounds=upper)

        self.assertIn(
            "Lower bounds must be less than upper bounds", str(context.exception)
        )

    def test_mismatched_bounds_dimensions(self):
        """Test that mismatched bounds dimensions raise error."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])  # Different dimension

        with self.assertRaises(ValueError) as context:
            NelderMead(objective_fn=objective, lower_bounds=lower, upper_bounds=upper)

        self.assertIn("same shape", str(context.exception))

    def test_non_callable_objective(self):
        """Test that non-callable objective raises error."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with self.assertRaises(TypeError) as context:
            NelderMead(
                objective_fn="not a function", lower_bounds=lower, upper_bounds=upper
            )

        self.assertIn("must be callable", str(context.exception))

    def test_invalid_stopping_criterion(self):
        """Test that invalid stopping criterion raises error."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with self.assertRaises(ValueError) as context:
            NelderMead(
                objective_fn=objective,
                lower_bounds=lower,
                upper_bounds=upper,
                stopping_criteria=["invalid_criterion"],
            )

        self.assertIn("not a valid stopping criterion", str(context.exception))

    def test_max_fes_less_than_num_solutions_warns(self):
        """Test that max_fes < num_solutions issues warning."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            alg = NelderMead(
                objective_fn=objective,
                lower_bounds=lower,
                upper_bounds=upper,
                max_fes=2,  # Less than dim+1 = 3
                num_solutions=3,
            )

            # Check that a warning was issued
            self.assertEqual(len(w), 1)
            self.assertIn("max_fes", str(w[0].message))

            # Check that max_fes was adjusted
            self.assertEqual(alg.max_fes, 3)


class TestNumericalEdgeCases(unittest.TestCase):
    """
    Test numerical edge cases and boundary conditions.

    Validates: Requirement 19.1-19.5 (numerical edge cases)
    """

    def test_very_small_bounds(self):
        """Test optimization with very small bounds."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1e-6, 1e-6])

        alg = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=42,
        )
        alg.run()

        # Should complete without errors
        result = alg.get_best_solution()
        self.assertIsNotNone(result[0])

    def test_very_large_bounds(self):
        """Test optimization with very large bounds."""

        def objective(x):
            return np.sum(x**2)

        lower = np.array([-1e6, -1e6])
        upper = np.array([1e6, 1e6])

        alg = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=42,
        )
        alg.run()

        # Should complete without errors
        result = alg.get_best_solution()
        self.assertIsNotNone(result[0])

    def test_asymmetric_bounds(self):
        """Test optimization with highly asymmetric bounds."""

        def objective(x):
            return x[0] ** 2 + x[1] ** 2

        lower = np.array([-1000.0, -0.001])
        upper = np.array([1000.0, 0.001])

        alg = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=42,
        )
        alg.run()

        # Should complete without errors
        result = alg.get_best_solution()
        self.assertIsNotNone(result[0])

    def test_objective_returns_nan(self):
        """Test handling of NaN from objective function."""
        call_count = [0]

        def objective_with_nan(x):
            call_count[0] += 1
            if call_count[0] == 5:  # Return NaN on 5th call
                return np.nan
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            alg = NelderMead(
                objective_fn=objective_with_nan,
                lower_bounds=lower,
                upper_bounds=upper,
                max_fes=20,
                seed=42,
            )
            alg.run()

            # Should issue warning about non-finite value
            self.assertTrue(any("non-finite" in str(warning.message) for warning in w))

    def test_objective_returns_inf(self):
        """Test handling of Inf from objective function."""
        call_count = [0]

        def objective_with_inf(x):
            call_count[0] += 1
            if call_count[0] == 5:  # Return Inf on 5th call
                return np.inf
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            alg = NelderMead(
                objective_fn=objective_with_inf,
                lower_bounds=lower,
                upper_bounds=upper,
                max_fes=20,
                seed=42,
            )
            alg.run()

            # Should issue warning about non-finite value
            self.assertTrue(any("non-finite" in str(warning.message) for warning in w))

    def test_objective_raises_exception(self):
        """Test handling of exception from objective function."""
        call_count = [0]

        def objective_with_exception(x):
            call_count[0] += 1
            if call_count[0] == 5:  # Raise exception on 5th call
                raise RuntimeError("Simulated error")
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            alg = NelderMead(
                objective_fn=objective_with_exception,
                lower_bounds=lower,
                upper_bounds=upper,
                max_fes=20,
                seed=42,
            )
            alg.run()

            # Should issue warning about evaluation failure
            self.assertTrue(
                any("evaluation failed" in str(warning.message) for warning in w)
            )

    def test_single_dimension_problem(self):
        """Test optimization in 1D."""

        def objective(x):
            return (x[0] - 3.0) ** 2

        lower = np.array([0.0])
        upper = np.array([10.0])

        alg = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=42,
        )
        alg.run()

        result = alg.get_best_solution()
        # Should find solution near x=3
        self.assertAlmostEqual(result[0][0], 3.0, places=1)

    def test_high_dimension_problem(self):
        """Test optimization in high dimensions."""
        dim = 20

        def objective(x):
            return np.sum(x**2)

        lower = np.zeros(dim)
        upper = np.ones(dim)

        alg = NelderMead(
            objective_fn=objective,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            seed=42,
        )
        alg.run()

        # Should complete without errors
        result = alg.get_best_solution()
        self.assertEqual(len(result[0]), dim)


class TestRestartStrategyReproducibility(unittest.TestCase):
    """
    Test that restart strategies are reproducible with same seed.

    Validates: Requirement 18.1 (restart mechanism reproducibility)
    """

    def test_uniform_restart_reproducibility(self):
        """Test uniform restart produces same points with same seed."""
        strategy1 = create_restart_strategy("uniform", seed=42)
        strategy2 = create_restart_strategy("uniform", seed=42)

        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        points1 = [strategy1.generate_point(2, bounds) for _ in range(5)]
        points2 = [strategy2.generate_point(2, bounds) for _ in range(5)]

        for p1, p2 in zip(points1, points2):
            np.testing.assert_array_equal(p1, p2)

    def test_gaussian_restart_reproducibility(self):
        """Test Gaussian restart produces same points with same seed."""
        strategy1 = create_restart_strategy("gaussian", seed=123)
        strategy2 = create_restart_strategy("gaussian", seed=123)

        bounds = (np.array([-5.0, -5.0]), np.array([5.0, 5.0]))

        points1 = [strategy1.generate_point(2, bounds) for _ in range(5)]
        points2 = [strategy2.generate_point(2, bounds) for _ in range(5)]

        for p1, p2 in zip(points1, points2):
            np.testing.assert_array_equal(p1, p2)

    def test_gaussian_best_restart_reproducibility(self):
        """Test Gaussian-best restart produces same points with same seed."""
        strategy1 = create_restart_strategy("gaussian_best", seed=999)
        strategy2 = create_restart_strategy("gaussian_best", seed=999)

        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        best = np.array([5.0, 5.0])

        points1 = [strategy1.generate_point(2, bounds, best) for _ in range(5)]
        points2 = [strategy2.generate_point(2, bounds, best) for _ in range(5)]

        for p1, p2 in zip(points1, points2):
            np.testing.assert_array_equal(p1, p2)


if __name__ == "__main__":
    unittest.main()
