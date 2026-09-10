#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for AdaptiveNelderMead algorithm class.

Tests cover:
- Dimension-dependent coefficient formulas (Requirements 6.1-6.4)
- Inheritance from NelderMead
- Optimization behavior on test problems
- Comparison with classic Nelder-Mead
"""

import numpy as np
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.constraints.hard_barrier import HardBarrier


class TestAdaptiveNelderMeadCoefficients:
    """Test dimension-dependent coefficient formulas."""

    def test_expansion_coefficient_formula(self):
        """Test that expansion coefficient follows formula: δₑ = 1 + (2/n) (Requirement 6.1)."""

        def obj_fn(x):
            return np.sum(x**2)

        # Test with different dimensions
        test_cases = [
            (2, 1.0 + 2.0 / 2.0),  # n=2: δₑ = 1 + 1 = 2.0
            (3, 1.0 + 2.0 / 3.0),  # n=3: δₑ = 1 + 0.667 = 1.667
            (5, 1.0 + 2.0 / 5.0),  # n=5: δₑ = 1 + 0.4 = 1.4
            (10, 1.0 + 2.0 / 10.0),  # n=10: δₑ = 1 + 0.2 = 1.2
        ]

        for dim, expected_delta_e in test_cases:
            lower = np.zeros(dim)
            upper = np.ones(dim)

            alg = AdaptiveNelderMead(
                objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
            )

            assert (
                abs(alg.delta_e - expected_delta_e) < 1e-10
            ), f"For dim={dim}, expected δₑ={expected_delta_e}, got {alg.delta_e}"

    def test_outside_contraction_coefficient_formula(self):
        """Test that outside contraction coefficient follows formula: δₒc = 0.75 - (1/(2n)) (Requirement 6.2)."""

        def obj_fn(x):
            return np.sum(x**2)

        # Test with different dimensions
        test_cases = [
            (2, 0.75 - 1.0 / (2.0 * 2.0)),  # n=2: δₒc = 0.75 - 0.25 = 0.5
            (3, 0.75 - 1.0 / (2.0 * 3.0)),  # n=3: δₒc = 0.75 - 0.167 = 0.583
            (5, 0.75 - 1.0 / (2.0 * 5.0)),  # n=5: δₒc = 0.75 - 0.1 = 0.65
            (10, 0.75 - 1.0 / (2.0 * 10.0)),  # n=10: δₒc = 0.75 - 0.05 = 0.7
        ]

        for dim, expected_delta_oc in test_cases:
            lower = np.zeros(dim)
            upper = np.ones(dim)

            alg = AdaptiveNelderMead(
                objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
            )

            assert (
                abs(alg.delta_oc - expected_delta_oc) < 1e-10
            ), f"For dim={dim}, expected δₒc={expected_delta_oc}, got {alg.delta_oc}"

    def test_inside_contraction_coefficient_formula(self):
        """Test that inside contraction coefficient follows formula: δᵢc = -(0.75 - (1/(2n))) (Requirement 6.3)."""

        def obj_fn(x):
            return np.sum(x**2)

        # Test with different dimensions
        test_cases = [
            (2, -(0.75 - 1.0 / (2.0 * 2.0))),  # n=2: δᵢc = -(0.75 - 0.25) = -0.5
            (3, -(0.75 - 1.0 / (2.0 * 3.0))),  # n=3: δᵢc = -(0.75 - 0.167) = -0.583
            (5, -(0.75 - 1.0 / (2.0 * 5.0))),  # n=5: δᵢc = -(0.75 - 0.1) = -0.65
            (10, -(0.75 - 1.0 / (2.0 * 10.0))),  # n=10: δᵢc = -(0.75 - 0.05) = -0.7
        ]

        for dim, expected_delta_ic in test_cases:
            lower = np.zeros(dim)
            upper = np.ones(dim)

            alg = AdaptiveNelderMead(
                objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
            )

            assert (
                abs(alg.delta_ic - expected_delta_ic) < 1e-10
            ), f"For dim={dim}, expected δᵢc={expected_delta_ic}, got {alg.delta_ic}"

    def test_shrink_coefficient_formula(self):
        """Test that shrink coefficient follows formula: γₛ = 1 - (1/n) (Requirement 6.4)."""

        def obj_fn(x):
            return np.sum(x**2)

        # Test with different dimensions
        test_cases = [
            (2, 1.0 - 1.0 / 2.0),  # n=2: γₛ = 1 - 0.5 = 0.5
            (3, 1.0 - 1.0 / 3.0),  # n=3: γₛ = 1 - 0.333 = 0.667
            (5, 1.0 - 1.0 / 5.0),  # n=5: γₛ = 1 - 0.2 = 0.8
            (10, 1.0 - 1.0 / 10.0),  # n=10: γₛ = 1 - 0.1 = 0.9
        ]

        for dim, expected_gamma_s in test_cases:
            lower = np.zeros(dim)
            upper = np.ones(dim)

            alg = AdaptiveNelderMead(
                objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
            )

            assert (
                abs(alg.gamma_s - expected_gamma_s) < 1e-10
            ), f"For dim={dim}, expected γₛ={expected_gamma_s}, got {alg.gamma_s}"

    def test_reflection_coefficient_remains_standard(self):
        """Test that reflection coefficient remains at standard value of 1.0."""

        def obj_fn(x):
            return np.sum(x**2)

        # Test with different dimensions
        for dim in [2, 3, 5, 10]:
            lower = np.zeros(dim)
            upper = np.ones(dim)

            alg = AdaptiveNelderMead(
                objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
            )

            assert (
                alg.delta_r == 1.0
            ), f"Reflection coefficient should always be 1.0, got {alg.delta_r}"

    def test_all_coefficients_together(self):
        """Test all four adaptive coefficients together for a specific dimension."""

        def obj_fn(x):
            return np.sum(x**2)

        dim = 4
        lower = np.zeros(dim)
        upper = np.ones(dim)

        alg = AdaptiveNelderMead(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # For n=4:
        # δₑ = 1 + 2/4 = 1.5
        # δₒc = 0.75 - 1/8 = 0.625
        # δᵢc = -(0.75 - 1/8) = -0.625
        # γₛ = 1 - 1/4 = 0.75

        assert abs(alg.delta_e - 1.5) < 1e-10
        assert abs(alg.delta_oc - 0.625) < 1e-10
        assert abs(alg.delta_ic - (-0.625)) < 1e-10
        assert abs(alg.gamma_s - 0.75) < 1e-10


class TestAdaptiveNelderMeadInheritance:
    """Test that AdaptiveNelderMead properly inherits from NelderMead."""

    def test_inherits_from_nelder_mead(self):
        """Test that AdaptiveNelderMead is a subclass of NelderMead."""
        assert issubclass(AdaptiveNelderMead, NelderMead)

    def test_has_all_parent_attributes(self):
        """Test that AdaptiveNelderMead has all attributes from NelderMead."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = AdaptiveNelderMead(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Check that key attributes from parent are present
        assert hasattr(alg, "objective_fn")
        assert hasattr(alg, "lower_bounds")
        assert hasattr(alg, "upper_bounds")
        assert hasattr(alg, "max_fes")
        assert hasattr(alg, "num_solutions")
        assert hasattr(alg, "dim")
        assert hasattr(alg, "fes")
        assert hasattr(alg, "restart_strategy")
        assert hasattr(alg, "stopping_criteria")

    def test_uses_parent_run_method(self):
        """Test that AdaptiveNelderMead uses the run() method from NelderMead."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = AdaptiveNelderMead(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper, max_fes=50
        )

        # Should be able to run without errors
        alg.run()

        # Should have executed optimization
        assert alg.fes > 0
        assert alg.simplex is not None

    def test_accepts_same_parameters_as_parent(self):
        """Test that AdaptiveNelderMead accepts the same parameters as NelderMead (except coefficients)."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Should accept all these parameters
        alg = AdaptiveNelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=100,
            num_solutions=None,
            init_method="spendleySimplex",
            seed=42,
            barrier=None,
            restart_strategy="gaussian_best",
            stopping_criteria=["fminsearch_fun", "fminsearch_x"],
        )

        assert alg is not None


class TestAdaptiveNelderMeadOptimization:
    """Test optimization behavior on test problems."""

    def test_sphere_function_2d(self):
        """Test optimization on 2D sphere function."""

        def sphere(x):
            return np.sum(x**2)

        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])

        alg = AdaptiveNelderMead(
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

    def test_sphere_function_5d(self):
        """Test optimization on 5D sphere function (higher dimension)."""

        def sphere(x):
            return np.sum(x**2)

        lower = np.array([-5.0] * 5)
        upper = np.array([5.0] * 5)

        alg = AdaptiveNelderMead(
            objective_fn=sphere,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=1000,
            seed=42,
        )

        alg.run()

        best_solution, best_fitness, _, _, _ = alg.get_best_solution()

        # Should find reasonable solution
        assert best_fitness < 1.0

    def test_rosenbrock_function_2d(self):
        """Test optimization on 2D Rosenbrock function."""

        def rosenbrock(x):
            return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])

        alg = AdaptiveNelderMead(
            objective_fn=rosenbrock,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=1000,
            seed=42,
        )

        alg.run()

        best_solution, best_fitness, _, _, _ = alg.get_best_solution()

        # Rosenbrock optimum is at [1, 1] with value 0
        # Adaptive NM should get reasonably close
        assert best_fitness < 1.0

    def test_fes_tracking(self):
        """Test that function evaluations are tracked correctly."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = AdaptiveNelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=42,
        )

        alg.run()

        # FES should be tracked and not exceed max_fes significantly
        assert alg.fes > 0
        assert alg.fes <= alg.max_fes + alg.num_solutions

    def test_best_solution_improves(self):
        """Test that best solution improves during optimization."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        alg = AdaptiveNelderMead(
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

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Run twice with same seed
        results = []
        for _ in range(2):
            alg = AdaptiveNelderMead(
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


class TestAdaptiveNelderMeadWithConstraints:
    """Test AdaptiveNelderMead with constraint handling."""

    def test_with_hard_barrier(self):
        """Test optimization with hard barrier constraint handling."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        barrier = HardBarrier(bounds=(lower, upper), num_solutions=3)

        alg = AdaptiveNelderMead(
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


class TestAdaptiveVsClassicComparison:
    """Compare Adaptive NM with Classic NM."""

    def test_different_coefficients(self):
        """Test that Adaptive NM has different coefficients than Classic NM."""

        def obj_fn(x):
            return np.sum(x**2)

        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        classic = NelderMead(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        adaptive = AdaptiveNelderMead(
            objective_fn=obj_fn, lower_bounds=lower, upper_bounds=upper
        )

        # Reflection should be the same
        assert classic.delta_r == adaptive.delta_r

        # Other coefficients should be different for dim=3
        assert classic.delta_e != adaptive.delta_e
        assert classic.delta_oc != adaptive.delta_oc
        assert classic.delta_ic != adaptive.delta_ic
        assert classic.gamma_s != adaptive.gamma_s

    def test_both_can_optimize_same_problem(self):
        """Test that both algorithms can optimize the same problem."""

        def sphere(x):
            return np.sum(x**2)

        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])

        classic = NelderMead(
            objective_fn=sphere,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            seed=42,
        )

        adaptive = AdaptiveNelderMead(
            objective_fn=sphere,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            seed=42,
        )

        classic.run()
        adaptive.run()

        _, classic_fitness, _, _, _ = classic.get_best_solution()
        _, adaptive_fitness, _, _, _ = adaptive.get_best_solution()

        # Both should find good solutions
        assert classic_fitness < 1.0
        assert adaptive_fitness < 1.0


class TestAdaptiveNelderMeadEdgeCases:
    """Test edge cases and special scenarios."""

    def test_dimension_one(self):
        """Test with 1D problem (edge case for formulas)."""

        def obj_fn(x):
            return x[0] ** 2

        lower = np.array([0.0])
        upper = np.array([1.0])

        alg = AdaptiveNelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=50,
            seed=42,
        )

        # For n=1:
        # δₑ = 1 + 2/1 = 3.0
        # δₒc = 0.75 - 1/2 = 0.25
        # δᵢc = -(0.75 - 1/2) = -0.25
        # γₛ = 1 - 1/1 = 0.0

        assert abs(alg.delta_e - 3.0) < 1e-10
        assert abs(alg.delta_oc - 0.25) < 1e-10
        assert abs(alg.delta_ic - (-0.25)) < 1e-10
        assert abs(alg.gamma_s - 0.0) < 1e-10

        # Should still be able to run
        alg.run()
        assert alg.fes > 0

    def test_high_dimension(self):
        """Test with higher dimensional problem."""

        def obj_fn(x):
            return np.sum(x**2)

        dim = 20
        lower = np.zeros(dim)
        upper = np.ones(dim)

        alg = AdaptiveNelderMead(
            objective_fn=obj_fn,
            lower_bounds=lower,
            upper_bounds=upper,
            max_fes=500,
            seed=42,
        )

        # For n=20:
        # δₑ = 1 + 2/20 = 1.1
        # δₒc = 0.75 - 1/40 = 0.725
        # δᵢc = -(0.75 - 1/40) = -0.725
        # γₛ = 1 - 1/20 = 0.95

        assert abs(alg.delta_e - 1.1) < 1e-10
        assert abs(alg.delta_oc - 0.725) < 1e-10
        assert abs(alg.delta_ic - (-0.725)) < 1e-10
        assert abs(alg.gamma_s - 0.95) < 1e-10

        # Should still be able to run
        alg.run()
        assert alg.fes > 0
