"""
Integration tests for unconstrained benchmark problems with optimization algorithms.

Tests verify that the Nelder-Mead algorithm can successfully optimize
the unconstrained benchmark problems and find solutions close to the known optima.
"""

import numpy as np
from nelder_mead.benchmarks.unconstrained import (
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
    Griewank,
    Schwefel,
)
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead


class TestUnconstrainedOptimization:
    """Integration tests for unconstrained optimization."""

    def test_sphere_optimization(self):
        """Test that NM can optimize the Sphere function."""
        problem = Sphere(dim=5)

        algorithm = NelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=5000,
            seed=42,
            init_method="spendleySimplex",
        )

        algorithm.run()
        best_x, best_f, _, _, _ = algorithm.get_best_solution()

        # Sphere is easy - should get very close to optimum
        assert best_f < 1e-4, f"Expected f < 1e-4, got {best_f}"
        assert np.allclose(
            best_x, problem.optimum_location, atol=0.1
        ), "Solution should be close to origin"

    def test_rosenbrock_optimization(self):
        """Test that NM can optimize the Rosenbrock function."""
        problem = Rosenbrock(dim=5)

        algorithm = NelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=10000,
            seed=42,
            init_method="spendleySimplex",
        )

        algorithm.run()
        best_x, best_f, _, _, _ = algorithm.get_best_solution()

        # Rosenbrock is harder - allow more tolerance
        assert best_f < 10.0, f"Expected f < 10.0, got {best_f}"

    def test_rastrigin_optimization(self):
        """Test that NM can optimize the Rastrigin function."""
        problem = Rastrigin(dim=3)  # Use lower dimension for multimodal function

        algorithm = AdaptiveNelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=10000,
            seed=42,
            init_method="spendleySimplex",
        )

        algorithm.run()
        best_x, best_f, _, _, _ = algorithm.get_best_solution()

        # Rastrigin is multimodal - may not reach global optimum
        # Just verify algorithm runs and improves from random start
        assert best_f < 50.0, f"Expected f < 50.0, got {best_f}"

    def test_ackley_optimization(self):
        """Test that NM can optimize the Ackley function."""
        problem = Ackley(dim=3)  # Use lower dimension for multimodal function

        algorithm = AdaptiveNelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=10000,
            seed=42,
            init_method="spendleySimplex",
        )

        algorithm.run()
        best_x, best_f, _, _, _ = algorithm.get_best_solution()

        # Ackley is multimodal - may not reach global optimum
        assert best_f < 10.0, f"Expected f < 10.0, got {best_f}"

    def test_griewank_optimization(self):
        """Test that NM can optimize the Griewank function."""
        problem = Griewank(dim=3)  # Use lower dimension for multimodal function

        algorithm = AdaptiveNelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=10000,
            seed=42,
            init_method="spendleySimplex",
        )

        algorithm.run()
        best_x, best_f, _, _, _ = algorithm.get_best_solution()

        # Griewank is multimodal - may not reach global optimum
        assert best_f < 5.0, f"Expected f < 5.0, got {best_f}"

    def test_schwefel_optimization(self):
        """Test that NM can optimize the Schwefel function."""
        problem = Schwefel(dim=3)  # Use lower dimension for multimodal function

        algorithm = AdaptiveNelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=10000,
            seed=42,
            init_method="spendleySimplex",
        )

        algorithm.run()
        best_x, best_f, _, _, _ = algorithm.get_best_solution()

        # Schwefel is highly multimodal and deceptive
        # Just verify algorithm runs and finds a reasonable solution
        assert best_f < 1000.0, f"Expected f < 1000.0, got {best_f}"

    def test_all_problems_run_without_error(self):
        """Test that all problems can be instantiated and evaluated."""
        problems = [
            Sphere(dim=3),
            Rosenbrock(dim=3),
            Rastrigin(dim=3),
            Ackley(dim=3),
            Griewank(dim=3),
            Schwefel(dim=3),
        ]

        for problem in problems:
            # Test evaluation at random point
            x = np.random.uniform(problem.bounds[0], problem.bounds[1])
            f = problem.objective(x)
            assert np.isfinite(f), f"{problem.name} should return finite value"

            # Test evaluation at optimum
            f_opt = problem.objective(problem.optimum_location)
            assert np.isfinite(
                f_opt
            ), f"{problem.name} should return finite value at optimum"

    def test_reproducibility(self):
        """Test that optimization is reproducible with same seed."""
        problem = Sphere(dim=5)

        # Run 1
        alg1 = NelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=1000,
            seed=123,
            init_method="spendleySimplex",
        )
        alg1.run()
        best_x1, best_f1, _, _, _ = alg1.get_best_solution()

        # Run 2 with same seed
        alg2 = NelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=1000,
            seed=123,
            init_method="spendleySimplex",
        )
        alg2.run()
        best_x2, best_f2, _, _, _ = alg2.get_best_solution()

        # Results should be identical
        assert np.allclose(
            best_x1, best_x2
        ), "Solutions should be identical with same seed"
        assert np.allclose(
            best_f1, best_f2
        ), "Fitness values should be identical with same seed"
