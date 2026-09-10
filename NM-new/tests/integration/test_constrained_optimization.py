"""
Integration tests for constrained optimization.

Tests verify that constrained benchmark problems can be used with
the Nelder-Mead algorithm and barrier methods.
"""

import numpy as np
from nelder_mead.benchmarks.constrained import G06, PressureVessel
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.constraints.deb_barrier import DebBarrier


class TestConstrainedOptimization:
    """Integration tests for constrained optimization."""

    def test_g06_with_nelder_mead(self):
        """Test G06 problem with Nelder-Mead algorithm."""
        problem = G06()

        # Create barrier method
        barrier = DebBarrier(
            bounds=problem.bounds, num_solutions=problem.dim + 1, penalty_coeff=1e6
        )

        # Create algorithm
        algorithm = NelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=500,
            seed=42,
            barrier=barrier,
        )

        # Run optimization
        algorithm.run()

        # Get results
        best_x, best_f, eq_vio, ineq_vio, total_vio = algorithm.get_best_solution()

        # Verify algorithm ran
        assert algorithm.fes > 0
        assert algorithm.fes <= 500

        # Verify we got a solution
        assert best_x is not None
        assert len(best_x) == problem.dim
        assert isinstance(best_f, (int, float, np.number))

        # Verify solution is within bounds
        lower, upper = problem.bounds
        assert np.all(best_x >= lower)
        assert np.all(best_x <= upper)

    def test_pressure_vessel_with_nelder_mead(self):
        """Test Pressure Vessel problem with Nelder-Mead algorithm."""
        problem = PressureVessel()

        # Create barrier method
        barrier = DebBarrier(
            bounds=problem.bounds, num_solutions=problem.dim + 1, penalty_coeff=1e6
        )

        # Create algorithm
        algorithm = NelderMead(
            objective_fn=problem.objective,
            constraint_eq_fn=problem.constraint_eq,
            constraint_ineq_fn=problem.constraint_ineq,
            lower_bounds=problem.bounds[0],
            upper_bounds=problem.bounds[1],
            max_fes=1000,
            seed=42,
            barrier=barrier,
        )

        # Run optimization
        algorithm.run()

        # Get results
        best_x, best_f, eq_vio, ineq_vio, total_vio = algorithm.get_best_solution()

        # Verify algorithm ran
        assert algorithm.fes > 0
        assert algorithm.fes <= 1000

        # Verify we got a solution
        assert best_x is not None
        assert len(best_x) == problem.dim
        assert isinstance(best_f, (int, float, np.number))

        # Verify solution is within bounds
        lower, upper = problem.bounds
        assert np.all(best_x >= lower)
        assert np.all(best_x <= upper)

        # Verify objective value is reasonable (positive cost)
        assert best_f > 0

    def test_problem_interface_compatibility(self):
        """Test that constrained problems are compatible with algorithm interface."""
        problem = G06()

        # Test that problem has required methods
        assert hasattr(problem, "objective")
        assert callable(problem.objective)
        assert hasattr(problem, "constraint_eq")
        assert callable(problem.constraint_eq)
        assert hasattr(problem, "constraint_ineq")
        assert callable(problem.constraint_ineq)

        # Test that methods return correct types
        x = np.array([14.0, 1.0])

        f = problem.objective(x)
        assert isinstance(f, (int, float, np.number))

        eq = problem.constraint_eq(x)
        assert isinstance(eq, np.ndarray)
        assert len(eq) == problem.num_eq_constraints

        ineq = problem.constraint_ineq(x)
        assert isinstance(ineq, np.ndarray)
        assert len(ineq) == problem.num_ineq_constraints
