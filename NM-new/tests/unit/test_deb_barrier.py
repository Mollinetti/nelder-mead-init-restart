#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the DebBarrier constraint handling method.

Tests cover:
- Constructor and initialization
- Penalty computation using Deb's method (worst_fitness + violations)
- Worst fitness tracking and updates
- Zero penalty for feasible solutions
- Bound violation handling
- Equality and inequality constraint handling
- Evaluation counter tracking

Validates: Requirement 7.2
"""

import pytest
import numpy as np
from nelder_mead.constraints.deb_barrier import DebBarrier


class TestDebBarrierConstructor:
    """Test cases for DebBarrier constructor."""

    def test_valid_initialization(self):
        """Test that valid inputs create a DebBarrier instance successfully."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)

        barrier = DebBarrier(bounds, num_solutions=10, penalty_coeff=1e4)

        assert barrier.bounds == bounds
        assert barrier.num_solutions == 10
        assert barrier.penalty_coeff == 1e4
        assert barrier.t == 0
        assert barrier.worst_fitness == 0.0

    def test_penalty_coeff_ignored(self):
        """Test that penalty_coeff is accepted but not used in Deb Barrier."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        # Different penalty coefficients should not affect behavior
        barrier1 = DebBarrier(bounds, num_solutions=10, penalty_coeff=1e4)
        barrier2 = DebBarrier(bounds, num_solutions=10, penalty_coeff=1e10)

        # Both should have the same initial worst fitness
        assert barrier1.worst_fitness == barrier2.worst_fitness

    def test_inherits_from_barrier(self):
        """Test that DebBarrier inherits from Barrier base class."""
        from nelder_mead.constraints.barrier_base import Barrier

        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)

        assert isinstance(barrier, Barrier)


class TestDebBarrierWorstFitnessTracking:
    """Test cases for worst fitness tracking and updates."""

    def test_set_worst_fitness(self):
        """Test that set_worst_fitness updates the worst fitness value."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)

        assert barrier.worst_fitness == 0.0

        barrier.set_worst_fitness(100.0)
        assert barrier.worst_fitness == 100.0

        barrier.set_worst_fitness(250.5)
        assert barrier.worst_fitness == 250.5

    def test_update_with_worst_fitness(self):
        """Test that update() method can update worst fitness."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)

        barrier.update(worst_fitness=150.0)
        assert barrier.worst_fitness == 150.0

    def test_update_without_worst_fitness(self):
        """Test that update() without worst_fitness does nothing."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)

        barrier.set_worst_fitness(100.0)
        barrier.update()  # Should not change worst_fitness
        assert barrier.worst_fitness == 100.0

    def test_update_with_kwargs(self):
        """Test that update() ignores extra keyword arguments."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)

        # Should not raise an error
        barrier.update(worst_fitness=50.0, extra_arg=123, another_arg="test")
        assert barrier.worst_fitness == 50.0


class TestDebBarrierFeasibleSolutions:
    """Test cases for feasible solutions (zero penalty)."""

    def test_feasible_solution_returns_zero_penalty(self):
        """Test that fully feasible solutions return zero penalty."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        # Solution within bounds, no constraint violations
        solution = np.array([5.0, 3.0, 7.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_solution_on_bounds_returns_zero_penalty(self):
        """Test that solutions on bounds (but not violating) return zero penalty."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        # Solution exactly on bounds
        solution = np.array([0.0, 10.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_satisfied_constraints_return_zero_penalty(self):
        """Test that satisfied constraints return zero penalty."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        solution = np.array([5.0])
        eq_violations = np.array([0.0, 0.0])  # All satisfied
        ineq_violations = np.array([0.0, 0.0, 0.0])  # All satisfied

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_feasible_solution_independent_of_worst_fitness(self):
        """Test that feasible solutions return zero penalty regardless of worst_fitness."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        # Test with different worst fitness values
        for worst_fit in [0.0, 100.0, 1000.0, 1e6]:
            barrier.set_worst_fitness(worst_fit)
            penalty = barrier.penalize_constraint_violation(
                solution, eq_violations, ineq_violations
            )
            assert penalty == 0.0


class TestDebBarrierBoundViolations:
    """Test cases for bound violation penalties (Requirement 7.2)."""

    def test_lower_bound_violation_uses_deb_formula(self):
        """Test that violating lower bound uses Deb's formula: worst_fitness + violation."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        # Solution violates lower bound in first dimension by 0.5
        solution = np.array([-0.5, 5.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + bound_violation = 100.0 + 0.5 = 100.5
        assert penalty == 100.5

    def test_upper_bound_violation_uses_deb_formula(self):
        """Test that violating upper bound uses Deb's formula."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(50.0)

        # Solution violates upper bound in second dimension by 2.0
        solution = np.array([5.0, 12.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + bound_violation = 50.0 + 2.0 = 52.0
        assert penalty == 52.0

    def test_multiple_bound_violations_sum_correctly(self):
        """Test that multiple bound violations are summed correctly."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(200.0)

        # Solution violates bounds: -1.0 (lower by 1), 11.0 (upper by 1), -2.0 (lower by 2)
        solution = np.array([-1.0, 11.0, -2.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + (1.0 + 1.0 + 2.0) = 200.0 + 4.0 = 204.0
        assert penalty == 204.0

    def test_penalty_scales_with_worst_fitness(self):
        """Test that penalty scales with worst fitness value."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)

        # Same violation, different worst fitness
        solution = np.array([-1.0])  # Violates lower bound by 1.0
        eq_violations = np.array([])
        ineq_violations = np.array([])

        barrier.set_worst_fitness(100.0)
        penalty1 = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert penalty1 == 101.0

        barrier.set_worst_fitness(500.0)
        penalty2 = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert penalty2 == 501.0


class TestDebBarrierEqualityConstraints:
    """Test cases for equality constraint violation penalties."""

    def test_equality_violation_uses_deb_formula(self):
        """Test that equality constraint violations use Deb's formula."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(75.0)

        # Solution within bounds but violates equality constraint
        solution = np.array([5.0, 5.0])
        eq_violations = np.array([1.5])  # One equality constraint violated
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + eq_violation = 75.0 + 1.5 = 76.5
        assert penalty == 76.5

    def test_multiple_equality_violations_sum_correctly(self):
        """Test that multiple equality violations are summed correctly."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        solution = np.array([5.0, 5.0])
        eq_violations = np.array([1.0, 2.0, 0.5])  # Multiple violations
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + sum(eq_violations) = 100.0 + 3.5 = 103.5
        assert penalty == 103.5

    def test_tiny_equality_violation_uses_deb_formula(self):
        """Test that even tiny equality violations use Deb's formula."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(50.0)

        solution = np.array([5.0])
        eq_violations = np.array([1e-10])  # Very small violation
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + 1e-10 = 50.0 + 1e-10
        assert np.isclose(penalty, 50.0 + 1e-10)


class TestDebBarrierInequalityConstraints:
    """Test cases for inequality constraint violation penalties."""

    def test_inequality_violation_uses_deb_formula(self):
        """Test that inequality constraint violations use Deb's formula."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(80.0)

        # Solution within bounds but violates inequality constraint
        solution = np.array([5.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([2.5])  # One inequality constraint violated

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + ineq_violation = 80.0 + 2.5 = 82.5
        assert penalty == 82.5

    def test_multiple_inequality_violations_sum_correctly(self):
        """Test that multiple inequality violations are summed correctly."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(150.0)

        solution = np.array([5.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([1.0, 0.5, 3.0])  # Multiple violations

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + sum(ineq_violations) = 150.0 + 4.5 = 154.5
        assert penalty == 154.5


class TestDebBarrierMixedViolations:
    """Test cases for solutions with multiple types of violations."""

    def test_bound_and_equality_violations(self):
        """Test that bound + equality violations are summed correctly."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        solution = np.array([-1.0, 5.0])  # Violates lower bound by 1.0
        eq_violations = np.array([2.0])  # Violates equality constraint
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + bound_vio + eq_vio = 100.0 + 1.0 + 2.0 = 103.0
        assert penalty == 103.0

    def test_bound_and_inequality_violations(self):
        """Test that bound + inequality violations are summed correctly."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(50.0)

        solution = np.array([5.0, 11.0])  # Violates upper bound by 1.0
        eq_violations = np.array([])
        ineq_violations = np.array([1.5])  # Violates inequality constraint

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + bound_vio + ineq_vio = 50.0 + 1.0 + 1.5 = 52.5
        assert penalty == 52.5

    def test_equality_and_inequality_violations(self):
        """Test that equality + inequality violations are summed correctly."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(200.0)

        solution = np.array([5.0, 5.0])  # Within bounds
        eq_violations = np.array([1.0])  # Violates equality
        ineq_violations = np.array([2.0])  # Violates inequality

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + eq_vio + ineq_vio = 200.0 + 1.0 + 2.0 = 203.0
        assert penalty == 203.0

    def test_all_constraint_types_violated(self):
        """Test that violating all constraint types sums correctly."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(300.0)

        solution = np.array([-1.0, 11.0])  # Violates both bounds (1.0 + 1.0 = 2.0)
        eq_violations = np.array(
            [1.0, 2.0]
        )  # Violates equality constraints (sum = 3.0)
        ineq_violations = np.array(
            [0.5, 1.5]
        )  # Violates inequality constraints (sum = 2.0)

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: worst_fitness + bound_vio + eq_vio + ineq_vio = 300.0 + 2.0 + 3.0 + 2.0 = 307.0
        assert penalty == 307.0


class TestDebBarrierEvaluationCounter:
    """Test cases for evaluation counter tracking."""

    def test_counter_increments_on_each_call(self):
        """Test that evaluation counter increments with each penalty computation."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        assert barrier.t == 0

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        barrier.penalize_constraint_violation(solution, eq_violations, ineq_violations)
        assert barrier.t == 1

        barrier.penalize_constraint_violation(solution, eq_violations, ineq_violations)
        assert barrier.t == 2

        barrier.penalize_constraint_violation(solution, eq_violations, ineq_violations)
        assert barrier.t == 3

    def test_counter_increments_for_violations(self):
        """Test that counter increments even when violations occur."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        solution = np.array([-1.0])  # Violates bound
        eq_violations = np.array([])
        ineq_violations = np.array([])

        barrier.penalize_constraint_violation(solution, eq_violations, ineq_violations)
        assert barrier.t == 1

    def test_counter_increments_for_feasible_solutions(self):
        """Test that counter increments for feasible solutions too."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        solution = np.array([5.0])  # Feasible
        eq_violations = np.array([])
        ineq_violations = np.array([])

        barrier.penalize_constraint_violation(solution, eq_violations, ineq_violations)
        assert barrier.t == 1


class TestDebBarrierEdgeCases:
    """Test cases for edge cases and special scenarios."""

    def test_empty_constraint_arrays(self):
        """Test that empty constraint arrays are handled correctly."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_high_dimensional_problem(self):
        """Test Deb Barrier on high-dimensional problem."""
        dim = 100
        lower = np.zeros(dim)
        upper = np.ones(dim)
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(1000.0)

        # Feasible solution
        solution = np.full(dim, 0.5)
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert penalty == 0.0

        # Infeasible solution (one dimension violates by 0.1)
        solution[50] = 1.1
        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert penalty == 1000.1  # worst_fitness + 0.1

    def test_asymmetric_bounds(self):
        """Test Deb Barrier with asymmetric bounds."""
        lower = np.array([-100.0, -5.0, 0.0])
        upper = np.array([50.0, 10.0, 1000.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(500.0)

        # Feasible
        solution = np.array([0.0, 0.0, 500.0])
        penalty = barrier.penalize_constraint_violation(
            solution, np.array([]), np.array([])
        )
        assert penalty == 0.0

        # Infeasible (violates upper bound by 1.0)
        solution = np.array([51.0, 0.0, 500.0])
        penalty = barrier.penalize_constraint_violation(
            solution, np.array([]), np.array([])
        )
        assert penalty == 501.0  # worst_fitness + 1.0

    def test_zero_worst_fitness(self):
        """Test Deb Barrier with zero worst fitness."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(0.0)

        # Infeasible solution
        solution = np.array([-1.0])  # Violates by 1.0
        penalty = barrier.penalize_constraint_violation(
            solution, np.array([]), np.array([])
        )

        # Expected: 0.0 + 1.0 = 1.0
        assert penalty == 1.0

    def test_negative_worst_fitness(self):
        """Test Deb Barrier with negative worst fitness (valid for some problems)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(-50.0)

        # Infeasible solution
        solution = np.array([-2.0])  # Violates by 2.0
        penalty = barrier.penalize_constraint_violation(
            solution, np.array([]), np.array([])
        )

        # Expected: -50.0 + 2.0 = -48.0
        assert penalty == -48.0

    def test_very_large_worst_fitness(self):
        """Test Deb Barrier with very large worst fitness."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(1e10)

        # Infeasible solution
        solution = np.array([-1.0])  # Violates by 1.0
        penalty = barrier.penalize_constraint_violation(
            solution, np.array([]), np.array([])
        )

        # Expected: 1e10 + 1.0
        assert np.isclose(penalty, 1e10 + 1.0)


class TestDebBarrierComparisonWithHardBarrier:
    """Test cases comparing Deb Barrier behavior with Hard Barrier."""

    def test_deb_allows_comparison_of_infeasible_solutions(self):
        """Test that Deb Barrier allows comparison of infeasible solutions by violation magnitude."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        # Two infeasible solutions with different violations
        solution1 = np.array([-1.0])  # Violates by 1.0
        solution2 = np.array([-5.0])  # Violates by 5.0

        penalty1 = barrier.penalize_constraint_violation(
            solution1, np.array([]), np.array([])
        )
        penalty2 = barrier.penalize_constraint_violation(
            solution2, np.array([]), np.array([])
        )

        # Deb Barrier should prefer solution with smaller violation
        assert penalty1 < penalty2
        assert penalty1 == 101.0  # 100.0 + 1.0
        assert penalty2 == 105.0  # 100.0 + 5.0

    def test_deb_prefers_feasible_over_infeasible(self):
        """Test that Deb Barrier always prefers feasible solutions over infeasible ones."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = DebBarrier(bounds, num_solutions=10)
        barrier.set_worst_fitness(100.0)

        # Feasible solution
        feasible = np.array([5.0])
        penalty_feasible = barrier.penalize_constraint_violation(
            feasible, np.array([]), np.array([])
        )

        # Infeasible solution with tiny violation
        infeasible = np.array([10.001])  # Violates by 0.001
        penalty_infeasible = barrier.penalize_constraint_violation(
            infeasible, np.array([]), np.array([])
        )

        # Feasible should have zero penalty, infeasible should have worst_fitness + violation
        assert penalty_feasible == 0.0
        assert penalty_infeasible == 100.001

        # Even with tiny violation, infeasible is worse than feasible
        assert penalty_feasible < penalty_infeasible


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
