#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the HardBarrier constraint handling method.

Tests cover:
- Constructor and initialization
- Infinite penalty for bound violations
- Infinite penalty for equality constraint violations
- Infinite penalty for inequality constraint violations
- Zero penalty for feasible solutions
- Evaluation counter tracking

Validates: Requirement 7.1
"""

import pytest
import numpy as np
import sys
from nelder_mead.constraints.hard_barrier import HardBarrier


class TestHardBarrierConstructor:
    """Test cases for HardBarrier constructor."""

    def test_valid_initialization(self):
        """Test that valid inputs create a HardBarrier instance successfully."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)

        barrier = HardBarrier(bounds, num_solutions=10, penalty_coeff=1e4)

        assert barrier.bounds == bounds
        assert barrier.num_solutions == 10
        assert barrier.penalty_coeff == 1e4
        assert barrier.t == 0
        assert barrier.infinite_penalty == float(sys.maxsize)

    def test_penalty_coeff_ignored(self):
        """Test that penalty_coeff is accepted but not used in Hard Barrier."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        # Different penalty coefficients should not affect behavior
        barrier1 = HardBarrier(bounds, num_solutions=10, penalty_coeff=1e4)
        barrier2 = HardBarrier(bounds, num_solutions=10, penalty_coeff=1e10)

        # Both should have the same infinite penalty
        assert barrier1.infinite_penalty == barrier2.infinite_penalty

    def test_inherits_from_barrier(self):
        """Test that HardBarrier inherits from Barrier base class."""
        from nelder_mead.constraints.barrier_base import Barrier

        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        assert isinstance(barrier, Barrier)


class TestHardBarrierBoundViolations:
    """Test cases for bound violation penalties (Requirement 7.1)."""

    def test_lower_bound_violation_returns_infinite_penalty(self):
        """Test that violating lower bound returns infinite penalty."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Solution violates lower bound in first dimension
        solution = np.array([-0.1, 5.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_upper_bound_violation_returns_infinite_penalty(self):
        """Test that violating upper bound returns infinite penalty."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Solution violates upper bound in second dimension
        solution = np.array([5.0, 10.1, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_multiple_bound_violations_return_infinite_penalty(self):
        """Test that multiple bound violations still return infinite penalty."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Solution violates bounds in all dimensions
        solution = np.array([-1.0, 11.0, -2.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_tiny_bound_violation_returns_infinite_penalty(self):
        """Test that even tiny bound violations return infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Very small violation (1e-10)
        solution = np.array([5.0, 10.0 + 1e-10])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty


class TestHardBarrierEqualityConstraints:
    """Test cases for equality constraint violation penalties."""

    def test_equality_violation_returns_infinite_penalty(self):
        """Test that equality constraint violations return infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Solution within bounds but violates equality constraint
        solution = np.array([5.0, 5.0])
        eq_violations = np.array([1.5])  # One equality constraint violated
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_multiple_equality_violations_return_infinite_penalty(self):
        """Test that multiple equality violations return infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 5.0])
        eq_violations = np.array([1.0, 2.0, 0.5])  # Multiple violations
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_tiny_equality_violation_returns_infinite_penalty(self):
        """Test that even tiny equality violations return infinite penalty."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])
        eq_violations = np.array([1e-10])  # Very small violation
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty


class TestHardBarrierInequalityConstraints:
    """Test cases for inequality constraint violation penalties."""

    def test_inequality_violation_returns_infinite_penalty(self):
        """Test that inequality constraint violations return infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Solution within bounds but violates inequality constraint
        solution = np.array([5.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([2.5])  # One inequality constraint violated

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_multiple_inequality_violations_return_infinite_penalty(self):
        """Test that multiple inequality violations return infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([1.0, 0.5, 3.0])  # Multiple violations

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_tiny_inequality_violation_returns_infinite_penalty(self):
        """Test that even tiny inequality violations return infinite penalty."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([1e-10])  # Very small violation

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty


class TestHardBarrierFeasibleSolutions:
    """Test cases for feasible solutions (zero penalty)."""

    def test_feasible_solution_returns_zero_penalty(self):
        """Test that fully feasible solutions return zero penalty."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

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
        barrier = HardBarrier(bounds, num_solutions=10)

        # Solution exactly on bounds
        solution = np.array([0.0, 10.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_satisfied_equality_constraints_return_zero_penalty(self):
        """Test that satisfied equality constraints (zero violation) return zero penalty."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])
        eq_violations = np.array([0.0, 0.0])  # All satisfied
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_satisfied_inequality_constraints_return_zero_penalty(self):
        """Test that satisfied inequality constraints (zero violation) return zero penalty."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([0.0, 0.0, 0.0])  # All satisfied

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_all_constraints_satisfied_returns_zero_penalty(self):
        """Test that solutions satisfying all constraint types return zero penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 5.0])
        eq_violations = np.array([0.0])  # Satisfied
        ineq_violations = np.array([0.0, 0.0])  # Satisfied

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0


class TestHardBarrierMixedViolations:
    """Test cases for solutions with multiple types of violations."""

    def test_bound_and_equality_violations(self):
        """Test that bound + equality violations return infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([-1.0, 5.0])  # Violates lower bound
        eq_violations = np.array([2.0])  # Violates equality constraint
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_bound_and_inequality_violations(self):
        """Test that bound + inequality violations return infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 11.0])  # Violates upper bound
        eq_violations = np.array([])
        ineq_violations = np.array([1.5])  # Violates inequality constraint

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_equality_and_inequality_violations(self):
        """Test that equality + inequality violations return infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 5.0])  # Within bounds
        eq_violations = np.array([1.0])  # Violates equality
        ineq_violations = np.array([2.0])  # Violates inequality

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty

    def test_all_constraint_types_violated(self):
        """Test that violating all constraint types returns infinite penalty."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([-1.0, 11.0])  # Violates both bounds
        eq_violations = np.array([1.0, 2.0])  # Violates equality constraints
        ineq_violations = np.array([0.5, 1.5])  # Violates inequality constraints

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == barrier.infinite_penalty


class TestHardBarrierEvaluationCounter:
    """Test cases for evaluation counter tracking."""

    def test_counter_increments_on_each_call(self):
        """Test that evaluation counter increments with each penalty computation."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

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
        barrier = HardBarrier(bounds, num_solutions=10)

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
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])  # Feasible
        eq_violations = np.array([])
        ineq_violations = np.array([])

        barrier.penalize_constraint_violation(solution, eq_violations, ineq_violations)
        assert barrier.t == 1


class TestHardBarrierEdgeCases:
    """Test cases for edge cases and special scenarios."""

    def test_empty_constraint_arrays(self):
        """Test that empty constraint arrays are handled correctly."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_high_dimensional_problem(self):
        """Test Hard Barrier on high-dimensional problem."""
        dim = 100
        lower = np.zeros(dim)
        upper = np.ones(dim)
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Feasible solution
        solution = np.full(dim, 0.5)
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert penalty == 0.0

        # Infeasible solution (one dimension violates)
        solution[50] = 1.1
        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert penalty == barrier.infinite_penalty

    def test_asymmetric_bounds(self):
        """Test Hard Barrier with asymmetric bounds."""
        lower = np.array([-100.0, -5.0, 0.0])
        upper = np.array([50.0, 10.0, 1000.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Feasible
        solution = np.array([0.0, 0.0, 500.0])
        penalty = barrier.penalize_constraint_violation(
            solution, np.array([]), np.array([])
        )
        assert penalty == 0.0

        # Infeasible
        solution = np.array([51.0, 0.0, 500.0])
        penalty = barrier.penalize_constraint_violation(
            solution, np.array([]), np.array([])
        )
        assert penalty == barrier.infinite_penalty

    def test_infinite_penalty_is_large_but_finite(self):
        """Test that infinite penalty is actually a finite float (not inf)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = HardBarrier(bounds, num_solutions=10)

        # Verify it's finite (not actual infinity)
        assert np.isfinite(barrier.infinite_penalty)

        # Verify it's very large (larger than any reasonable objective function value)
        assert barrier.infinite_penalty > 1e15

        # Verify it equals sys.maxsize
        assert barrier.infinite_penalty == float(sys.maxsize)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
