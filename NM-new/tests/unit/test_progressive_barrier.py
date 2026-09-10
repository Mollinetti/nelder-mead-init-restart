#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the ProgressiveBarrier constraint handling method.

Tests cover:
- Constructor and initialization
- Penalty computation with threshold management
- Best feasible and best infeasible solution tracking
- Threshold updates based on optimization progress
- Zero penalty for feasible solutions
- Large penalty for violations beyond threshold
- Actual violation penalty for violations within threshold
- Evaluation counter tracking

Validates: Requirements 7.4, 9.1, 9.2, 9.3, 9.4, 9.5
"""

import pytest
import numpy as np
from nelder_mead.constraints.progressive_barrier import ProgressiveBarrier


class TestProgressiveBarrierConstructor:
    """Test cases for ProgressiveBarrier constructor."""

    def test_valid_initialization(self):
        """Test that valid inputs create a ProgressiveBarrier instance successfully."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)

        barrier = ProgressiveBarrier(bounds, penalty_coeff=1e4)

        assert barrier.bounds == bounds
        assert barrier.penalty_coeff == 1e4
        assert barrier.t == 0
        assert barrier.threshold == float("inf")
        assert barrier.best_feasible is None
        assert barrier.best_infeasible is None

    def test_custom_rho_function(self):
        """Test that custom rho function can be provided."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        # Use L1 norm instead of L2
        barrier = ProgressiveBarrier(bounds, rho_fn=lambda x: np.linalg.norm(x, ord=1))

        assert barrier.rho_fn is not None

    def test_default_rho_function_is_l2_norm(self):
        """Test that default rho function is L2 norm."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        barrier = ProgressiveBarrier(bounds)

        # Test that default rho_fn behaves like L2 norm
        test_array = np.array([3.0, 4.0])
        assert barrier.rho_fn(test_array) == 5.0  # sqrt(9 + 16) = 5

    def test_inherits_from_barrier(self):
        """Test that ProgressiveBarrier inherits from Barrier base class."""
        from nelder_mead.constraints.barrier_base import Barrier

        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        assert isinstance(barrier, Barrier)


class TestProgressiveBarrierFeasibleSolutions:
    """Test cases for feasible solutions (zero penalty)."""

    def test_feasible_solution_returns_zero_penalty(self):
        """Test that fully feasible solutions return zero penalty."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds, penalty_coeff=1e4)

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
        barrier = ProgressiveBarrier(bounds)

        # Solution exactly on bounds
        solution = np.array([0.0, 10.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0


class TestProgressiveBarrierThresholdPenalty:
    """Test cases for penalty computation based on threshold (Requirements 9.4, 9.5)."""

    def test_violation_within_threshold_uses_actual_violation(self):
        """Test that violations within threshold use actual violation value (Requirement 9.5)."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds, penalty_coeff=1e4)

        # Set threshold to 5.0
        barrier.threshold = 5.0

        # Solution with violation = 2.0 (within threshold)
        solution = np.array([-1.0, 5.0])  # Violates lower bound by 1.0
        eq_violations = np.array([1.0])  # Equality violation = 1.0
        ineq_violations = np.array([])
        # Total violation = 1.0 (bound) + 1.0 (eq) = 2.0

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: actual violation = 2.0 (since 2.0 <= 5.0)
        assert penalty == 2.0

    def test_violation_beyond_threshold_uses_large_penalty(self):
        """Test that violations beyond threshold use large penalty (Requirement 9.4)."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds, penalty_coeff=1e4)

        # Set threshold to 2.0
        barrier.threshold = 2.0

        # Solution with violation = 5.0 (beyond threshold)
        solution = np.array([-2.0, 5.0])  # Violates lower bound by 2.0
        eq_violations = np.array([3.0])  # Equality violation = 3.0
        ineq_violations = np.array([])
        # Total violation = 2.0 (bound) + 3.0 (eq) = 5.0

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: violation * penalty_coeff = 5.0 * 1e4 = 50000.0 (since 5.0 > 2.0)
        assert penalty == 50000.0

    def test_violation_exactly_at_threshold_uses_actual_violation(self):
        """Test that violations exactly at threshold use actual violation."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds, penalty_coeff=1e4)

        # Set threshold to 3.0
        barrier.threshold = 3.0

        # Solution with violation = 3.0 (exactly at threshold)
        solution = np.array([-3.0])  # Violates lower bound by 3.0
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: actual violation = 3.0 (since 3.0 <= 3.0)
        assert penalty == 3.0

    def test_initial_infinite_threshold_accepts_all_violations(self):
        """Test that initial infinite threshold treats all violations as within threshold."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds, penalty_coeff=1e4)

        # Initial threshold is infinity
        assert barrier.threshold == float("inf")

        # Large violation should still use actual violation (not large penalty)
        solution = np.array([-100.0])  # Violates lower bound by 100.0
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: actual violation = 100.0 (since 100.0 < inf)
        assert penalty == 100.0


class TestProgressiveBarrierBestFeasibleTracking:
    """Test cases for best feasible solution tracking (Requirement 9.1)."""

    def test_update_tracks_best_feasible_solution(self):
        """Test that update() tracks the best feasible solution (Requirement 9.1)."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # Create feasible solutions
        solutions = np.array(
            [
                [5.0, 5.0],  # Feasible, obj = 10.0
                [3.0, 3.0],  # Feasible, obj = 5.0 (best)
                [7.0, 7.0],  # Feasible, obj = 15.0
            ]
        )
        obj_vals = np.array([10.0, 5.0, 15.0])
        eq_violations_list = [np.array([]), np.array([]), np.array([])]
        ineq_violations_list = [np.array([]), np.array([]), np.array([])]

        barrier.update(solutions, obj_vals, eq_violations_list, ineq_violations_list)

        # Best feasible should be solution with obj = 5.0
        assert barrier.best_feasible is not None
        assert np.allclose(barrier.best_feasible[0], np.array([3.0, 3.0]))
        assert barrier.best_feasible[1] == 5.0
        assert barrier.best_feasible[2] == 0.0  # Zero violation

    def test_update_replaces_best_feasible_when_better_found(self):
        """Test that better feasible solutions replace previous best."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # First iteration: feasible solution with obj = 10.0
        solutions1 = np.array([[5.0]])
        obj_vals1 = np.array([10.0])
        barrier.update(solutions1, obj_vals1, [np.array([])], [np.array([])])

        assert barrier.best_feasible[1] == 10.0

        # Second iteration: better feasible solution with obj = 5.0
        solutions2 = np.array([[3.0]])
        obj_vals2 = np.array([5.0])
        barrier.update(solutions2, obj_vals2, [np.array([])], [np.array([])])

        # Best feasible should be updated
        assert barrier.best_feasible[1] == 5.0
        assert np.allclose(barrier.best_feasible[0], np.array([3.0]))

    def test_update_keeps_best_feasible_when_worse_found(self):
        """Test that worse feasible solutions don't replace previous best."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # First iteration: feasible solution with obj = 5.0
        solutions1 = np.array([[3.0]])
        obj_vals1 = np.array([5.0])
        barrier.update(solutions1, obj_vals1, [np.array([])], [np.array([])])

        assert barrier.best_feasible[1] == 5.0

        # Second iteration: worse feasible solution with obj = 10.0
        solutions2 = np.array([[7.0]])
        obj_vals2 = np.array([10.0])
        barrier.update(solutions2, obj_vals2, [np.array([])], [np.array([])])

        # Best feasible should remain unchanged
        assert barrier.best_feasible[1] == 5.0
        assert np.allclose(barrier.best_feasible[0], np.array([3.0]))


class TestProgressiveBarrierBestInfeasibleTracking:
    """Test cases for best infeasible solution tracking (Requirement 9.2)."""

    def test_update_tracks_best_infeasible_within_threshold(self):
        """Test that update() tracks best infeasible solution within threshold (Requirement 9.2)."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # Initial threshold is infinity, so all solutions are within threshold
        # Create infeasible solutions
        solutions = np.array(
            [
                [-1.0, 5.0],  # Violation = 1.0, obj = 10.0
                [-2.0, 5.0],  # Violation = 2.0, obj = 5.0 (best obj)
                [-3.0, 5.0],  # Violation = 3.0, obj = 15.0
            ]
        )
        obj_vals = np.array([10.0, 5.0, 15.0])
        eq_violations_list = [np.array([]), np.array([]), np.array([])]
        ineq_violations_list = [np.array([]), np.array([]), np.array([])]

        barrier.update(solutions, obj_vals, eq_violations_list, ineq_violations_list)

        # Best infeasible should be solution with best objective (5.0)
        assert barrier.best_infeasible is not None
        assert np.allclose(barrier.best_infeasible[0], np.array([-2.0, 5.0]))
        assert barrier.best_infeasible[1] == 5.0
        assert barrier.best_infeasible[2] == 2.0  # Violation

    def test_threshold_updated_to_best_infeasible_violation(self):
        """Test that threshold is updated to best infeasible violation (Requirement 9.2)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # Initial threshold is infinity
        assert barrier.threshold == float("inf")

        # Update with infeasible solution (violation = 2.0)
        solutions = np.array([[-2.0]])
        obj_vals = np.array([5.0])
        barrier.update(solutions, obj_vals, [np.array([])], [np.array([])])

        # Threshold should be updated to 2.0
        assert barrier.threshold == 2.0
        assert barrier.best_infeasible[2] == 2.0

    def test_infeasible_beyond_threshold_not_tracked(self):
        """Test that infeasible solutions beyond threshold are not tracked."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # Set threshold to 2.0
        barrier.threshold = 2.0

        # Update with infeasible solution beyond threshold (violation = 5.0)
        solutions = np.array([[-5.0]])
        obj_vals = np.array([1.0])  # Good objective but too infeasible
        barrier.update(solutions, obj_vals, [np.array([])], [np.array([])])

        # Best infeasible should remain None (solution beyond threshold)
        assert barrier.best_infeasible is None


class TestProgressiveBarrierThresholdManagement:
    """Test cases for threshold management (Requirement 9.3)."""

    def test_threshold_updated_to_max_dominating_violation_when_no_better_found(self):
        """Test threshold update when no better solution found (Requirement 9.3)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # First iteration: establish best infeasible with violation = 5.0
        solutions1 = np.array([[-5.0]])
        obj_vals1 = np.array([10.0])
        barrier.update(solutions1, obj_vals1, [np.array([])], [np.array([])])

        assert barrier.threshold == 5.0
        assert barrier.best_infeasible[1] == 10.0

        # Second iteration: no better solution (worse objective)
        # Solutions within threshold but with worse objectives
        solutions2 = np.array(
            [
                [-2.0],  # Violation = 2.0, obj = 15.0 (worse than 10.0)
                [-3.0],  # Violation = 3.0, obj = 20.0 (worse than 10.0)
            ]
        )
        obj_vals2 = np.array([15.0, 20.0])
        barrier.update(
            solutions2,
            obj_vals2,
            [np.array([]), np.array([])],
            [np.array([]), np.array([])],
        )

        # Threshold should be updated to max violation among dominating solutions
        # Max violation among solutions within threshold = 3.0
        assert barrier.threshold == 3.0

    def test_threshold_not_increased_when_no_better_found(self):
        """Test that threshold only decreases, never increases."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # Set threshold to 2.0
        barrier.threshold = 2.0

        # Update with solutions that have larger violations but within threshold
        # But no better objective
        solutions = np.array([[-1.0]])  # Violation = 1.0
        obj_vals = np.array([100.0])  # Worse objective
        barrier.update(solutions, obj_vals, [np.array([])], [np.array([])])

        # Threshold should decrease to 1.0 (max dominating violation)
        assert barrier.threshold == 1.0


class TestProgressiveBarrierIterationType:
    """Test cases for iteration type return value."""

    def test_returns_dominating_when_better_feasible_found(self):
        """Test that update() returns 'dominating' when better feasible found."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        solutions = np.array([[5.0]])
        obj_vals = np.array([10.0])

        result = barrier.update(solutions, obj_vals, [np.array([])], [np.array([])])

        assert result == "dominating"

    def test_returns_dominating_when_better_infeasible_found(self):
        """Test that update() returns 'dominating' when better infeasible found."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        solutions = np.array([[-1.0]])
        obj_vals = np.array([5.0])

        result = barrier.update(solutions, obj_vals, [np.array([])], [np.array([])])

        assert result == "dominating"

    def test_returns_non_dominating_when_no_better_found(self):
        """Test that update() returns 'non-dominating' when no better solution found."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # First iteration: establish best
        solutions1 = np.array([[5.0]])
        obj_vals1 = np.array([10.0])
        barrier.update(solutions1, obj_vals1, [np.array([])], [np.array([])])

        # Second iteration: worse solution
        solutions2 = np.array([[7.0]])
        obj_vals2 = np.array([20.0])
        result = barrier.update(solutions2, obj_vals2, [np.array([])], [np.array([])])

        assert result == "non-dominating"


class TestProgressiveBarrierGetBestSolution:
    """Test cases for get_best_solution() method."""

    def test_returns_best_feasible_when_available(self):
        """Test that get_best_solution() returns best feasible when available."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # Add both feasible and infeasible solutions
        solutions = np.array([[5.0], [-1.0]])
        obj_vals = np.array([10.0, 5.0])  # Infeasible has better obj
        barrier.update(
            solutions,
            obj_vals,
            [np.array([]), np.array([])],
            [np.array([]), np.array([])],
        )

        best = barrier.get_best_solution()

        # Should return feasible solution even though infeasible has better objective
        assert best is not None
        assert np.allclose(best[0], np.array([5.0]))
        assert best[1] == 10.0
        assert best[2] == 0.0

    def test_returns_best_infeasible_when_no_feasible(self):
        """Test that get_best_solution() returns best infeasible when no feasible available."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # Add only infeasible solutions
        solutions = np.array([[-1.0], [-2.0]])
        obj_vals = np.array([10.0, 5.0])
        barrier.update(
            solutions,
            obj_vals,
            [np.array([]), np.array([])],
            [np.array([]), np.array([])],
        )

        best = barrier.get_best_solution()

        # Should return best infeasible
        assert best is not None
        assert np.allclose(best[0], np.array([-2.0]))
        assert best[1] == 5.0
        assert best[2] == 2.0

    def test_returns_none_when_no_solutions(self):
        """Test that get_best_solution() returns None when no solutions evaluated."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        best = barrier.get_best_solution()

        assert best is None


class TestProgressiveBarrierReset:
    """Test cases for reset() method."""

    def test_reset_clears_all_state(self):
        """Test that reset() clears all tracked state."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        # Add some solutions
        solutions = np.array([[5.0], [-1.0]])
        obj_vals = np.array([10.0, 5.0])
        barrier.update(
            solutions,
            obj_vals,
            [np.array([]), np.array([])],
            [np.array([]), np.array([])],
        )

        # Verify state is populated
        assert barrier.best_feasible is not None
        assert barrier.best_infeasible is not None
        assert barrier.threshold != float("inf")

        # Call penalize to increment counter
        barrier.penalize_constraint_violation(solutions[0], np.array([]), np.array([]))
        assert barrier.t > 0

        # Reset
        barrier.reset()

        # Verify state is cleared
        assert barrier.best_feasible is None
        assert barrier.best_infeasible is None
        assert barrier.threshold == float("inf")
        assert barrier.t == 0


class TestProgressiveBarrierEvaluationCounter:
    """Test cases for evaluation counter tracking."""

    def test_counter_increments_on_each_call(self):
        """Test that evaluation counter increments with each penalty computation."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

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


class TestProgressiveBarrierEdgeCases:
    """Test cases for edge cases and special scenarios."""

    def test_empty_constraint_arrays(self):
        """Test that empty constraint arrays are handled correctly."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_update_without_arguments_does_nothing(self):
        """Test that update() without arguments does nothing."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        initial_threshold = barrier.threshold

        result = barrier.update()

        assert result == "non-dominating"
        assert barrier.threshold == initial_threshold
        assert barrier.best_feasible is None
        assert barrier.best_infeasible is None

    def test_high_dimensional_problem(self):
        """Test Progressive Barrier on high-dimensional problem."""
        dim = 100
        lower = np.zeros(dim)
        upper = np.ones(dim)
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds, penalty_coeff=1e4)

        # Feasible solution
        solution = np.full(dim, 0.5)
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert penalty == 0.0

        # Infeasible solution within threshold (one dimension violates by 0.1)
        solution[50] = 1.1
        barrier.threshold = 1.0  # Set threshold to accept this violation
        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert np.isclose(penalty, 0.1)  # Actual violation

        # Same solution but beyond threshold
        barrier.threshold = 0.05
        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert np.isclose(penalty, 0.1 * 1e4)  # Large penalty

    def test_mixed_feasible_and_infeasible_update(self):
        """Test update with mix of feasible and infeasible solutions."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ProgressiveBarrier(bounds)

        solutions = np.array(
            [
                [5.0, 5.0],  # Feasible, obj = 10.0
                [-1.0, 5.0],  # Infeasible, obj = 5.0
                [3.0, 3.0],  # Feasible, obj = 8.0 (best feasible)
                [-2.0, 5.0],  # Infeasible, obj = 7.0
            ]
        )
        obj_vals = np.array([10.0, 5.0, 8.0, 7.0])
        eq_violations_list = [np.array([]) for _ in range(4)]
        ineq_violations_list = [np.array([]) for _ in range(4)]

        barrier.update(solutions, obj_vals, eq_violations_list, ineq_violations_list)

        # Best feasible should be solution with obj = 8.0
        assert barrier.best_feasible is not None
        assert barrier.best_feasible[1] == 8.0

        # Best infeasible should be solution with obj = 5.0
        assert barrier.best_infeasible is not None
        assert barrier.best_infeasible[1] == 5.0
        assert barrier.best_infeasible[2] == 1.0  # Violation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
