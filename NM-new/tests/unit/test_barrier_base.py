#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the base Barrier abstract class.

Tests cover:
- Constructor validation
- Bound violation computation
- Total violation computation
- Bound enforcement
- Error handling
"""

import pytest
import numpy as np
from nelder_mead.constraints.barrier_base import Barrier


# Concrete implementation for testing the abstract base class
class ConcreteBarrier(Barrier):
    """Simple concrete implementation for testing purposes."""

    def penalize_constraint_violation(self, solution, eq_violations, ineq_violations):
        """Simple penalty: sum of all violations."""
        self.t += 1
        return self.penalty_coeff * (np.sum(eq_violations) + np.sum(ineq_violations))


class TestBarrierConstructor:
    """Test cases for Barrier constructor and input validation."""

    def test_valid_initialization(self):
        """Test that valid inputs create a Barrier instance successfully."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)

        barrier = ConcreteBarrier(bounds, num_solutions=10, penalty_coeff=1e4)

        assert barrier.bounds == bounds
        assert barrier.num_solutions == 10
        assert barrier.penalty_coeff == 1e4
        assert barrier.t == 0

    def test_invalid_bounds_type(self):
        """Test that non-tuple bounds raise TypeError."""
        with pytest.raises(TypeError, match="bounds must be a tuple"):
            ConcreteBarrier([0, 1], num_solutions=10)

    def test_invalid_bounds_length(self):
        """Test that bounds tuple with wrong length raises TypeError."""
        with pytest.raises(TypeError, match="bounds must be a tuple"):
            ConcreteBarrier(
                (np.array([0]), np.array([1]), np.array([2])), num_solutions=10
            )

    def test_bounds_not_numpy_arrays(self):
        """Test that bounds must be numpy arrays."""
        with pytest.raises(TypeError, match="bounds must contain numpy arrays"):
            ConcreteBarrier(([0, 0], [1, 1]), num_solutions=10)

    def test_mismatched_bounds_shape(self):
        """Test that lower and upper bounds must have same shape."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])

        with pytest.raises(ValueError, match="same shape"):
            ConcreteBarrier((lower, upper), num_solutions=10)

    def test_inconsistent_bounds(self):
        """Test that lower bounds must be less than upper bounds."""
        lower = np.array([5.0, 0.0, 0.0])
        upper = np.array([1.0, 10.0, 10.0])  # First dimension is inconsistent

        with pytest.raises(ValueError, match="strictly less than"):
            ConcreteBarrier((lower, upper), num_solutions=10)

    def test_equal_bounds(self):
        """Test that equal lower and upper bounds are rejected."""
        lower = np.array([0.0, 5.0, 0.0])
        upper = np.array([10.0, 5.0, 10.0])  # Second dimension is equal

        with pytest.raises(ValueError, match="strictly less than"):
            ConcreteBarrier((lower, upper), num_solutions=10)

    def test_invalid_num_solutions(self):
        """Test that num_solutions must be a positive integer."""
        bounds = (np.array([0.0]), np.array([1.0]))

        with pytest.raises(ValueError, match="positive integer"):
            ConcreteBarrier(bounds, num_solutions=0)

        with pytest.raises(ValueError, match="positive integer"):
            ConcreteBarrier(bounds, num_solutions=-5)

    def test_invalid_penalty_coeff(self):
        """Test that penalty_coeff must be positive."""
        bounds = (np.array([0.0]), np.array([1.0]))

        with pytest.raises(ValueError, match="penalty_coeff must be positive"):
            ConcreteBarrier(bounds, num_solutions=10, penalty_coeff=0)

        with pytest.raises(ValueError, match="penalty_coeff must be positive"):
            ConcreteBarrier(bounds, num_solutions=10, penalty_coeff=-100)


class TestPenalizeBounds:
    """Test cases for bound violation computation (Requirements 12.1, 12.2)."""

    def test_feasible_solution(self):
        """Test that feasible solutions have zero bound violation."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        # Solution within bounds
        solution = np.array([5.0, 3.0, 7.0])
        violation = barrier.penalize_bounds(solution)

        assert violation == 0.0

    def test_solution_on_lower_bound(self):
        """Test that solutions on lower bound have zero violation."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([0.0, 5.0, 10.0])
        violation = barrier.penalize_bounds(solution)

        assert violation == 0.0

    def test_solution_on_upper_bound(self):
        """Test that solutions on upper bound have zero violation."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([10.0, 5.0, 0.0])
        violation = barrier.penalize_bounds(solution)

        assert violation == 0.0

    def test_lower_bound_violation(self):
        """Test computation of lower bound violations (Requirement 12.1)."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        # Solution violates lower bound in first dimension by 2.0
        solution = np.array([-2.0, 5.0, 7.0])
        violation = barrier.penalize_bounds(solution)

        assert violation == 2.0

    def test_upper_bound_violation(self):
        """Test computation of upper bound violations (Requirement 12.2)."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        # Solution violates upper bound in second dimension by 3.0
        solution = np.array([5.0, 13.0, 7.0])
        violation = barrier.penalize_bounds(solution)

        assert violation == 3.0

    def test_multiple_bound_violations(self):
        """Test that multiple bound violations are summed correctly."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        # Violates lower bound by 1.5 in dim 0, upper bound by 2.5 in dim 1
        solution = np.array([-1.5, 12.5, 5.0])
        violation = barrier.penalize_bounds(solution)

        assert violation == pytest.approx(4.0)

    def test_all_dimensions_violate_bounds(self):
        """Test solution that violates bounds in all dimensions."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([-1.0, 11.0, -2.0])
        violation = barrier.penalize_bounds(solution)

        # Lower violations: 1.0 + 0 + 2.0 = 3.0
        # Upper violations: 0 + 1.0 + 0 = 1.0
        # Total: 4.0
        assert violation == pytest.approx(4.0)

    def test_asymmetric_bounds(self):
        """Test bound violations with asymmetric bounds."""
        lower = np.array([-5.0, -10.0, 0.0])
        upper = np.array([5.0, 20.0, 100.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([-7.0, 25.0, 50.0])
        violation = barrier.penalize_bounds(solution)

        # Lower violation in dim 0: 2.0
        # Upper violation in dim 1: 5.0
        # Total: 7.0
        assert violation == pytest.approx(7.0)


class TestEnforceBounds:
    """Test cases for bound enforcement (Requirements 16.1, 16.4)."""

    def test_enforce_bounds_feasible_solution(self):
        """Test that feasible solutions are unchanged."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 3.0, 7.0])
        enforced = barrier.enforce_bounds(solution)

        np.testing.assert_array_equal(enforced, solution)

    def test_enforce_bounds_clips_to_lower(self):
        """Test that out-of-bounds solutions are clipped to lower bound."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([-2.0, 5.0, 7.0])
        enforced = barrier.enforce_bounds(solution)

        expected = np.array([0.0, 5.0, 7.0])
        np.testing.assert_array_equal(enforced, expected)

    def test_enforce_bounds_clips_to_upper(self):
        """Test that out-of-bounds solutions are clipped to upper bound."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 13.0, 7.0])
        enforced = barrier.enforce_bounds(solution)

        expected = np.array([5.0, 10.0, 7.0])
        np.testing.assert_array_equal(enforced, expected)

    def test_enforce_bounds_clips_multiple_dimensions(self):
        """Test clipping in multiple dimensions simultaneously."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([-1.5, 12.5, 5.0])
        enforced = barrier.enforce_bounds(solution)

        expected = np.array([0.0, 10.0, 5.0])
        np.testing.assert_array_equal(enforced, expected)

    def test_enforce_bounds_preserves_original(self):
        """Test that enforce_bounds doesn't modify the original array."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([-1.0, 11.0])
        original_copy = solution.copy()
        enforced = barrier.enforce_bounds(solution)

        # Original should be unchanged
        np.testing.assert_array_equal(solution, original_copy)
        # Enforced should be clipped
        np.testing.assert_array_equal(enforced, np.array([0.0, 10.0]))


class TestComputeTotalViolation:
    """Test cases for total violation computation (Requirements 12.1-12.5)."""

    def test_feasible_solution_zero_violation(self):
        """Test that feasible solutions have zero total violation (Requirement 12.5)."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        total = barrier.compute_total_violation(
            solution, eq_violations, ineq_violations
        )

        assert total == 0.0

    def test_bound_violations_only(self):
        """Test total violation with only bound violations."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([-2.0, 12.0])  # Violates both bounds
        eq_violations = np.array([])
        ineq_violations = np.array([])

        total = barrier.compute_total_violation(
            solution, eq_violations, ineq_violations
        )

        # Lower violation: 2.0, Upper violation: 2.0
        assert total == pytest.approx(4.0)

    def test_equality_constraint_violations(self):
        """Test total violation with equality constraint violations (Requirement 12.3)."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 5.0])  # Within bounds
        eq_violations = np.array([1.5, 2.5])  # Already absolute values
        ineq_violations = np.array([])

        total = barrier.compute_total_violation(
            solution, eq_violations, ineq_violations
        )

        # Bound: 0, Equality: 1.5 + 2.5 = 4.0
        assert total == pytest.approx(4.0)

    def test_inequality_constraint_violations(self):
        """Test total violation with inequality constraint violations (Requirement 12.4)."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([5.0, 5.0])  # Within bounds
        eq_violations = np.array([])
        ineq_violations = np.array([0.5, 1.5, 0.0])  # Already positive parts

        total = barrier.compute_total_violation(
            solution, eq_violations, ineq_violations
        )

        # Bound: 0, Inequality: 0.5 + 1.5 + 0.0 = 2.0
        assert total == pytest.approx(2.0)

    def test_all_violation_types(self):
        """Test total violation with all types of violations combined."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([-1.0, 11.0])  # Violates bounds
        eq_violations = np.array([2.0, 3.0])  # Equality violations
        ineq_violations = np.array([1.0, 0.5])  # Inequality violations

        total = barrier.compute_total_violation(
            solution, eq_violations, ineq_violations
        )

        # Bound: 1.0 + 1.0 = 2.0
        # Equality: 2.0 + 3.0 = 5.0
        # Inequality: 1.0 + 0.5 = 1.5
        # Total: 8.5
        assert total == pytest.approx(8.5)

    def test_empty_constraint_arrays(self):
        """Test that empty constraint arrays are handled correctly."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        total = barrier.compute_total_violation(
            solution, eq_violations, ineq_violations
        )

        assert total == 0.0


class TestUpdateMethod:
    """Test cases for the update method."""

    def test_base_update_does_nothing(self):
        """Test that base class update method does nothing."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        # Should not raise any errors
        barrier.update()
        barrier.update(some_arg=123, another_arg="test")


class TestPenalizeConstraintViolation:
    """Test cases for the abstract penalize_constraint_violation method."""

    def test_concrete_implementation_increments_counter(self):
        """Test that calling penalize_constraint_violation increments counter."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10, penalty_coeff=100)

        assert barrier.t == 0

        solution = np.array([5.0])
        eq_vios = np.array([1.0])
        ineq_vios = np.array([2.0])

        penalty = barrier.penalize_constraint_violation(solution, eq_vios, ineq_vios)

        assert barrier.t == 1
        assert penalty == 300.0  # 100 * (1.0 + 2.0)

    def test_multiple_calls_increment_counter(self):
        """Test that multiple calls increment the counter correctly."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = ConcreteBarrier(bounds, num_solutions=10)

        solution = np.array([5.0])
        eq_vios = np.array([])
        ineq_vios = np.array([])

        for i in range(5):
            barrier.penalize_constraint_violation(solution, eq_vios, ineq_vios)
            assert barrier.t == i + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
