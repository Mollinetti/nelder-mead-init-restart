#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the AugmentedLagrangian constraint handling method.

Tests cover:
- Constructor and initialization
- Multiplier initialization (λ for equality, μ for inequality)
- Penalty computation with Augmented Lagrangian formula
- Multiplier updates (λ and μ)
- Rho adaptation when violations don't decrease
- Beta adaptation when bound violations increase
- Multiplier bounds enforcement
- Zero penalty for feasible solutions
- Evaluation counter tracking

Validates: Requirements 7.3, 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pytest
import numpy as np
from nelder_mead.constraints.augmented_lagrangian import AugmentedLagrangian


class TestAugmentedLagrangianConstructor:
    """Test cases for AugmentedLagrangian constructor."""

    def test_valid_initialization(self):
        """Test that valid inputs create an AugmentedLagrangian instance successfully."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)

        barrier = AugmentedLagrangian(
            bounds,
            num_solutions=10,
            m_eq=2,
            p_ineq=3,
            penalty_coeff=1e4,
            rho=2.0,
            gamma=1.2,
            tau=0.95,
            beta=100.0,
            gamma_beta=1.2,
        )

        assert barrier.bounds == bounds
        assert barrier.num_solutions == 10
        assert barrier.m_eq == 2
        assert barrier.p_ineq == 3
        assert barrier.rho == 2.0
        assert barrier.gamma == 1.2
        assert barrier.tau == 0.95
        assert barrier.beta == 100.0
        assert barrier.gamma_beta == 1.2
        assert barrier.t == 0

    def test_multipliers_initialized_to_zero(self):
        """Test that Lagrange multipliers are initialized to zero."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        barrier = AugmentedLagrangian(bounds, num_solutions=10, m_eq=3, p_ineq=2)

        # Equality multipliers should be zero
        assert len(barrier.lambda_eq) == 3
        assert np.all(barrier.lambda_eq == 0.0)

        # Inequality multipliers should be zero
        assert len(barrier.mu_ineq) == 2
        assert np.all(barrier.mu_ineq == 0.0)

    def test_zero_constraints_creates_empty_arrays(self):
        """Test that zero constraints create empty multiplier arrays."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        barrier = AugmentedLagrangian(bounds, num_solutions=10, m_eq=0, p_ineq=0)

        assert len(barrier.lambda_eq) == 0
        assert len(barrier.mu_ineq) == 0

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        barrier = AugmentedLagrangian(bounds, num_solutions=10, m_eq=1, p_ineq=1)

        assert barrier.rho == 2.0
        assert barrier.gamma == 1.2
        assert barrier.tau == 0.95
        assert barrier.beta == 100.0
        assert barrier.gamma_beta == 1.2
        assert barrier.lambda_min == -1e6
        assert barrier.lambda_max == 1e6
        assert barrier.mu_max == 1e6

    def test_custom_multiplier_bounds(self):
        """Test that custom multiplier bounds are set correctly."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        barrier = AugmentedLagrangian(
            bounds,
            num_solutions=10,
            m_eq=1,
            p_ineq=1,
            lambda_min=-100.0,
            lambda_max=100.0,
            mu_max=50.0,
        )

        assert barrier.lambda_min == -100.0
        assert barrier.lambda_max == 100.0
        assert barrier.mu_max == 50.0

    def test_inherits_from_barrier(self):
        """Test that AugmentedLagrangian inherits from Barrier base class."""
        from nelder_mead.constraints.barrier_base import Barrier

        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(bounds, num_solutions=10, m_eq=1, p_ineq=1)

        assert isinstance(barrier, Barrier)


class TestAugmentedLagrangianInputValidation:
    """Test cases for input validation."""

    def test_negative_m_eq_raises_error(self):
        """Test that negative m_eq raises ValueError."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        with pytest.raises(ValueError, match="m_eq.*must be non-negative"):
            AugmentedLagrangian(bounds, num_solutions=10, m_eq=-1, p_ineq=1)

    def test_negative_p_ineq_raises_error(self):
        """Test that negative p_ineq raises ValueError."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        with pytest.raises(ValueError, match="p_ineq.*must be non-negative"):
            AugmentedLagrangian(bounds, num_solutions=10, m_eq=1, p_ineq=-1)

    def test_non_positive_rho_raises_error(self):
        """Test that non-positive rho raises ValueError."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        with pytest.raises(ValueError, match="rho must be positive"):
            AugmentedLagrangian(bounds, num_solutions=10, m_eq=1, p_ineq=1, rho=0.0)

    def test_gamma_less_than_one_raises_error(self):
        """Test that gamma <= 1.0 raises ValueError."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        with pytest.raises(ValueError, match="gamma must be greater than 1.0"):
            AugmentedLagrangian(bounds, num_solutions=10, m_eq=1, p_ineq=1, gamma=1.0)

    def test_tau_out_of_range_raises_error(self):
        """Test that tau outside (0, 1) raises ValueError."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        with pytest.raises(ValueError, match="tau must be in"):
            AugmentedLagrangian(bounds, num_solutions=10, m_eq=1, p_ineq=1, tau=0.0)

        with pytest.raises(ValueError, match="tau must be in"):
            AugmentedLagrangian(bounds, num_solutions=10, m_eq=1, p_ineq=1, tau=1.0)

    def test_invalid_lambda_bounds_raises_error(self):
        """Test that lambda_min >= lambda_max raises ValueError."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)

        with pytest.raises(ValueError, match="lambda_min must be less than lambda_max"):
            AugmentedLagrangian(
                bounds,
                num_solutions=10,
                m_eq=1,
                p_ineq=1,
                lambda_min=100.0,
                lambda_max=50.0,
            )


class TestAugmentedLagrangianFeasibleSolutions:
    """Test cases for feasible solutions."""

    def test_feasible_solution_returns_zero_penalty(self):
        """Test that fully feasible solutions return zero penalty."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([10.0, 10.0, 10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(bounds, num_solutions=10, m_eq=0, p_ineq=0)

        # Solution within bounds, no constraint violations
        solution = np.array([5.0, 3.0, 7.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0

    def test_feasible_with_zero_violations(self):
        """Test that solutions with zero constraint violations return zero penalty."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(bounds, num_solutions=10, m_eq=2, p_ineq=2)

        solution = np.array([5.0])
        eq_violations = np.array([0.0, 0.0])
        ineq_violations = np.array([0.0, 0.0])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        assert penalty == 0.0


class TestAugmentedLagrangianPenaltyFormula:
    """Test cases for Augmented Lagrangian penalty formula (Requirement 7.3)."""

    def test_equality_constraint_penalty_formula(self):
        """Test penalty formula for equality constraints: (rho/2) * ||h + lambda/rho||^2."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=0, rho=2.0
        )

        # Set multiplier to zero for simplicity
        barrier.lambda_eq = np.array([0.0])

        solution = np.array([5.0])
        eq_violations = np.array([1.0])  # h(x) = 1.0
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: (rho/2) * ||h + lambda/rho||^2 = (2.0/2) * (1.0 + 0.0)^2 = 1.0
        assert penalty == 1.0

    def test_equality_penalty_with_nonzero_multiplier(self):
        """Test equality penalty with non-zero multiplier."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=0, rho=2.0
        )

        # Set multiplier to 4.0
        barrier.lambda_eq = np.array([4.0])

        solution = np.array([5.0])
        eq_violations = np.array([1.0])  # h(x) = 1.0
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: (rho/2) * ||h + lambda/rho||^2 = (2.0/2) * (1.0 + 4.0/2.0)^2 = 1.0 * 3.0^2 = 9.0
        assert penalty == 9.0

    def test_inequality_constraint_penalty_formula(self):
        """Test penalty formula for inequality constraints: (rho/2) * ||max(0, g + mu/rho)||^2."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=1, rho=2.0
        )

        # Set multiplier to zero for simplicity
        barrier.mu_ineq = np.array([0.0])

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([1.0])  # g(x) = 1.0

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: (rho/2) * ||max(0, g + mu/rho)||^2 = (2.0/2) * max(0, 1.0 + 0.0)^2 = 1.0
        assert penalty == 1.0

    def test_inequality_penalty_with_nonzero_multiplier(self):
        """Test inequality penalty with non-zero multiplier."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=1, rho=2.0
        )

        # Set multiplier to 2.0
        barrier.mu_ineq = np.array([2.0])

        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([1.0])  # g(x) = 1.0

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: (rho/2) * ||max(0, g + mu/rho)||^2 = (2.0/2) * max(0, 1.0 + 2.0/2.0)^2 = 1.0 * 2.0^2 = 4.0
        assert penalty == 4.0

    def test_bound_violation_penalty(self):
        """Test that bound violations are penalized with beta."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=0, beta=100.0
        )

        # Solution violates lower bound by 1.0
        solution = np.array([-1.0])
        eq_violations = np.array([])
        ineq_violations = np.array([])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected: beta * bound_violation = 100.0 * 1.0 = 100.0
        assert penalty == 100.0

    def test_combined_penalty(self):
        """Test combined penalty with equality, inequality, and bound violations."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=1, rho=2.0, beta=10.0
        )

        # Set multipliers to zero for simplicity
        barrier.lambda_eq = np.array([0.0])
        barrier.mu_ineq = np.array([0.0])

        # Solution violates lower bound by 1.0
        solution = np.array([-1.0])
        eq_violations = np.array([2.0])  # h(x) = 2.0
        ineq_violations = np.array([1.0])  # g(x) = 1.0

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )

        # Expected:
        # - Equality: (2.0/2) * 2.0^2 = 4.0
        # - Inequality: (2.0/2) * 1.0^2 = 1.0
        # - Bound: 10.0 * 1.0 = 10.0
        # Total: 4.0 + 1.0 + 10.0 = 15.0
        assert penalty == 15.0


class TestAugmentedLagrangianMultiplierUpdates:
    """Test cases for multiplier updates (Requirements 8.2, 8.3, 8.5)."""

    def test_equality_multiplier_update_formula(self):
        """Test equality multiplier update: lambda <- lambda + rho * h (Requirement 8.2)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=0, rho=2.0
        )

        # Initial multiplier is zero
        assert barrier.lambda_eq[0] == 0.0

        # Update with violation h = 1.0
        solution = np.array([5.0])
        eq_violations = np.array([1.0])
        ineq_violations = np.array([])

        barrier.update(solution, eq_violations, ineq_violations)

        # Expected: lambda = 0.0 + 2.0 * 1.0 = 2.0
        assert barrier.lambda_eq[0] == 2.0

    def test_equality_multiplier_multiple_updates(self):
        """Test that equality multipliers accumulate over multiple updates."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=0, rho=2.0
        )

        solution = np.array([5.0])

        # First update: h = 1.0
        barrier.update(solution, np.array([1.0]), np.array([]))
        assert barrier.lambda_eq[0] == 2.0

        # Second update: h = 0.5
        barrier.update(solution, np.array([0.5]), np.array([]))
        assert barrier.lambda_eq[0] == 3.0  # 2.0 + 2.0 * 0.5

    def test_inequality_multiplier_update_formula(self):
        """Test inequality multiplier update: mu <- max(0, mu + rho * g) (Requirement 8.3)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=1, rho=2.0
        )

        # Initial multiplier is zero
        assert barrier.mu_ineq[0] == 0.0

        # Update with violation g = 1.0
        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([1.0])

        barrier.update(solution, eq_violations, ineq_violations)

        # Expected: mu = max(0, 0.0 + 2.0 * 1.0) = 2.0
        assert barrier.mu_ineq[0] == 2.0

    def test_inequality_multiplier_remains_nonnegative(self):
        """Test that inequality multipliers remain non-negative (Requirement 8.3)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=1, rho=2.0
        )

        # Set initial multiplier to 1.0
        barrier.mu_ineq[0] = 1.0

        # Update with negative violation (would make mu negative without max(0, .))
        solution = np.array([5.0])
        eq_violations = np.array([])
        ineq_violations = np.array([-1.0])  # Negative violation

        barrier.update(solution, eq_violations, ineq_violations)

        # Expected: mu = max(0, 1.0 + 2.0 * (-1.0)) = max(0, -1.0) = 0.0
        assert barrier.mu_ineq[0] == 0.0

    def test_equality_multiplier_bounds_enforcement(self):
        """Test that equality multipliers are clipped to [lambda_min, lambda_max] (Requirement 8.5)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds,
            num_solutions=10,
            m_eq=1,
            p_ineq=0,
            rho=10.0,
            lambda_min=-5.0,
            lambda_max=5.0,
        )

        solution = np.array([5.0])

        # Update with large positive violation (would exceed lambda_max)
        barrier.update(solution, np.array([10.0]), np.array([]))

        # Expected: lambda = clip(0.0 + 10.0 * 10.0, -5.0, 5.0) = 5.0
        assert barrier.lambda_eq[0] == 5.0

        # Reset and test negative bound
        barrier.lambda_eq[0] = 0.0
        barrier.update(solution, np.array([-10.0]), np.array([]))

        # Expected: lambda = clip(0.0 + 10.0 * (-10.0), -5.0, 5.0) = -5.0
        assert barrier.lambda_eq[0] == -5.0

    def test_inequality_multiplier_upper_bound_enforcement(self):
        """Test that inequality multipliers are clipped to [0, mu_max] (Requirement 8.5)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=1, rho=10.0, mu_max=5.0
        )

        solution = np.array([5.0])

        # Update with large violation (would exceed mu_max)
        barrier.update(solution, np.array([]), np.array([10.0]))

        # Expected: mu = min(max(0, 0.0 + 10.0 * 10.0), 5.0) = 5.0
        assert barrier.mu_ineq[0] == 5.0

    def test_multiple_equality_constraints_update_independently(self):
        """Test that multiple equality constraints update their multipliers independently."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=3, p_ineq=0, rho=2.0
        )

        solution = np.array([5.0])
        eq_violations = np.array([1.0, 2.0, 0.5])

        barrier.update(solution, eq_violations, np.array([]))

        # Expected: lambda[i] = 0.0 + 2.0 * h[i]
        assert barrier.lambda_eq[0] == 2.0
        assert barrier.lambda_eq[1] == 4.0
        assert barrier.lambda_eq[2] == 1.0

    def test_multiple_inequality_constraints_update_independently(self):
        """Test that multiple inequality constraints update their multipliers independently."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=3, rho=2.0
        )

        solution = np.array([5.0])
        ineq_violations = np.array([1.0, 0.5, 2.0])

        barrier.update(solution, np.array([]), ineq_violations)

        # Expected: mu[i] = max(0, 0.0 + 2.0 * g[i])
        assert barrier.mu_ineq[0] == 2.0
        assert barrier.mu_ineq[1] == 1.0
        assert barrier.mu_ineq[2] == 4.0


class TestAugmentedLagrangianRhoAdaptation:
    """Test cases for rho adaptation (Requirement 8.1)."""

    def test_rho_increases_when_violations_dont_decrease(self):
        """Test that rho increases when violations don't decrease sufficiently (Requirement 8.1)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=0, rho=2.0, gamma=1.5, tau=0.9
        )

        solution = np.array([5.0])

        # First update with large violation
        barrier.update(solution, np.array([10.0]), np.array([]))
        initial_rho = barrier.rho

        # Second update with violation that doesn't decrease enough
        # Current violation: 10.0, previous: 10.0, threshold: 0.9 * 10.0 = 9.0
        # Since 10.0 > 9.0, rho should increase
        barrier.update(solution, np.array([10.0]), np.array([]))

        # Expected: rho = gamma * initial_rho = 1.5 * 2.0 = 3.0
        assert barrier.rho == initial_rho * 1.5

    def test_rho_increases_for_insufficient_equality_decrease(self):
        """Test that rho increases when equality violations decrease insufficiently."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=0, rho=2.0, gamma=2.0, tau=0.5
        )

        solution = np.array([5.0])

        # First update: violation = 10.0
        barrier.update(solution, np.array([10.0]), np.array([]))
        assert barrier.rho == 2.0

        # Second update: violation = 8.0 (decreased but not enough)
        # Threshold: 0.5 * 10.0 = 5.0, since 8.0 > 5.0, rho should increase
        barrier.update(solution, np.array([8.0]), np.array([]))
        assert barrier.rho == 4.0  # 2.0 * 2.0

    def test_rho_increases_for_insufficient_inequality_decrease(self):
        """Test that rho increases when inequality violations decrease insufficiently."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=1, rho=2.0, gamma=1.5, tau=0.8
        )

        solution = np.array([5.0])

        # First update: violation = 5.0
        barrier.update(solution, np.array([]), np.array([5.0]))
        assert barrier.rho == 2.0

        # Second update: violation = 4.5 (decreased but not enough)
        # Threshold: 0.8 * 5.0 = 4.0, since 4.5 > 4.0, rho should increase
        barrier.update(solution, np.array([]), np.array([4.5]))
        assert barrier.rho == 3.0  # 2.0 * 1.5

    def test_rho_does_not_increase_when_violations_decrease_sufficiently(self):
        """Test that rho doesn't increase when violations decrease sufficiently."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=0, rho=2.0, gamma=1.5, tau=0.9
        )

        solution = np.array([5.0])

        # First update: violation = 10.0
        barrier.update(solution, np.array([10.0]), np.array([]))
        assert barrier.rho == 2.0

        # Second update: violation = 5.0 (decreased sufficiently)
        # Threshold: 0.9 * 10.0 = 9.0, since 5.0 <= 9.0, rho should NOT increase
        barrier.update(solution, np.array([5.0]), np.array([]))
        assert barrier.rho == 2.0  # Unchanged


class TestAugmentedLagrangianBetaAdaptation:
    """Test cases for beta adaptation (Requirement 8.4)."""

    def test_beta_increases_when_bound_violations_increase(self):
        """Test that beta increases when bound violations increase (Requirement 8.4)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=0, beta=10.0, gamma_beta=2.0
        )

        # First update: bound violation = 1.0
        solution1 = np.array([-1.0])
        barrier.update(solution1, np.array([]), np.array([]))
        assert barrier.beta == 10.0

        # Second update: bound violation = 2.0 (increased)
        solution2 = np.array([-2.0])
        barrier.update(solution2, np.array([]), np.array([]))

        # Expected: beta = gamma_beta * 10.0 = 20.0
        assert barrier.beta == 20.0

    def test_beta_does_not_increase_when_bound_violations_decrease(self):
        """Test that beta doesn't increase when bound violations decrease."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=0, beta=10.0, gamma_beta=2.0
        )

        # First update: bound violation = 2.0
        solution1 = np.array([-2.0])
        barrier.update(solution1, np.array([]), np.array([]))
        assert barrier.beta == 10.0

        # Second update: bound violation = 1.0 (decreased)
        solution2 = np.array([-1.0])
        barrier.update(solution2, np.array([]), np.array([]))

        # Expected: beta unchanged = 10.0
        assert barrier.beta == 10.0

    def test_beta_does_not_increase_when_bound_violations_stay_same(self):
        """Test that beta doesn't increase when bound violations stay the same."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=0, beta=10.0, gamma_beta=2.0
        )

        # First update: bound violation = 1.0
        solution1 = np.array([-1.0])
        barrier.update(solution1, np.array([]), np.array([]))
        assert barrier.beta == 10.0

        # Second update: bound violation = 1.0 (same)
        solution2 = np.array([-1.0])
        barrier.update(solution2, np.array([]), np.array([]))

        # Expected: beta unchanged = 10.0
        assert barrier.beta == 10.0

    def test_beta_increases_multiple_times(self):
        """Test that beta can increase multiple times."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=0, p_ineq=0, beta=10.0, gamma_beta=1.5
        )

        # First update: violation = 1.0
        barrier.update(np.array([-1.0]), np.array([]), np.array([]))
        assert barrier.beta == 10.0

        # Second update: violation = 2.0 (increased)
        barrier.update(np.array([-2.0]), np.array([]), np.array([]))
        assert barrier.beta == 15.0  # 10.0 * 1.5

        # Third update: violation = 3.0 (increased again)
        barrier.update(np.array([-3.0]), np.array([]), np.array([]))
        assert barrier.beta == 22.5  # 15.0 * 1.5


class TestAugmentedLagrangianEvaluationCounter:
    """Test cases for evaluation counter tracking."""

    def test_counter_increments_on_each_call(self):
        """Test that evaluation counter increments with each penalty computation."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(bounds, num_solutions=10, m_eq=0, p_ineq=0)

        assert barrier.t == 0

        solution = np.array([5.0])
        barrier.penalize_constraint_violation(solution, np.array([]), np.array([]))
        assert barrier.t == 1

        barrier.penalize_constraint_violation(solution, np.array([]), np.array([]))
        assert barrier.t == 2

        barrier.penalize_constraint_violation(solution, np.array([]), np.array([]))
        assert barrier.t == 3


class TestAugmentedLagrangianEdgeCases:
    """Test cases for edge cases and special scenarios."""

    def test_empty_constraint_arrays(self):
        """Test that empty constraint arrays are handled correctly."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(bounds, num_solutions=10, m_eq=0, p_ineq=0)

        solution = np.array([5.0])
        penalty = barrier.penalize_constraint_violation(
            solution, np.array([]), np.array([])
        )

        assert penalty == 0.0

    def test_high_dimensional_problem(self):
        """Test Augmented Lagrangian on high-dimensional problem."""
        dim = 50
        lower = np.zeros(dim)
        upper = np.ones(dim)
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=5, p_ineq=10, rho=2.0
        )

        # Feasible solution
        solution = np.full(dim, 0.5)
        eq_violations = np.zeros(5)
        ineq_violations = np.zeros(10)

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, ineq_violations
        )
        assert penalty == 0.0

    def test_very_large_rho(self):
        """Test Augmented Lagrangian with very large rho."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=0, rho=1e6
        )

        solution = np.array([5.0])
        eq_violations = np.array([0.001])

        penalty = barrier.penalize_constraint_violation(
            solution, eq_violations, np.array([])
        )

        # Expected: (1e6/2) * 0.001^2 = 500 * 1e-6 = 0.5
        assert np.isclose(penalty, 0.5)

    def test_update_without_violations_provided(self):
        """Test that update() can be called without violations (does nothing)."""
        lower = np.array([0.0])
        upper = np.array([10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds, num_solutions=10, m_eq=1, p_ineq=1, rho=2.0
        )

        initial_lambda = barrier.lambda_eq.copy()
        initial_mu = barrier.mu_ineq.copy()
        initial_rho = barrier.rho

        # Call update without violations
        barrier.update()

        # Nothing should change
        assert np.array_equal(barrier.lambda_eq, initial_lambda)
        assert np.array_equal(barrier.mu_ineq, initial_mu)
        assert barrier.rho == initial_rho


class TestAugmentedLagrangianIntegration:
    """Integration tests combining multiple features."""

    def test_full_optimization_cycle(self):
        """Test a full cycle of penalty computation and parameter updates."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        bounds = (lower, upper)
        barrier = AugmentedLagrangian(
            bounds,
            num_solutions=10,
            m_eq=1,
            p_ineq=1,
            rho=2.0,
            gamma=1.5,
            tau=0.9,
            beta=10.0,
            gamma_beta=1.5,
        )

        # Iteration 1: Large violations
        solution1 = np.array([5.0, 5.0])
        eq_vio1 = np.array([2.0])
        ineq_vio1 = np.array([1.0])

        barrier.penalize_constraint_violation(solution1, eq_vio1, ineq_vio1)
        barrier.update(solution1, eq_vio1, ineq_vio1)

        # Check multipliers updated
        assert barrier.lambda_eq[0] == 4.0  # 0 + 2.0 * 2.0
        assert barrier.mu_ineq[0] == 2.0  # max(0, 0 + 2.0 * 1.0)

        # Iteration 2: Violations don't decrease enough
        solution2 = np.array([5.0, 5.0])
        eq_vio2 = np.array(
            [1.9]
        )  # Decreased but not enough (threshold: 0.9 * 2.0 = 1.8)
        ineq_vio2 = np.array([0.95])  # Decreased but not enough

        barrier.penalize_constraint_violation(solution2, eq_vio2, ineq_vio2)
        barrier.update(solution2, eq_vio2, ineq_vio2)

        # Check rho increased
        assert barrier.rho == 3.0  # 2.0 * 1.5

        # Check multipliers still updated
        assert barrier.lambda_eq[0] == 4.0 + 2.0 * 1.9  # Previous + rho * h
        assert barrier.mu_ineq[0] > 2.0  # Should have increased


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
