#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augmented Lagrangian Method for Constraint Handling

This module implements the Augmented Lagrangian method, which uses Lagrange
multipliers to handle constraints. Unlike simple penalty methods, the Augmented
Lagrangian adapts both penalty parameters and multipliers to guide the search
toward feasible optimal solutions.

The Augmented Lagrangian method combines:
1. Lagrange multipliers (λ for equality, μ for inequality constraints)
2. Quadratic penalty terms with parameter ρ
3. Adaptive updates of multipliers and penalty parameters

The penalty formula is:
    P(x) = (ρ/2) * ||h(x) + λ/ρ||² + (ρ/2) * ||max(0, g(x) + μ/ρ)||² + β * bound_violations

where:
- h(x) are equality constraint violations
- g(x) are inequality constraint violations
- λ, μ are Lagrange multipliers
- ρ is the penalty parameter
- β is the bound penalty parameter

References:
    - Conn, A. R., Gould, N. I., & Toint, P. L. (1991). A globally convergent
      augmented Lagrangian algorithm for optimization with general constraints
      and simple bounds. SIAM Journal on Numerical Analysis, 28(2), 545-572.
    - Birgin, E. G., & Martínez, J. M. (2014). Practical augmented Lagrangian
      methods for constrained optimization. SIAM.
"""

from typing import Tuple, Optional
import numpy as np

from nelder_mead.constraints.barrier_base import Barrier


class AugmentedLagrangian(Barrier):
    """
    Augmented Lagrangian method for constrained optimization.

    The Augmented Lagrangian method maintains Lagrange multipliers for each
    constraint and adapts them based on constraint violations. This provides
    better convergence properties than simple penalty methods.

    The penalty is computed as:
        P(x) = (ρ/2) * ||h(x) + λ/ρ||² + (ρ/2) * ||max(0, g(x) + μ/ρ)||² + β * bound_vio

    Multipliers are updated after each iteration:
        λ ← clip(λ + ρ*h(x), λ_min, λ_max)
        μ ← clip(max(0, μ + ρ*g(x)), 0, μ_max)

    The penalty parameter ρ is increased when constraint violations don't
    decrease sufficiently, and β is increased when bound violations increase.

    Attributes:
        bounds (Tuple[np.ndarray, np.ndarray]): Lower and upper bounds
        num_solutions (int): Number of solutions in the population
        penalty_coeff (float): Not used (kept for interface compatibility)
        t (int): Counter tracking penalty function evaluations
        m_eq (int): Number of equality constraints
        p_ineq (int): Number of inequality constraints
        lambda_eq (np.ndarray): Lagrange multipliers for equality constraints
        mu_ineq (np.ndarray): Lagrange multipliers for inequality constraints
        rho (float): Penalty parameter for constraint violations
        gamma (float): Multiplier for increasing rho
        tau (float): Threshold for detecting insufficient violation decrease
        beta (float): Penalty parameter for bound violations
        gamma_beta (float): Multiplier for increasing beta
        lambda_min (float): Minimum value for equality multipliers
        lambda_max (float): Maximum value for equality multipliers
        mu_max (float): Maximum value for inequality multipliers
        prev_eq_violation (float): Previous iteration's equality violation
        prev_ineq_violation (float): Previous iteration's inequality violation
        prev_bound_violation (float): Previous iteration's bound violation

    Validates: Requirements 7.3, 8.1, 8.2, 8.3, 8.4, 8.5
    """

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        num_solutions: int,
        m_eq: int,
        p_ineq: int,
        penalty_coeff: float = 1e4,
        rho: float = 2.0,
        gamma: float = 1.2,
        tau: float = 0.95,
        beta: float = 100.0,
        gamma_beta: float = 1.2,
        lambda_min: float = -1e6,
        lambda_max: float = 1e6,
        mu_max: float = 1e6,
    ):
        """
        Initialize the Augmented Lagrangian method.

        Args:
            bounds: Tuple of (lower_bounds, upper_bounds) as numpy arrays
            num_solutions: Number of solutions in the population
            m_eq: Number of equality constraints
            p_ineq: Number of inequality constraints
            penalty_coeff: Not used (kept for interface compatibility)
            rho: Initial penalty parameter for constraint violations (default: 2.0)
            gamma: Multiplier for increasing rho when violations don't decrease (default: 1.2)
            tau: Threshold for detecting insufficient violation decrease (default: 0.95)
            beta: Initial penalty parameter for bound violations (default: 100.0)
            gamma_beta: Multiplier for increasing beta when bound violations increase (default: 1.2)
            lambda_min: Minimum value for equality multipliers (default: -1e6)
            lambda_max: Maximum value for equality multipliers (default: 1e6)
            mu_max: Maximum value for inequality multipliers (default: 1e6)

        Raises:
            ValueError: If m_eq or p_ineq are negative, or if parameters are invalid
        """
        super().__init__(bounds, num_solutions, penalty_coeff)

        # Validate constraint counts
        if m_eq < 0:
            raise ValueError(
                "m_eq (number of equality constraints) must be non-negative"
            )
        if p_ineq < 0:
            raise ValueError(
                "p_ineq (number of inequality constraints) must be non-negative"
            )

        # Validate parameters
        if rho <= 0:
            raise ValueError("rho must be positive")
        if gamma <= 1.0:
            raise ValueError("gamma must be greater than 1.0")
        if tau <= 0 or tau >= 1.0:
            raise ValueError("tau must be in (0, 1)")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if gamma_beta <= 1.0:
            raise ValueError("gamma_beta must be greater than 1.0")
        if lambda_min >= lambda_max:
            raise ValueError("lambda_min must be less than lambda_max")
        if mu_max <= 0:
            raise ValueError("mu_max must be positive")

        # Store constraint dimensions
        self.m_eq = m_eq
        self.p_ineq = p_ineq

        # Initialize Lagrange multipliers to zero
        # λ for equality constraints (can be positive or negative)
        self.lambda_eq = np.zeros(m_eq) if m_eq > 0 else np.array([])

        # μ for inequality constraints (must be non-negative)
        self.mu_ineq = np.zeros(p_ineq) if p_ineq > 0 else np.array([])

        # Penalty parameters
        self.rho = rho  # Penalty parameter for constraint violations
        self.gamma = gamma  # Multiplier for increasing rho
        self.tau = tau  # Threshold for detecting insufficient decrease

        # Bound penalty parameters
        self.beta = beta  # Penalty parameter for bound violations
        self.gamma_beta = gamma_beta  # Multiplier for increasing beta

        # Multiplier bounds
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.mu_max = mu_max

        # Track previous violations for adaptation
        self.prev_eq_violation = float("inf")
        self.prev_ineq_violation = float("inf")
        self.prev_bound_violation = float("inf")

    def penalize_constraint_violation(
        self,
        solution: np.ndarray,
        eq_violations: np.ndarray,
        ineq_violations: np.ndarray,
    ) -> float:
        """
        Compute penalty for constraint violations using Augmented Lagrangian.

        The penalty is computed as:
            P(x) = (ρ/2) * ||h(x) + λ/ρ||² + (ρ/2) * ||max(0, g(x) + μ/ρ)||² + β * bound_vio

        where:
        - h(x) are equality constraint violations (eq_violations)
        - g(x) are inequality constraint violations (ineq_violations)
        - λ, μ are Lagrange multipliers
        - ρ is the penalty parameter
        - β is the bound penalty parameter

        The Augmented Lagrangian combines the benefits of penalty methods (which
        drive constraint violations to zero) and Lagrange multiplier methods (which
        provide information about constraint activity). The key insight is that by
        shifting the penalty terms by λ/ρ and μ/ρ, we can achieve convergence to
        the optimal solution without requiring ρ → ∞, which would cause numerical
        difficulties.

        The quadratic penalty (ρ/2)||·||² provides a smooth, differentiable penalty
        that increases quadratically with constraint violation. The multiplier shifts
        λ/ρ and μ/ρ adjust the penalty based on constraint activity, allowing the
        method to distinguish between active and inactive constraints.

        Reference: Conn et al. (1991), Section 2, Equation (2.1)
        Reference: Birgin & Martínez (2014), Algorithm 2.1

        Args:
            solution: Decision variable vector
            eq_violations: Array of equality constraint violations |h(x)|
            ineq_violations: Array of inequality constraint violations max(0, g(x))

        Returns:
            Penalty value to be added to the objective function

        Validates: Requirement 7.3 - "WHEN using AugmentedLagrangian, THE Barrier_Method
                   SHALL update Lagrange multipliers after each iteration"
        Validates: Requirement 7.5 - "WHEN evaluating a solution, THE System SHALL
                   combine objective function value with constraint penalty"

        Note:
            This method computes the penalty but does NOT update multipliers.
            Call update() after evaluating all solutions in an iteration to update
            multipliers and penalty parameters.
        """
        # Increment evaluation counter
        self.t += 1

        penalty = 0.0

        # Equality constraint penalty: (ρ/2) * ||h + λ/ρ||²
        # The shift by λ/ρ allows the penalty to adapt based on constraint activity
        if self.m_eq > 0 and len(eq_violations) > 0:
            # Note: eq_violations are already absolute values |h(x)|
            # For Augmented Lagrangian, we need the signed violations
            # Since we only have |h(x)|, we'll use them directly
            # (This is a simplification; ideally we'd have signed violations)

            # Compute shifted violations: h + λ/ρ
            shifted_eq = eq_violations + self.lambda_eq / self.rho

            # Compute quadratic penalty: (ρ/2) * ||shifted_eq||²
            # The factor ρ/2 comes from the standard Augmented Lagrangian formulation
            penalty += (self.rho / 2.0) * np.sum(shifted_eq**2)

        # Inequality constraint penalty: (ρ/2) * ||max(0, g + μ/ρ)||²
        # The max(0, ·) ensures we only penalize active constraints (g > -μ/ρ)
        if self.p_ineq > 0 and len(ineq_violations) > 0:
            # ineq_violations are already max(0, g(x))

            # Compute shifted violations: g + μ/ρ
            shifted_ineq = ineq_violations + self.mu_ineq / self.rho

            # Apply max(0, ·) to ensure non-negativity
            # This implements the complementarity condition: μ ≥ 0, g ≤ 0, μ*g = 0
            shifted_ineq = np.maximum(0, shifted_ineq)

            # Compute quadratic penalty: (ρ/2) * ||shifted_ineq||²
            penalty += (self.rho / 2.0) * np.sum(shifted_ineq**2)

        # Bound violation penalty: β * bound_vio
        # Bounds are handled separately with a simple linear penalty
        bound_violation = self.penalize_bounds(solution)
        penalty += self.beta * bound_violation

        return penalty

    def update(
        self,
        solution: Optional[np.ndarray] = None,
        eq_violations: Optional[np.ndarray] = None,
        ineq_violations: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Update Lagrange multipliers and penalty parameters.

        This method should be called after each iteration (after evaluating all
        solutions) to update:
        1. Equality multipliers: λ ← clip(λ + ρ*h, λ_min, λ_max)
        2. Inequality multipliers: μ ← clip(max(0, μ + ρ*g), 0, μ_max)
        3. Penalty parameter ρ if violations don't decrease sufficiently
        4. Bound penalty parameter β if bound violations increase

        The multiplier update rules come from the first-order optimality conditions
        (KKT conditions) for constrained optimization. The update λ ← λ + ρ*h is a
        gradient ascent step on the dual problem, moving the multipliers toward
        values that would make the constraints satisfied at optimality.

        The penalty parameter ρ is increased when violations don't decrease by at
        least a factor of τ (typically 0.95). This ensures that if the current ρ
        is insufficient to drive violations to zero, we increase the penalty strength.
        However, unlike simple penalty methods, we don't need ρ → ∞ because the
        multipliers adapt to compensate.

        Reference: Conn et al. (1991), Algorithm 14.4.2
        Reference: Birgin & Martínez (2014), Algorithm 2.1, Steps 3-4

        Args:
            solution: Best solution from current iteration (optional, for bound checking)
            eq_violations: Equality constraint violations for best solution
            ineq_violations: Inequality constraint violations for best solution
            **kwargs: Additional keyword arguments (ignored)

        Validates: Requirement 8.1 - "WHEN constraint violation decreases insufficiently,
                   THE AugmentedLagrangian SHALL increase the penalty parameter rho"
        Validates: Requirement 8.2 - "WHEN updating equality multipliers, THE
                   AugmentedLagrangian SHALL adjust them based on constraint violations and rho"
        Validates: Requirement 8.3 - "WHEN updating inequality multipliers, THE
                   AugmentedLagrangian SHALL ensure they remain non-negative"
        Validates: Requirement 8.4 - "WHEN bound violations increase, THE
                   AugmentedLagrangian SHALL increase the bound penalty parameter beta"
        Validates: Requirement 8.5 - "WHEN updating multipliers, THE AugmentedLagrangian
                   SHALL enforce upper and lower bounds on multiplier values"

        Note:
            If solution, eq_violations, or ineq_violations are not provided,
            the method will only update based on previously stored violations.
        """
        # If violations are provided, update multipliers
        if eq_violations is not None or ineq_violations is not None:
            # Update equality multipliers: λ ← clip(λ + ρ*h, λ_min, λ_max)
            # This is a gradient ascent step on the dual problem
            if self.m_eq > 0 and eq_violations is not None and len(eq_violations) > 0:
                # Update multipliers based on violations: λ ← λ + ρ*h
                # The step size ρ controls how aggressively we update
                self.lambda_eq = self.lambda_eq + self.rho * eq_violations

                # Enforce bounds on multipliers (Requirement 8.5)
                # This prevents multipliers from growing unbounded
                self.lambda_eq = np.clip(
                    self.lambda_eq, self.lambda_min, self.lambda_max
                )

            # Update inequality multipliers: μ ← clip(max(0, μ + ρ*g), 0, μ_max)
            # The max(0, ·) enforces the complementarity condition: μ ≥ 0
            if (
                self.p_ineq > 0
                and ineq_violations is not None
                and len(ineq_violations) > 0
            ):
                # Update multipliers based on violations: μ ← μ + ρ*g
                self.mu_ineq = self.mu_ineq + self.rho * ineq_violations

                # Ensure non-negativity (Requirement 8.3)
                # For inequality constraints g(x) ≤ 0, multipliers must be μ ≥ 0
                self.mu_ineq = np.maximum(0, self.mu_ineq)

                # Enforce upper bound on multipliers (Requirement 8.5)
                self.mu_ineq = np.minimum(self.mu_ineq, self.mu_max)

            # Compute current violation norms for adaptation
            current_eq_violation = 0.0
            current_ineq_violation = 0.0

            if self.m_eq > 0 and eq_violations is not None and len(eq_violations) > 0:
                # Use infinity norm: ||h||∞ = max|hᵢ|
                current_eq_violation = np.linalg.norm(eq_violations, ord=np.inf)

            if (
                self.p_ineq > 0
                and ineq_violations is not None
                and len(ineq_violations) > 0
            ):
                # For inequality, we care about max(0, g + μ/ρ)
                # This measures how much the shifted constraints are violated
                shifted_ineq = ineq_violations + self.mu_ineq / self.rho
                shifted_ineq = np.maximum(0, shifted_ineq)
                current_ineq_violation = np.linalg.norm(shifted_ineq, ord=np.inf)

            # Adapt rho if violations don't decrease sufficiently (Requirement 8.1)
            # Check if ||h||∞ or ||max(0, g + μ/ρ)||∞ > τ * previous_violation
            # If violations haven't decreased by at least (1-τ), increase ρ
            should_increase_rho = False

            if current_eq_violation > self.tau * self.prev_eq_violation:
                should_increase_rho = True

            if current_ineq_violation > self.tau * self.prev_ineq_violation:
                should_increase_rho = True

            if should_increase_rho:
                # Increase penalty parameter: ρ ← γ * ρ
                # Typical values: γ = 1.2 (20% increase)
                self.rho = self.gamma * self.rho

            # Update previous violations for next iteration
            self.prev_eq_violation = current_eq_violation
            self.prev_ineq_violation = current_ineq_violation

        # Adapt beta if bound violations increase (Requirement 8.4)
        if solution is not None:
            current_bound_violation = self.penalize_bounds(solution)

            # If bound violations increased, increase beta
            # This provides additional penalty for bound violations
            if current_bound_violation > self.prev_bound_violation:
                self.beta = self.gamma_beta * self.beta

            # Update previous bound violation for next iteration
            self.prev_bound_violation = current_bound_violation
