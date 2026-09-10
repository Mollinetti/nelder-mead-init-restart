#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Barrier Abstract Class for Constraint Handling

This module defines the abstract base class for barrier methods used in
constrained optimization. Barrier methods transform constrained optimization
problems into unconstrained ones by adding penalty terms to the objective
function when constraints are violated.

References:
    - Deb, K. (2000). An efficient constraint handling method for genetic algorithms.
      Computer Methods in Applied Mechanics and Engineering, 186(2-4), 311-338.
    - Audet, C., & Hare, W. (2017). Derivative-free and blackbox optimization.
      Springer International Publishing.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class Barrier(ABC):
    """
    Abstract base class for barrier methods in constrained optimization.

    Barrier methods handle constraints by adding penalty terms to the objective
    function. Different barrier methods implement different penalty strategies,
    but all share common functionality for computing bound violations.

    Attributes:
        bounds (Tuple[np.ndarray, np.ndarray]): Lower and upper bounds for decision variables
        num_solutions (int): Number of solutions in the population (used for tracking iterations)
        penalty_coeff (float): Base penalty coefficient for constraint violations
        t (int): Counter tracking the number of penalty function evaluations
    """

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        num_solutions: int,
        penalty_coeff: float = 1e4,
    ):
        """
        Initialize the barrier method.

        Args:
            bounds: Tuple of (lower_bounds, upper_bounds) as numpy arrays
            num_solutions: Number of solutions in the population
            penalty_coeff: Penalty coefficient for constraint violations (default: 1e4)

        Raises:
            ValueError: If bounds are inconsistent or invalid
            TypeError: If inputs have incorrect types
        """
        # Validate inputs
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise TypeError("bounds must be a tuple of (lower_bounds, upper_bounds)")

        lower_bounds, upper_bounds = bounds

        if not isinstance(lower_bounds, np.ndarray) or not isinstance(
            upper_bounds, np.ndarray
        ):
            raise TypeError("bounds must contain numpy arrays")

        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError("Lower and upper bounds must have the same shape")

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError(
                "Lower bounds must be strictly less than upper bounds in all dimensions"
            )

        if not isinstance(num_solutions, int) or num_solutions <= 0:
            raise ValueError("num_solutions must be a positive integer")

        if penalty_coeff <= 0:
            raise ValueError("penalty_coeff must be positive")

        self.bounds = bounds
        self.num_solutions = num_solutions
        self.penalty_coeff = penalty_coeff
        self.t = 0  # Evaluation counter

    @abstractmethod
    def penalize_constraint_violation(
        self,
        solution: np.ndarray,
        eq_violations: np.ndarray,
        ineq_violations: np.ndarray,
    ) -> float:
        """
        Compute penalty for constraint violations.

        This is the core method that each barrier method must implement.
        It defines how constraint violations are penalized.

        Args:
            solution: Decision variable vector
            eq_violations: Array of equality constraint violations |h(x)|
            ineq_violations: Array of inequality constraint violations max(0, g(x))

        Returns:
            Penalty value to be added to the objective function

        Note:
            Implementations should increment self.t to track evaluations.
        """
        pass

    def penalize_bounds(self, solution: np.ndarray) -> float:
        """
        Compute penalty for bound violations.

        This method computes the total violation of box constraints (bounds).
        It is shared by all barrier methods and implements the formula from
        Requirements 12.1 and 12.2:

        bound_violation = sum(max(0, lower - x)) + sum(max(0, x - upper))

        The formula computes:
        1. Lower bound violations: For each dimension i, if xᵢ < lowerᵢ, add (lowerᵢ - xᵢ)
        2. Upper bound violations: For each dimension i, if xᵢ > upperᵢ, add (xᵢ - upperᵢ)

        The max(0, ·) operation ensures we only count actual violations (negative
        deviations for lower bounds, positive deviations for upper bounds).

        Reference: Deb (2000), Section 3.1 - constraint violation computation

        Args:
            solution: Decision variable vector

        Returns:
            Sum of bound violations (0 if all bounds are satisfied)

        Validates: Requirements 12.1, 12.2
        """
        lower_bounds, upper_bounds = self.bounds

        # Compute lower bound violations: sum of negative deviations
        # For each dimension: max(0, lowerᵢ - xᵢ)
        lower_violations = np.maximum(lower_bounds - solution, 0)

        # Compute upper bound violations: sum of positive deviations
        # For each dimension: max(0, xᵢ - upperᵢ)
        upper_violations = np.maximum(solution - upper_bounds, 0)

        # Total bound violation: sum across all dimensions
        total_violation = np.sum(lower_violations) + np.sum(upper_violations)

        return total_violation

    def update(self, *args, **kwargs) -> None:
        """
        Update barrier parameters (optional, for adaptive methods).

        Some barrier methods (e.g., Augmented Lagrangian, Progressive Barrier)
        need to update their internal parameters based on optimization progress.
        The base implementation does nothing; subclasses override as needed.

        Args:
            *args: Variable positional arguments (method-specific)
            **kwargs: Variable keyword arguments (method-specific)
        """
        pass

    def enforce_bounds(self, solution: np.ndarray) -> np.ndarray:
        """
        Project a solution back to the feasible bounds.

        This method clips each dimension of the solution to lie within
        [lower_bound, upper_bound]. This is used to enforce box constraints
        after simplex operations that may produce out-of-bounds points.

        Args:
            solution: Decision variable vector (may be out of bounds)

        Returns:
            Solution projected to the nearest point within bounds

        Validates: Requirements 16.1, 16.4
        """
        lower_bounds, upper_bounds = self.bounds
        return np.clip(solution, lower_bounds, upper_bounds)

    def compute_total_violation(
        self,
        solution: np.ndarray,
        eq_violations: np.ndarray,
        ineq_violations: np.ndarray,
    ) -> float:
        """
        Compute total constraint violation for a solution.

        This method computes the sum of all constraint violations:
        - Bound violations (from box constraints)
        - Equality constraint violations
        - Inequality constraint violations

        A feasible solution has total violation = 0.

        Args:
            solution: Decision variable vector
            eq_violations: Array of equality constraint violations |h(x)|
            ineq_violations: Array of inequality constraint violations max(0, g(x))

        Returns:
            Total constraint violation (0 if feasible)

        Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5
        """
        # Bound violations
        bound_vio = self.penalize_bounds(solution)

        # Equality constraint violations (already absolute values)
        eq_vio = np.sum(eq_violations) if len(eq_violations) > 0 else 0.0

        # Inequality constraint violations (already positive parts)
        ineq_vio = np.sum(ineq_violations) if len(ineq_violations) > 0 else 0.0

        # Total violation
        total = bound_vio + eq_vio + ineq_vio

        return total
