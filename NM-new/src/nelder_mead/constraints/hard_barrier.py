#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hard Barrier Method for Constraint Handling

This module implements the Hard Barrier method, which returns an infinite penalty
for any constraint violation. This is the simplest barrier method but can cause
numerical issues when solutions violate constraints.

The Hard Barrier is useful for problems where maintaining strict feasibility is
critical, but it may struggle with highly constrained problems where finding
initial feasible solutions is difficult.

References:
    - Audet, C., & Hare, W. (2017). Derivative-free and blackbox optimization.
      Springer International Publishing.
"""

from typing import Tuple
import numpy as np
import sys

from nelder_mead.constraints.barrier_base import Barrier


class HardBarrier(Barrier):
    """
    Hard Barrier method for constraint handling.

    The Hard Barrier returns an infinite penalty (maximum representable integer)
    for any solution that violates constraints. This enforces strict feasibility
    but can cause numerical difficulties in optimization.

    The penalty is applied for:
    - Bound violations (x < lower or x > upper)
    - Equality constraint violations (h(x) ≠ 0)
    - Inequality constraint violations (g(x) > 0)

    Attributes:
        bounds (Tuple[np.ndarray, np.ndarray]): Lower and upper bounds
        num_solutions (int): Number of solutions in the population
        penalty_coeff (float): Not used in Hard Barrier (kept for interface compatibility)
        t (int): Counter tracking penalty function evaluations
        infinite_penalty (float): The "infinite" penalty value (sys.maxsize)

    Validates: Requirement 7.1
    """

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        num_solutions: int,
        penalty_coeff: float = 1e4,
    ):
        """
        Initialize the Hard Barrier method.

        Args:
            bounds: Tuple of (lower_bounds, upper_bounds) as numpy arrays
            num_solutions: Number of solutions in the population
            penalty_coeff: Not used in Hard Barrier (kept for interface compatibility)

        Note:
            The penalty_coeff parameter is ignored by Hard Barrier but kept
            for consistency with the Barrier interface.
        """
        super().__init__(bounds, num_solutions, penalty_coeff)

        # Use maximum representable integer as "infinite" penalty
        # This is large enough to dominate any objective function value
        # while avoiding actual float('inf') which can cause numerical issues
        self.infinite_penalty = float(sys.maxsize)

    def penalize_constraint_violation(
        self,
        solution: np.ndarray,
        eq_violations: np.ndarray,
        ineq_violations: np.ndarray,
    ) -> float:
        """
        Compute penalty for constraint violations using Hard Barrier.

        Returns infinite penalty if ANY constraint is violated:
        - Any bound violation (checked via penalize_bounds)
        - Any equality constraint violation (|h(x)| > 0)
        - Any inequality constraint violation (g(x) > 0)

        Args:
            solution: Decision variable vector
            eq_violations: Array of equality constraint violations |h(x)|
            ineq_violations: Array of inequality constraint violations max(0, g(x))

        Returns:
            infinite_penalty if any constraint is violated, 0.0 otherwise

        Validates: Requirement 7.1 - "WHEN using HardBarrier, THE Barrier_Method
                   SHALL return infinite penalty for any bound violation"

        Note:
            This implementation checks ALL constraint types (bounds, equality,
            inequality) and returns infinite penalty for ANY violation.
        """
        # Increment evaluation counter
        self.t += 1

        # Check bound violations
        bound_violation = self.penalize_bounds(solution)
        if bound_violation > 0:
            return self.infinite_penalty

        # Check equality constraint violations
        if len(eq_violations) > 0:
            total_eq_violation = np.sum(eq_violations)
            if total_eq_violation > 0:
                return self.infinite_penalty

        # Check inequality constraint violations
        if len(ineq_violations) > 0:
            total_ineq_violation = np.sum(ineq_violations)
            if total_ineq_violation > 0:
                return self.infinite_penalty

        # No violations - return zero penalty
        return 0.0
