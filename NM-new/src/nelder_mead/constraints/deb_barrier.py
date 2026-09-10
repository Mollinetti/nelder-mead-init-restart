#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deb Barrier Method for Constraint Handling

This module implements Deb's constraint handling approach, which penalizes
constraint violations by adding them to the worst fitness value in the population.
This method is particularly effective for evolutionary algorithms and population-based
optimization methods.

The Deb Barrier uses a simple but effective penalty strategy:
- For feasible solutions: return the actual objective value
- For infeasible solutions: return worst_fitness + sum_of_violations

This approach ensures that:
1. Feasible solutions are always preferred over infeasible ones
2. Among infeasible solutions, those with smaller violations are preferred
3. The penalty is adaptive based on the current population's worst fitness

References:
    - Deb, K. (2000). An efficient constraint handling method for genetic algorithms.
      Computer Methods in Applied Mechanics and Engineering, 186(2-4), 311-338.
"""

from typing import Tuple
import numpy as np

from nelder_mead.constraints.barrier_base import Barrier


class DebBarrier(Barrier):
    """
    Deb's constraint handling method for constrained optimization.

    The Deb Barrier implements a simple but effective penalty strategy that uses
    the worst fitness value in the population as a reference point. For infeasible
    solutions, the penalty is computed as:

        penalty = worst_fitness + sum(|h(x)|) + sum(max(0, g(x)))

    where:
    - worst_fitness is the worst objective value seen in the current population
    - h(x) are equality constraint violations
    - g(x) are inequality constraint violations

    This ensures that any feasible solution is preferred over any infeasible solution,
    and among infeasible solutions, those with smaller violations are preferred.

    Attributes:
        bounds (Tuple[np.ndarray, np.ndarray]): Lower and upper bounds
        num_solutions (int): Number of solutions in the population
        penalty_coeff (float): Not used in Deb Barrier (kept for interface compatibility)
        t (int): Counter tracking penalty function evaluations
        worst_fitness (float): Worst fitness value in the current population

    Validates: Requirement 7.2
    """

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        num_solutions: int,
        penalty_coeff: float = 1e4,
    ):
        """
        Initialize the Deb Barrier method.

        Args:
            bounds: Tuple of (lower_bounds, upper_bounds) as numpy arrays
            num_solutions: Number of solutions in the population
            penalty_coeff: Not used in Deb Barrier (kept for interface compatibility)

        Note:
            The penalty_coeff parameter is ignored by Deb Barrier but kept
            for consistency with the Barrier interface. The penalty is determined
            by the worst fitness value in the population.
        """
        super().__init__(bounds, num_solutions, penalty_coeff)

        # Initialize worst fitness to a very large value
        # This will be updated as solutions are evaluated
        self.worst_fitness = 0.0

    def penalize_constraint_violation(
        self,
        solution: np.ndarray,
        eq_violations: np.ndarray,
        ineq_violations: np.ndarray,
    ) -> float:
        """
        Compute penalty for constraint violations using Deb's method.

        The penalty is computed as:
            penalty = worst_fitness + bound_violations + sum(eq_violations) + sum(ineq_violations)

        where worst_fitness is the worst objective function value in the current population.

        This ensures that:
        1. Any feasible solution has a lower penalized fitness than any infeasible solution
        2. Among infeasible solutions, those with smaller violations are preferred
        3. The penalty adapts to the scale of the objective function

        Args:
            solution: Decision variable vector
            eq_violations: Array of equality constraint violations |h(x)|
            ineq_violations: Array of inequality constraint violations max(0, g(x))

        Returns:
            Penalty value to be added to the objective function

        Validates: Requirement 7.2 - "WHEN using DebBarrier, THE Barrier_Method
                   SHALL penalize constraint violations using Deb's constraint handling approach"

        Note:
            The worst_fitness value should be updated externally by calling
            set_worst_fitness() with the worst objective value in the population.
        """
        # Increment evaluation counter
        self.t += 1

        # Compute bound violations
        bound_violation = self.penalize_bounds(solution)

        # Compute total constraint violation
        eq_violation = np.sum(eq_violations) if len(eq_violations) > 0 else 0.0
        ineq_violation = np.sum(ineq_violations) if len(ineq_violations) > 0 else 0.0

        # Total violation
        total_violation = bound_violation + eq_violation + ineq_violation

        # If no violations, return zero penalty
        if total_violation == 0.0:
            return 0.0

        # Deb's penalty: worst_fitness + sum of all violations
        # This ensures feasible solutions are always preferred over infeasible ones
        penalty = self.worst_fitness + total_violation

        return penalty

    def set_worst_fitness(self, worst_fitness: float) -> None:
        """
        Update the worst fitness value in the population.

        This method should be called by the optimization algorithm to update
        the reference worst fitness value used in penalty computation.

        Args:
            worst_fitness: The worst (highest) objective function value in the
                          current population

        Note:
            For minimization problems, this should be the maximum objective value.
            The algorithm should call this method before evaluating solutions in
            each iteration to ensure the penalty is based on current population state.
        """
        self.worst_fitness = worst_fitness

    def update(self, worst_fitness: float = None, **kwargs) -> None:
        """
        Update barrier parameters (specifically, the worst fitness value).

        This method provides a convenient way to update the worst fitness value
        through the standard Barrier interface update() method.

        Args:
            worst_fitness: The worst (highest) objective function value in the
                          current population (optional)
            **kwargs: Additional keyword arguments (ignored)

        Note:
            If worst_fitness is not provided, the method does nothing.
            This allows the update() method to be called without arguments
            for compatibility with other barrier methods.
        """
        if worst_fitness is not None:
            self.set_worst_fitness(worst_fitness)
