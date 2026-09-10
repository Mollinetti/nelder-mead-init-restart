#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progressive Barrier Method for Constraint Handling

This module implements the Progressive Barrier method, which adaptively manages
a feasibility threshold to guide the search toward feasible regions. The method
tracks both the best feasible and best infeasible solutions, and adjusts the
threshold based on optimization progress.

The Progressive Barrier uses an adaptive penalty strategy:
- Solutions within the threshold: penalty = actual violation
- Solutions beyond the threshold: penalty = violation * large_penalty_coefficient

The threshold is updated based on:
1. When a better feasible solution is found: maintain current threshold
2. When a better infeasible solution is found within threshold: update threshold to that violation
3. When no better infeasible solution is found: update threshold to max violation among dominating solutions

This approach provides a smooth transition from infeasible to feasible regions,
allowing the algorithm to explore promising infeasible solutions while maintaining
pressure toward feasibility.

References:
    - Takahama, T., & Sakai, S. (2006). Constrained optimization by the ε constrained
      differential evolution with gradient-based mutation and feasible elites.
      IEEE Congress on Evolutionary Computation (CEC), 1-8.
    - Tessema, B., & Yen, G. G. (2009). An adaptive penalty formulation for constrained
      evolutionary optimization. IEEE Transactions on Systems, Man, and Cybernetics-Part A:
      Systems and Humans, 39(3), 565-578.
"""

from typing import Tuple, Optional, Callable
import numpy as np

from nelder_mead.constraints.barrier_base import Barrier


class ProgressiveBarrier(Barrier):
    """
    Progressive Barrier method for constrained optimization.

    The Progressive Barrier maintains a dynamic feasibility threshold that adapts
    based on the quality of solutions found. It tracks:
    - Best feasible solution (if any)
    - Best infeasible solution within the current threshold
    - Current feasibility threshold

    The penalty is computed as:
        - If total_violation <= threshold: penalty = total_violation
        - If total_violation > threshold: penalty = total_violation * penalty_coeff

    The threshold is updated after each iteration based on optimization progress:
    - Better feasible solution found: keep current threshold
    - Better infeasible solution found within threshold: update threshold to that violation
    - No better solution found: update threshold to max violation among dominating solutions

    Attributes:
        bounds (Tuple[np.ndarray, np.ndarray]): Lower and upper bounds
        penalty_coeff (float): Large penalty coefficient for violations beyond threshold
        rho_fn (Callable): Function to compute violation magnitude (default: L2 norm)
        t (int): Counter tracking penalty function evaluations
        threshold (float): Current feasibility threshold
        best_feasible (Optional[Tuple]): Best feasible solution (solution, obj_val, violation)
        best_infeasible (Optional[Tuple]): Best infeasible solution within threshold

    Validates: Requirements 7.4, 9.1, 9.2, 9.3, 9.4, 9.5
    """

    def __init__(
        self,
        bounds: Tuple[np.ndarray, np.ndarray],
        penalty_coeff: float = 1e4,
        rho_fn: Optional[Callable[[np.ndarray], float]] = None,
    ):
        """
        Initialize the Progressive Barrier method.

        Args:
            bounds: Tuple of (lower_bounds, upper_bounds) as numpy arrays
            penalty_coeff: Large penalty coefficient for violations beyond threshold (default: 1e4)
            rho_fn: Function to compute violation magnitude (default: L2 norm)

        Note:
            The num_solutions parameter is not used by Progressive Barrier but is
            required by the Barrier interface. We pass a dummy value of 1.

            The rho_fn parameter allows customization of how constraint violations
            are aggregated. Common choices:
            - np.linalg.norm (L2 norm, default)
            - lambda x: np.linalg.norm(x, ord=1) (L1 norm)
            - lambda x: np.linalg.norm(x, ord=np.inf) (L-infinity norm)
        """
        # Pass dummy num_solutions=1 since Progressive Barrier doesn't use it
        super().__init__(bounds, num_solutions=1, penalty_coeff=penalty_coeff)

        # Set violation aggregation function (default: L2 norm)
        if rho_fn is None:
            self.rho_fn = np.linalg.norm
        else:
            self.rho_fn = rho_fn

        # Initialize threshold to infinity (accept all solutions initially)
        self.threshold = float("inf")

        # Track best feasible solution: (solution, obj_val, violation)
        self.best_feasible: Optional[Tuple[np.ndarray, float, float]] = None

        # Track best infeasible solution within threshold: (solution, obj_val, violation)
        self.best_infeasible: Optional[Tuple[np.ndarray, float, float]] = None

    def penalize_constraint_violation(
        self,
        solution: np.ndarray,
        eq_violations: np.ndarray,
        ineq_violations: np.ndarray,
    ) -> float:
        """
        Compute penalty for constraint violations using Progressive Barrier.

        The penalty depends on whether the violation is within the current threshold:
        - If total_violation <= threshold: penalty = total_violation
        - If total_violation > threshold: penalty = total_violation * penalty_coeff

        This creates a smooth penalty landscape within the threshold while strongly
        penalizing solutions far from feasibility. The key insight is that by
        maintaining an adaptive threshold, we allow the algorithm to explore
        promising infeasible regions (those close to feasibility) while still
        maintaining pressure toward feasibility.

        The threshold starts at infinity (accepting all solutions) and gradually
        decreases as better solutions are found, creating a progressive tightening
        of the feasibility requirement. This is particularly effective for problems
        where the feasible region is small or difficult to find, as it allows the
        algorithm to approach feasibility gradually rather than requiring immediate
        feasibility.

        Reference: Takahama & Sakai (2006), Section 2.2 - ε-constrained method
        Reference: Tessema & Yen (2009), Section III - adaptive penalty formulation

        Args:
            solution: Decision variable vector
            eq_violations: Array of equality constraint violations |h(x)|
            ineq_violations: Array of inequality constraint violations max(0, g(x))

        Returns:
            Penalty value to be added to the objective function

        Validates: Requirement 9.4 - "WHEN a solution violates constraints beyond the
                   threshold, THE ProgressiveBarrier SHALL apply a large penalty"
        Validates: Requirement 9.5 - "WHEN a solution violates constraints within the
                   threshold, THE ProgressiveBarrier SHALL apply the actual violation value"

        Note:
            This method computes the penalty but does NOT update the threshold or
            best solutions. Call update() after evaluating all solutions in an
            iteration to update the barrier state.
        """
        # Increment evaluation counter
        self.t += 1

        # Compute total constraint violation
        # This includes bound violations, equality violations, and inequality violations
        total_violation = self.compute_total_violation(
            solution, eq_violations, ineq_violations
        )

        # If feasible, return zero penalty
        if total_violation == 0.0:
            return 0.0

        # Apply penalty based on threshold
        if total_violation <= self.threshold:
            # Within threshold: use actual violation (Requirement 9.5)
            # This creates a smooth penalty landscape that guides the search
            # toward feasibility without harsh discontinuities
            penalty = total_violation
        else:
            # Beyond threshold: apply large penalty (Requirement 9.4)
            # This strongly discourages solutions that are far from feasibility
            # The large penalty coefficient (typically 1e4) makes these solutions
            # much worse than any solution within the threshold
            penalty = total_violation * self.penalty_coeff

        return penalty

    def update(
        self,
        solutions: Optional[np.ndarray] = None,
        obj_vals: Optional[np.ndarray] = None,
        eq_violations_list: Optional[list] = None,
        ineq_violations_list: Optional[list] = None,
        **kwargs
    ) -> str:
        """
        Update the Progressive Barrier threshold and best solutions.

        This method should be called after each iteration (after evaluating all
        solutions) to update:
        1. Best feasible solution (if a better one is found)
        2. Best infeasible solution within threshold (if a better one is found)
        3. Feasibility threshold based on optimization progress

        The threshold update follows these rules:
        - If a better feasible solution is found: keep current threshold
        - If a better infeasible solution is found within threshold: update threshold to that violation
        - If no better solution is found: update threshold to max violation among dominating solutions

        Args:
            solutions: Array of solutions from current iteration (shape: num_solutions × dim)
            obj_vals: Array of objective values for solutions
            eq_violations_list: List of equality violation arrays for each solution
            ineq_violations_list: List of inequality violation arrays for each solution
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            String indicating iteration type: "dominating" or "non-dominating"

        Validates: Requirement 9.1 - "WHEN a better feasible solution is found, THE
                   ProgressiveBarrier SHALL update the best feasible solution"
        Validates: Requirement 9.2 - "WHEN a better infeasible solution is found within
                   the threshold, THE ProgressiveBarrier SHALL update the threshold to
                   that violation level"
        Validates: Requirement 9.3 - "WHEN no better infeasible solution is found, THE
                   ProgressiveBarrier SHALL update the threshold to the maximum violation
                   among dominating solutions"

        Note:
            If solutions, obj_vals, or violations are not provided, the method does nothing.
            This allows the update() method to be called without arguments for compatibility
            with other barrier methods.
        """
        # If no data provided, do nothing
        if solutions is None or obj_vals is None:
            return "non-dominating"

        if eq_violations_list is None:
            eq_violations_list = [np.array([]) for _ in range(len(solutions))]

        if ineq_violations_list is None:
            ineq_violations_list = [np.array([]) for _ in range(len(solutions))]

        # Compute total violations for all solutions
        total_violations = []
        for i in range(len(solutions)):
            vio = self.compute_total_violation(
                solutions[i], eq_violations_list[i], ineq_violations_list[i]
            )
            total_violations.append(vio)

        total_violations = np.array(total_violations)

        # Track if we found a better solution (dominating iteration)
        found_better = False

        # Update best feasible solution (Requirement 9.1)
        feasible_mask = total_violations == 0.0
        if np.any(feasible_mask):
            feasible_indices = np.where(feasible_mask)[0]
            feasible_obj_vals = obj_vals[feasible_indices]
            best_feasible_idx = feasible_indices[np.argmin(feasible_obj_vals)]
            best_feasible_obj = obj_vals[best_feasible_idx]

            # Update if this is the first feasible solution or if it's better
            if self.best_feasible is None or best_feasible_obj < self.best_feasible[1]:
                self.best_feasible = (
                    solutions[best_feasible_idx].copy(),
                    best_feasible_obj,
                    0.0,
                )
                found_better = True

        # Update best infeasible solution within threshold (Requirement 9.2)
        infeasible_mask = total_violations > 0.0
        within_threshold_mask = total_violations <= self.threshold
        infeasible_within_threshold = infeasible_mask & within_threshold_mask

        if np.any(infeasible_within_threshold):
            indices = np.where(infeasible_within_threshold)[0]

            # Among infeasible solutions within threshold, find the one with best objective
            best_idx = indices[np.argmin(obj_vals[indices])]
            best_obj = obj_vals[best_idx]
            best_vio = total_violations[best_idx]

            # Update if this is the first infeasible solution or if it's better
            # "Better" means: lower objective value (assuming minimization)
            if self.best_infeasible is None or best_obj < self.best_infeasible[1]:
                self.best_infeasible = (solutions[best_idx].copy(), best_obj, best_vio)
                # Update threshold to this violation level (Requirement 9.2)
                self.threshold = best_vio
                found_better = True

        # If no better solution found, update threshold to max violation among dominating solutions
        # (Requirement 9.3)
        if not found_better:
            # "Dominating solutions" are those that would be preferred in the current iteration
            # This typically means solutions with violations within the current threshold
            dominating_mask = within_threshold_mask

            if np.any(dominating_mask):
                dominating_violations = total_violations[dominating_mask]
                max_dominating_violation = np.max(dominating_violations)

                # Update threshold to max violation among dominating solutions
                # Only update if it would decrease the threshold (make it more restrictive)
                if max_dominating_violation < self.threshold:
                    self.threshold = max_dominating_violation

        # Return iteration type
        return "dominating" if found_better else "non-dominating"

    def get_best_solution(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Get the best solution found so far.

        Returns the best feasible solution if one exists, otherwise returns
        the best infeasible solution within the threshold.

        Returns:
            Tuple of (solution, objective_value, violation) or None if no solutions evaluated
        """
        if self.best_feasible is not None:
            return self.best_feasible
        return self.best_infeasible

    def reset(self) -> None:
        """
        Reset the Progressive Barrier to its initial state.

        This clears all tracked solutions and resets the threshold to infinity.
        Useful for starting a new optimization run.
        """
        self.threshold = float("inf")
        self.best_feasible = None
        self.best_infeasible = None
        self.t = 0
