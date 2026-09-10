#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nelder-Mead Algorithm Implementation

This module implements the classic Nelder-Mead simplex optimization algorithm
with configurable parameters, stopping criteria, and restart mechanisms.

The Nelder-Mead algorithm is a direct search method that uses a simplex
(a geometric figure with n+1 vertices in n-dimensional space) to explore
the search space and converge to a local optimum.

References:
    - Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization.
      The Computer Journal, 7(4), 308-313.
    - Lagarias, J. C., et al. (1998). Convergence properties of the Nelder-Mead
      simplex method in low dimensions. SIAM Journal on Optimization, 9(1), 112-147.
"""

from typing import Callable, Optional, List
import numpy as np

from .base_algorithm import BaseAlgorithm
from ..core import simplex_operations as ops
from ..core import stopping_criteria as stop
from ..core.restart_strategies import create_restart_strategy
from ..constraints.barrier_base import Barrier


class NelderMead(BaseAlgorithm):
    """
    Classic Nelder-Mead simplex optimization algorithm.

    The algorithm maintains a simplex of n+1 points in n-dimensional space
    and iteratively transforms the simplex using reflection, expansion,
    contraction, and shrink operations to search for the optimum.

    Attributes:
        delta_r (float): Reflection coefficient (default: 1.0)
        delta_e (float): Expansion coefficient (default: 2.0)
        delta_oc (float): Outside contraction coefficient (default: 0.5)
        delta_ic (float): Inside contraction coefficient (default: -0.5)
        gamma_s (float): Shrink coefficient (default: 0.5)
        restart_strategy (RestartStrategy): Strategy for reinitializing simplex
        stopping_criteria (List[str]): List of stopping criterion names to check
        simplex (np.ndarray): Current simplex vertices (n+1, n)
        fitness_values (np.ndarray): Fitness values for each vertex
        eq_violations_list (List[np.ndarray]): Equality violations for each vertex
        ineq_violations_list (List[np.ndarray]): Inequality violations for each vertex
        total_violations (np.ndarray): Total constraint violations for each vertex
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        constraint_eq_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        constraint_ineq_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        lower_bounds: np.ndarray = None,
        upper_bounds: np.ndarray = None,
        max_fes: int = 10000,
        num_solutions: Optional[int] = None,
        init_method: str = "spendleySimplex",
        seed: int = 101,
        barrier: Optional[Barrier] = None,
        delta_r: float = 1.0,
        delta_e: float = 2.0,
        delta_oc: float = 0.5,
        delta_ic: float = -0.5,
        gamma_s: float = 0.5,
        restart_strategy: str = "gaussian_best",
        stopping_criteria: Optional[List[str]] = None,
        criterion_tolerances: Optional[dict] = None,
        trace_conditioning: bool = False,
    ):
        """
        Initialize the Nelder-Mead algorithm.

        Args:
            objective_fn: Function to minimize, takes np.ndarray and returns float
            constraint_eq_fn: Equality constraint function (returns violations as array)
            constraint_ineq_fn: Inequality constraint function (returns violations as array)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            max_fes: Maximum number of function evaluations (default: 10000)
            num_solutions: Number of solutions in simplex (default: dim+1)
            init_method: Initialization method name (default: "spendleySimplex")
            seed: Random seed for reproducibility (default: 101)
            barrier: Barrier method for constraint handling (default: None)
            delta_r: Reflection coefficient (default: 1.0)
            delta_e: Expansion coefficient (default: 2.0)
            delta_oc: Outside contraction coefficient (default: 0.5)
            delta_ic: Inside contraction coefficient (default: -0.5)
            gamma_s: Shrink coefficient (default: 0.5)
            restart_strategy: Name of restart strategy (default: "gaussian_best")
            stopping_criteria: List of stopping criterion names (default: ["fminsearch_fun", "fminsearch_x"])

        Raises:
            ValueError: If restart_strategy is not valid or stopping criteria are invalid

        Validates: Requirements 17.1, 17.2, 17.3, 17.4, 17.5
        """
        # Initialize base algorithm
        super().__init__(
            objective_fn=objective_fn,
            constraint_eq_fn=constraint_eq_fn,
            constraint_ineq_fn=constraint_ineq_fn,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            max_fes=max_fes,
            num_solutions=num_solutions,
            init_method=init_method,
            seed=seed,
            barrier=barrier,
        )

        # Configure simplex operation coefficients (Requirement 17.2)
        self.delta_r = delta_r  # Reflection coefficient
        self.delta_e = delta_e  # Expansion coefficient
        self.delta_oc = delta_oc  # Outside contraction coefficient
        self.delta_ic = delta_ic  # Inside contraction coefficient
        self.gamma_s = gamma_s  # Shrink coefficient

        # Configure restart strategy (Requirement 17.5)
        valid_restart_strategies = ["uniform", "gaussian", "gaussian_best"]
        if restart_strategy not in valid_restart_strategies:
            raise ValueError(
                f"'{restart_strategy}' is not a valid restart strategy. "
                f"Available strategies: {', '.join(valid_restart_strategies)}"
            )
        # Pass the same RNG to restart strategy for reproducibility
        self.restart_strategy = create_restart_strategy(restart_strategy, rng=self.rng)

        # Configure stopping criteria (Requirement 17.4)
        if stopping_criteria is None:
            # Default to MATLAB fminsearch-style criteria
            stopping_criteria = ["fminsearch_fun", "fminsearch_x"]

        valid_criteria = [
            "oriented_length",
            "std_dev",
            "small_simplex",
            "flat_simplex",
            "degenerate_simplex",
            "fminsearch_fun",
            "fminsearch_x",
        ]
        for criterion in stopping_criteria:
            if criterion not in valid_criteria:
                raise ValueError(
                    f"'{criterion}' is not a valid stopping criterion. "
                    f"Available criteria: {', '.join(valid_criteria)}"
                )
        self.stopping_criteria = stopping_criteria

        # Tolerances for the criteria above. Exposed because the degeneracy
        # tolerances in particular are what Luersen & Le Riche (2004) report as
        # "difficult to tune" -- calibrating them is an experiment, not a constant.
        self.criterion_tolerances = {
            "oriented_length": 1e-4,
            "std_dev": 1e-7,
            "small_simplex": 1e-8,
            "flat_simplex": 1e-8,
            "degenerate_edge": 1e-6,      # GBNM eps_s3
            "degenerate_hadamard": 1e-6,  # GBNM eps_s4
            "fminsearch_fun": 1e-12,
            "fminsearch_x": 1e-8,
        }
        if criterion_tolerances:
            unknown = set(criterion_tolerances) - set(self.criterion_tolerances)
            if unknown:
                raise ValueError(f"Unknown criterion tolerances: {sorted(unknown)}")
            self.criterion_tolerances.update(criterion_tolerances)

        # Instrumentation for the initialization x restart study. Off by default:
        # the trace is one row per iteration and is pure overhead otherwise.
        self.trace_conditioning = trace_conditioning
        self.conditioning_trace: List[tuple] = []  # (fes, edge_ratio, hadamard_ratio)
        self.restart_log: List[tuple] = []         # (fes, criterion_that_fired)
        self.last_trigger: Optional[str] = None

        # Initialize simplex state (will be populated in run())
        self.simplex = None
        self.fitness_values = None
        self.eq_violations_list = None
        self.ineq_violations_list = None
        self.total_violations = None

    def _check_stopping_criteria(self) -> bool:
        """
        Check if any configured stopping criterion is met.

        Returns:
            True if any stopping criterion is met, False otherwise

        Validates: Requirements 4.1-4.5
        """
        tol = self.criterion_tolerances
        for criterion in self.stopping_criteria:
            fired = False
            if criterion == "oriented_length":
                fired = stop.check_oriented_length(
                    self.simplex, epsilon=tol["oriented_length"]
                )
            elif criterion == "std_dev":
                fired = stop.check_std_dev(
                    self.fitness_values, epsilon=tol["std_dev"]
                )
            elif criterion == "small_simplex":
                bounds = (self.lower_bounds, self.upper_bounds)
                fired = stop.check_small_simplex(
                    self.simplex, bounds, epsilon=tol["small_simplex"]
                )
            elif criterion == "flat_simplex":
                fired = stop.check_flat_simplex(
                    np.max(self.fitness_values),
                    np.min(self.fitness_values),
                    epsilon=tol["flat_simplex"],
                )
            elif criterion == "degenerate_simplex":
                fired = stop.check_degenerate_simplex(
                    self.simplex,
                    epsilon1=tol["degenerate_edge"],
                    epsilon2=tol["degenerate_hadamard"],
                )
            elif criterion == "fminsearch_fun":
                fired = stop.check_fminsearch_fun(
                    self.fitness_values, tol_fun=tol["fminsearch_fun"]
                )
            elif criterion == "fminsearch_x":
                fired = stop.check_fminsearch_x(
                    self.simplex, tol_x=tol["fminsearch_x"]
                )

            if fired:
                # Record which criterion fired. GBNM's authors report that a
                # shrinking simplex gets tagged "degenerate" before it gets
                # tagged "small"; attribution is how we measure that.
                self.last_trigger = criterion
                return True

        return False

    def _restart_simplex(self) -> None:
        """
        Restart the simplex using the configured restart strategy.

        Generates new simplex vertices around the best solution and re-evaluates them.
        The best solution is preserved as the first vertex.

        Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
        """
        # Keep the best solution as the first vertex
        best_solution = self.simplex[0].copy()

        # Generate new vertices using restart strategy
        bounds = (self.lower_bounds, self.upper_bounds)
        new_simplex = np.zeros_like(self.simplex)
        new_simplex[0] = best_solution

        for i in range(1, self.num_solutions):
            # Generate new point
            new_point = self.restart_strategy.generate_point(
                self.dim, bounds, best_solution=best_solution
            )
            # Enforce bounds
            new_point = ops.enforce_bounds(
                new_point, self.lower_bounds, self.upper_bounds
            )
            new_simplex[i] = new_point

        # Update simplex
        self.simplex = new_simplex

        # Re-evaluate all vertices (Requirement 5.5)
        (
            self.fitness_values,
            self.eq_violations_list,
            self.ineq_violations_list,
            self.total_violations,
        ) = self.evaluate_population(self.simplex)

        # Sort simplex by fitness
        self._sort_simplex()

    def _on_contraction_failure(
        self,
        kind: str,
        contracted_point: np.ndarray,
        contracted_fitness: float,
        centroid: np.ndarray,
        worst_point: np.ndarray,
    ) -> bool:
        """
        Hook called when a contraction fails, just before falling back to shrink.

        Classic Nelder-Mead always shrinks here, so this is a no-op returning False.
        Subclasses may attempt a rescue step and return True to skip the shrink.

        Args:
            kind: Either "inside" or "outside"
            contracted_point: The contraction point that was rejected
            contracted_fitness: Its objective value
            centroid: Centroid of the n best vertices
            worst_point: The worst vertex the contraction was computed from

        Returns:
            True if the failure was handled and the shrink should be skipped
        """
        return False

    def _on_accept(self, point: np.ndarray, fitness: float) -> None:
        """
        Hook called whenever a geometric transformation produces an accepted point.

        Classic Nelder-Mead keeps no history, so this is a no-op. Subclasses that
        maintain a record of previously sampled points (e.g. GradientNelderMead)
        override this to capture every accepted vertex.

        Args:
            point: The accepted point that replaced the worst vertex
            fitness: Objective value of the accepted point
        """
        pass

    def _sort_simplex(self) -> None:
        """
        Sort simplex vertices by fitness (best to worst).

        Updates simplex, fitness_values, and violation arrays in place.
        """
        # Get sort indices (ascending order - best fitness first)
        sort_indices = np.argsort(self.fitness_values)

        # Sort all arrays
        self.simplex = self.simplex[sort_indices]
        self.fitness_values = self.fitness_values[sort_indices]
        self.total_violations = self.total_violations[sort_indices]

        # Sort violation lists
        self.eq_violations_list = [self.eq_violations_list[i] for i in sort_indices]
        self.ineq_violations_list = [self.ineq_violations_list[i] for i in sort_indices]

    def run(self) -> None:
        """
        Execute the Nelder-Mead optimization algorithm.

        Algorithm flow:
        1. Initialize simplex with d+1 solutions
        2. Evaluate initial solutions
        3. Main loop:
           a. Sort simplex by fitness
           b. Check stopping criteria → restart if met
           c. Compute centroid (excluding worst point)
           d. Attempt reflection
           e. If reflection is best → attempt expansion
           f. If reflection is good → accept reflection
           g. If reflection is mediocre → attempt outside contraction
           h. If reflection is worst → attempt inside contraction
           i. If all contractions fail → shrink entire simplex
        4. Repeat until max FES reached

        Validates: Requirements 2.1-2.6, 4.1-4.5, 5.4, 5.5, 17.1
        """
        # Initialize simplex with d+1 solutions (Requirement 17.1)
        self.simplex = self.initialize_population()

        # Evaluate initial population
        (
            self.fitness_values,
            self.eq_violations_list,
            self.ineq_violations_list,
            self.total_violations,
        ) = self.evaluate_population(self.simplex)

        # Sort simplex by fitness (best to worst)
        self._sort_simplex()

        # Main optimization loop
        while not self.should_terminate():
            if self.trace_conditioning:
                self.conditioning_trace.append(
                    (self.fes,) + ops.simplex_conditioning(self.simplex)
                )

            # Check stopping criteria and restart if met (Requirement 5.4)
            if self._check_stopping_criteria():
                if self.trace_conditioning:
                    self.restart_log.append((self.fes, self.last_trigger))
                self._restart_simplex()
                continue

            # Get worst point and fitness values
            worst_point = self.simplex[-1]
            best_fitness = self.fitness_values[0]
            worst_fitness = self.fitness_values[-1]
            second_worst_fitness = self.fitness_values[-2]

            # Compute centroid excluding worst point (Requirement 2.6)
            centroid = ops.compute_centroid(self.simplex, exclude_worst=True)

            # Attempt reflection (Requirement 2.1)
            reflected_point = ops.reflect(centroid, worst_point, self.delta_r)
            reflected_point = ops.enforce_bounds(
                reflected_point, self.lower_bounds, self.upper_bounds
            )

            # Evaluate reflected point
            if self.should_terminate():
                break
            (
                reflected_fitness,
                reflected_eq_vio,
                reflected_ineq_vio,
                reflected_total_vio,
            ) = self.evaluate(reflected_point)

            # Decision logic based on reflected point quality
            # The Nelder-Mead algorithm uses a hierarchical decision tree to determine
            # which operation to apply based on the quality of the reflected point
            # relative to the current simplex vertices.

            if reflected_fitness < best_fitness:
                # Reflection is best → attempt expansion (Requirement 2.2)
                # If reflection improved upon the best point, the search direction is
                # very promising, so we try to go even further with expansion
                expanded_point = ops.expand(centroid, reflected_point, self.delta_e)
                expanded_point = ops.enforce_bounds(
                    expanded_point, self.lower_bounds, self.upper_bounds
                )

                if self.should_terminate():
                    break
                (
                    expanded_fitness,
                    expanded_eq_vio,
                    expanded_ineq_vio,
                    expanded_total_vio,
                ) = self.evaluate(expanded_point)

                # Accept better of expansion or reflection
                # If expansion succeeded, use it; otherwise, reflection was still good
                if expanded_fitness < reflected_fitness:
                    self.simplex[-1] = expanded_point
                    self.fitness_values[-1] = expanded_fitness
                    self.eq_violations_list[-1] = expanded_eq_vio
                    self.ineq_violations_list[-1] = expanded_ineq_vio
                    self.total_violations[-1] = expanded_total_vio
                    self._on_accept(expanded_point, expanded_fitness)
                else:
                    self.simplex[-1] = reflected_point
                    self.fitness_values[-1] = reflected_fitness
                    self.eq_violations_list[-1] = reflected_eq_vio
                    self.ineq_violations_list[-1] = reflected_ineq_vio
                    self.total_violations[-1] = reflected_total_vio
                    self._on_accept(reflected_point, reflected_fitness)

            elif reflected_fitness < second_worst_fitness:
                # Reflection is good (better than second worst) → accept reflection
                # The reflected point is not the best, but it's better than at least
                # one other point, so it's worth keeping
                self.simplex[-1] = reflected_point
                self.fitness_values[-1] = reflected_fitness
                self.eq_violations_list[-1] = reflected_eq_vio
                self.ineq_violations_list[-1] = reflected_ineq_vio
                self.total_violations[-1] = reflected_total_vio
                self._on_accept(reflected_point, reflected_fitness)

            elif reflected_fitness < worst_fitness:
                # Reflection is mediocre → attempt outside contraction (Requirement 2.3)
                # The reflected point is better than the worst but not good enough to
                # accept directly. Try contracting it back toward the centroid.
                contracted_point = ops.contract_outside(
                    centroid, reflected_point, self.delta_oc
                )
                contracted_point = ops.enforce_bounds(
                    contracted_point, self.lower_bounds, self.upper_bounds
                )

                if self.should_terminate():
                    break
                (
                    contracted_fitness,
                    contracted_eq_vio,
                    contracted_ineq_vio,
                    contracted_total_vio,
                ) = self.evaluate(contracted_point)

                if contracted_fitness < reflected_fitness:
                    # Accept outside contraction
                    self.simplex[-1] = contracted_point
                    self.fitness_values[-1] = contracted_fitness
                    self.eq_violations_list[-1] = contracted_eq_vio
                    self.ineq_violations_list[-1] = contracted_ineq_vio
                    self.total_violations[-1] = contracted_total_vio
                    self._on_accept(contracted_point, contracted_fitness)
                elif not self._on_contraction_failure(
                    "outside",
                    contracted_point,
                    contracted_fitness,
                    centroid,
                    worst_point,
                ):
                    # Contraction failed → shrink (Requirement 2.5)
                    # Neither reflection nor contraction produced an acceptable point,
                    # so we shrink the entire simplex toward the best point
                    self._shrink_simplex()

            else:
                # Reflection is worst → attempt inside contraction (Requirement 2.4)
                # The reflected point is even worse than the current worst point,
                # indicating we went in the wrong direction. Try contracting toward
                # the centroid from the worst point instead.
                contracted_point = ops.contract_inside(
                    centroid, worst_point, self.delta_ic
                )
                contracted_point = ops.enforce_bounds(
                    contracted_point, self.lower_bounds, self.upper_bounds
                )

                if self.should_terminate():
                    break
                (
                    contracted_fitness,
                    contracted_eq_vio,
                    contracted_ineq_vio,
                    contracted_total_vio,
                ) = self.evaluate(contracted_point)

                if contracted_fitness < worst_fitness:
                    # Accept inside contraction
                    self.simplex[-1] = contracted_point
                    self.fitness_values[-1] = contracted_fitness
                    self.eq_violations_list[-1] = contracted_eq_vio
                    self.ineq_violations_list[-1] = contracted_ineq_vio
                    self.total_violations[-1] = contracted_total_vio
                    self._on_accept(contracted_point, contracted_fitness)
                elif not self._on_contraction_failure(
                    "inside",
                    contracted_point,
                    contracted_fitness,
                    centroid,
                    worst_point,
                ):
                    # Contraction failed → shrink (Requirement 2.5)
                    # Even inside contraction didn't help, so we shrink the entire
                    # simplex toward the best point as a last resort
                    self._shrink_simplex()

            # Sort simplex for next iteration
            self._sort_simplex()

    def _shrink_simplex(self) -> None:
        """
        Shrink all simplex vertices toward the best point.

        Applies the shrink operation to all vertices except the best,
        then re-evaluates all modified vertices.

        Validates: Requirements 2.5
        """
        # Get best point
        best_point = self.simplex[0]

        # Shrink all vertices toward best
        self.simplex = ops.shrink(best_point, self.simplex, self.gamma_s)

        # Enforce bounds on all vertices
        for i in range(len(self.simplex)):
            self.simplex[i] = ops.enforce_bounds(
                self.simplex[i], self.lower_bounds, self.upper_bounds
            )

        # Re-evaluate all vertices except the best (which hasn't changed)
        for i in range(1, self.num_solutions):
            if self.should_terminate():
                break
            fitness, eq_vio, ineq_vio, total_vio = self.evaluate(self.simplex[i])
            self.fitness_values[i] = fitness
            self.eq_violations_list[i] = eq_vio
            self.ineq_violations_list[i] = ineq_vio
            self.total_violations[i] = total_vio
