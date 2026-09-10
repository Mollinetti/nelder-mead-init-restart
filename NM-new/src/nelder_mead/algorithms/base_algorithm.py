#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Algorithm Class for Optimization Algorithms

This module defines the abstract base class for all optimization algorithms.
It provides common functionality including:
- Function evaluation tracking (FES)
- Best/worst solution tracking
- Barrier method integration for constraint handling
- Termination checking
- Input validation and error handling

References:
    - Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization.
      The Computer Journal, 7(4), 308-313.
    - Deb, K. (2000). An efficient constraint handling method for genetic algorithms.
      Computer Methods in Applied Mechanics and Engineering, 186(2-4), 311-338.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List
import numpy as np
import warnings

from ..constraints.barrier_base import Barrier
from ..initialization.simplex_init import SimplexInitializer


class BaseAlgorithm(ABC):
    """
    Abstract base class for optimization algorithms.

    This class provides common functionality for all optimization algorithms,
    including function evaluation tracking, best/worst solution management,
    constraint handling through barrier methods, and termination checking.

    Attributes:
        objective_fn (Callable): Objective function to minimize
        constraint_eq_fn (Optional[Callable]): Equality constraint function
        constraint_ineq_fn (Optional[Callable]): Inequality constraint function
        lower_bounds (np.ndarray): Lower bounds for decision variables
        upper_bounds (np.ndarray): Upper bounds for decision variables
        dim (int): Problem dimension
        max_fes (int): Maximum function evaluations allowed
        num_solutions (int): Number of solutions in population
        init_method (str): Initialization method name
        seed (int): Random seed for reproducibility
        barrier (Barrier): Barrier method for constraint handling
        rng (np.random.Generator): Random number generator
        fes (int): Current function evaluation count
        best_solution (np.ndarray): Best solution found so far
        best_fitness (float): Best fitness value found
        best_eq_violations (np.ndarray): Equality violations of best solution
        best_ineq_violations (np.ndarray): Inequality violations of best solution
        best_total_violation (float): Total constraint violation of best solution
        worst_fitness (float): Worst fitness in current population
        memory_global_best (List[Tuple[int, float]]): History of (FES, best_fitness)
        tracking_interval (int): Interval for recording convergence history
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
    ):
        """
        Initialize the base algorithm.

        Args:
            objective_fn: Function to minimize, takes np.ndarray and returns float
            constraint_eq_fn: Equality constraint function (returns violations as array)
            constraint_ineq_fn: Inequality constraint function (returns violations as array)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            max_fes: Maximum number of function evaluations
            num_solutions: Number of solutions in population (if None, set to dim+1)
            init_method: Initialization method name
            seed: Random seed for reproducibility
            barrier: Barrier method for constraint handling (if None, no constraints)

        Raises:
            TypeError: If objective_fn is not callable or bounds are not arrays
            ValueError: If bounds are inconsistent or parameters are invalid
        """
        # Validate objective function
        if not callable(objective_fn):
            raise TypeError("objective_fn must be callable")

        # Validate bounds
        if lower_bounds is None or upper_bounds is None:
            raise ValueError("lower_bounds and upper_bounds must be provided")

        if not isinstance(lower_bounds, np.ndarray) or not isinstance(
            upper_bounds, np.ndarray
        ):
            raise TypeError("lower_bounds and upper_bounds must be numpy arrays")

        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError("Lower and upper bounds must have the same shape")

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError(
                "Lower bounds must be less than upper bounds in all dimensions"
            )

        # Store problem definition
        self.objective_fn = objective_fn
        self.constraint_eq_fn = constraint_eq_fn
        self.constraint_ineq_fn = constraint_ineq_fn
        self.lower_bounds = lower_bounds.copy()
        self.upper_bounds = upper_bounds.copy()
        self.dim = len(lower_bounds)

        # Validate and store algorithm parameters
        if max_fes <= 0:
            raise ValueError("max_fes must be positive")
        self.max_fes = max_fes

        # Set num_solutions (default to dim+1 for simplex-based algorithms)
        if num_solutions is None:
            num_solutions = self.dim + 1

        if num_solutions <= 0:
            raise ValueError("num_solutions must be positive")

        # Warn if max_fes is too small
        if max_fes < num_solutions:
            warnings.warn(
                f"max_fes ({max_fes}) < num_solutions ({num_solutions}). "
                "Setting max_fes = num_solutions",
                UserWarning,
            )
            self.max_fes = num_solutions

        self.num_solutions = num_solutions

        # Validate initialization method
        valid_methods = [
            "uniform",
            "gaussian",
            "spendleySimplex",
            "pfefferSimplex",
            "adaptiveSimplex",
            "minimalPositiveBasis",
        ]
        if init_method not in valid_methods:
            raise ValueError(
                f"'{init_method}' is not a valid initialization method. "
                f"Available: {', '.join(valid_methods)}"
            )
        self.init_method = init_method

        # Set up random number generator for reproducibility
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Store barrier method
        self.barrier = barrier

        # Initialize tracking variables
        self.fes = 0  # Function evaluation counter

        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float("inf")
        self.best_eq_violations = np.array([])
        self.best_ineq_violations = np.array([])
        self.best_total_violation = float("inf")

        # Worst fitness tracking (for Deb barrier)
        self.worst_fitness = float("-inf")

        # Convergence history tracking
        self.memory_global_best = []  # List of (fes, best_fitness) tuples
        self.tracking_interval = max(1, max_fes // 100)  # Record every 1% of budget

        # Population storage (for subclasses)
        self.population = None
        self.fitness_values = None

    def evaluate(
        self, solution: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Evaluate a solution and update tracking variables.

        This method:
        1. Increments the function evaluation counter
        2. Evaluates the objective function
        3. Evaluates constraint functions (if present)
        4. Applies barrier penalty (if barrier method is used)
        5. Updates best/worst solution tracking
        6. Records convergence history at intervals

        Args:
            solution: Decision variable vector to evaluate

        Returns:
            Tuple of (penalized_fitness, eq_violations, ineq_violations, total_violation)
            - penalized_fitness: Objective value + constraint penalty
            - eq_violations: Array of equality constraint violations
            - ineq_violations: Array of inequality constraint violations
            - total_violation: Sum of all constraint violations

        Validates: Requirements 13.1, 13.4, 14.1, 14.2, 14.5
        """
        # Increment function evaluation counter
        self.fes += 1

        # Evaluate objective function with error handling
        try:
            fitness = self.objective_fn(solution)

            # Check for NaN or Inf
            if not np.isfinite(fitness):
                warnings.warn(
                    f"Objective function returned non-finite value: {fitness}. "
                    "Replacing with large penalty.",
                    UserWarning,
                )
                fitness = 1e10
        except Exception as e:
            warnings.warn(
                f"Objective function evaluation failed: {str(e)}. "
                "Assigning large penalty value.",
                UserWarning,
            )
            fitness = 1e10

        # Evaluate constraint functions
        eq_violations = np.array([])
        ineq_violations = np.array([])

        if self.constraint_eq_fn is not None:
            try:
                eq_vals = self.constraint_eq_fn(solution)
                # Equality violations are absolute values
                eq_violations = np.abs(eq_vals)
            except Exception as e:
                warnings.warn(
                    f"Equality constraint evaluation failed: {str(e)}. "
                    "Treating as maximum violation.",
                    UserWarning,
                )
                # Assume worst case violation
                eq_violations = np.array([1e6])

        if self.constraint_ineq_fn is not None:
            try:
                ineq_vals = self.constraint_ineq_fn(solution)
                # Inequality violations are positive parts (g(x) > 0 means violation)
                ineq_violations = np.maximum(ineq_vals, 0)
            except Exception as e:
                warnings.warn(
                    f"Inequality constraint evaluation failed: {str(e)}. "
                    "Treating as maximum violation.",
                    UserWarning,
                )
                # Assume worst case violation
                ineq_violations = np.array([1e6])

        # Compute total constraint violation
        if self.barrier is not None:
            total_violation = self.barrier.compute_total_violation(
                solution, eq_violations, ineq_violations
            )
        else:
            # No barrier method - just sum violations
            bound_vio = np.sum(np.maximum(self.lower_bounds - solution, 0)) + np.sum(
                np.maximum(solution - self.upper_bounds, 0)
            )
            eq_vio = np.sum(eq_violations) if len(eq_violations) > 0 else 0.0
            ineq_vio = np.sum(ineq_violations) if len(ineq_violations) > 0 else 0.0
            total_violation = bound_vio + eq_vio + ineq_vio

        # Apply barrier penalty if present
        penalized_fitness = fitness
        if self.barrier is not None:
            penalty = self.barrier.penalize_constraint_violation(
                solution, eq_violations, ineq_violations
            )
            penalized_fitness = fitness + penalty

        # Update worst fitness (for Deb barrier and other methods)
        if penalized_fitness > self.worst_fitness:
            self.worst_fitness = penalized_fitness

        # Update best solution using constraint-aware comparison
        if self._is_better_solution(
            fitness, total_violation, self.best_fitness, self.best_total_violation
        ):
            self.best_solution = solution.copy()
            self.best_fitness = fitness
            self.best_eq_violations = eq_violations.copy()
            self.best_ineq_violations = ineq_violations.copy()
            self.best_total_violation = total_violation

        # Record convergence history at intervals
        if self.fes % self.tracking_interval == 0 or self.fes == 1:
            self.memory_global_best.append((self.fes, self.best_fitness))

        return penalized_fitness, eq_violations, ineq_violations, total_violation

    def _is_better_solution(
        self, fitness1: float, violation1: float, fitness2: float, violation2: float
    ) -> bool:
        """
        Compare two solutions considering both fitness and constraint violations.

        Comparison rules (Deb's constraint handling):
        1. If both feasible: prefer lower fitness
        2. If one feasible, one infeasible: prefer feasible
        3. If both infeasible: prefer lower total violation

        Args:
            fitness1: Fitness of first solution
            violation1: Total constraint violation of first solution
            fitness2: Fitness of second solution
            violation2: Total constraint violation of second solution

        Returns:
            True if solution 1 is better than solution 2

        Validates: Requirements 14.3
        """
        # Define feasibility threshold (small tolerance for numerical errors)
        feasibility_tol = 1e-8

        feasible1 = violation1 < feasibility_tol
        feasible2 = violation2 < feasibility_tol

        if feasible1 and feasible2:
            # Both feasible: prefer lower fitness
            return fitness1 < fitness2
        elif feasible1 and not feasible2:
            # Solution 1 feasible, solution 2 infeasible: prefer solution 1
            return True
        elif not feasible1 and feasible2:
            # Solution 1 infeasible, solution 2 feasible: prefer solution 2
            return False
        else:
            # Both infeasible: prefer lower violation
            return violation1 < violation2

    def should_terminate(self) -> bool:
        """
        Check if the algorithm should terminate.

        Termination occurs when the function evaluation budget is exhausted.

        Returns:
            True if FES >= max_fes, False otherwise

        Validates: Requirements 13.2
        """
        return self.fes >= self.max_fes

    def get_best_solution(
        self,
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
        """
        Return the best solution found during optimization.

        Returns:
            Tuple of (solution, fitness, eq_violations, ineq_violations, total_violation)
            - solution: Best decision variable vector
            - fitness: Best objective function value
            - eq_violations: Equality constraint violations of best solution
            - ineq_violations: Inequality constraint violations of best solution
            - total_violation: Total constraint violation of best solution

        Validates: Requirements 14.4
        """
        return (
            self.best_solution,
            self.best_fitness,
            self.best_eq_violations,
            self.best_ineq_violations,
            self.best_total_violation,
        )

    def initialize_population(self, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Initialize the population using the specified initialization method.

        Args:
            x0: Optional starting point for structured initialization methods

        Returns:
            Array of shape (num_solutions, dim) with initialized population

        Validates: Requirements 3.1-3.5, 13.4
        """
        population = SimplexInitializer.initialize(
            method=self.init_method,
            num_solutions=self.num_solutions,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            dim=self.dim,
            x0=x0,
            rng=self.rng,
        )

        return population

    def evaluate_population(
        self, population: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Evaluate all solutions in a population.

        Args:
            population: Array of shape (num_solutions, dim)

        Returns:
            Tuple of (fitness_values, eq_violations_list, ineq_violations_list, total_violations)
            - fitness_values: Array of penalized fitness values
            - eq_violations_list: List of equality violation arrays
            - ineq_violations_list: List of inequality violation arrays
            - total_violations: Array of total constraint violations
        """
        num_sols = len(population)
        fitness_values = np.zeros(num_sols)
        eq_violations_list = []
        ineq_violations_list = []
        total_violations = np.zeros(num_sols)

        for i in range(num_sols):
            fitness, eq_vio, ineq_vio, total_vio = self.evaluate(population[i])
            fitness_values[i] = fitness
            eq_violations_list.append(eq_vio)
            ineq_violations_list.append(ineq_vio)
            total_violations[i] = total_vio

        return (
            fitness_values,
            eq_violations_list,
            ineq_violations_list,
            total_violations,
        )

    @abstractmethod
    def run(self) -> None:
        """
        Execute the optimization algorithm.

        This method must be implemented by subclasses to define the specific
        optimization strategy (e.g., Nelder-Mead, Adaptive Nelder-Mead, etc.).

        The implementation should:
        1. Initialize the population
        2. Evaluate initial solutions
        3. Execute the main optimization loop
        4. Check termination conditions
        5. Update the barrier method if needed
        """
        pass
