"""
Benchmark Problem Suite Interface

This module defines the base interface for optimization problems used in
benchmarking and testing. It provides a standardized way to represent
optimization problems with objectives, constraints, bounds, and known optima.

References:
    - CEC benchmark suite for constrained optimization
    - Standard unconstrained test problems (Sphere, Rosenbrock, etc.)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class OptimizationProblem(ABC):
    """
    Base class for optimization problems.

    This class defines the interface for objective functions, equality constraints,
    inequality constraints, and stores problem metadata including name, dimension,
    bounds, and known optimum (if available).

    Attributes:
        name (str): Name of the optimization problem
        dim (int): Dimensionality of the problem
        bounds (Tuple[np.ndarray, np.ndarray]): Lower and upper bounds for each dimension
        optimum_value (Optional[float]): Known optimal objective value (if available)
        optimum_location (Optional[np.ndarray]): Known optimal solution location (if available)
        num_eq_constraints (int): Number of equality constraints
        num_ineq_constraints (int): Number of inequality constraints

    Example:
        >>> class Sphere(OptimizationProblem):
        ...     def __init__(self, dim=10):
        ...         super().__init__(
        ...             name="Sphere",
        ...             dim=dim,
        ...             bounds=(np.full(dim, -100.0), np.full(dim, 100.0)),
        ...             optimum_value=0.0,
        ...             optimum_location=np.zeros(dim)
        ...         )
        ...
        ...     def objective(self, x):
        ...         return np.sum(x**2)
    """

    def __init__(
        self,
        name: str,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        optimum_value: Optional[float] = None,
        optimum_location: Optional[np.ndarray] = None,
        num_eq_constraints: int = 0,
        num_ineq_constraints: int = 0,
    ):
        """
        Initialize an optimization problem.

        Args:
            name: Name of the problem
            dim: Dimensionality of the problem
            bounds: Tuple of (lower_bounds, upper_bounds) as numpy arrays
            optimum_value: Known optimal objective value (None if unknown)
            optimum_location: Known optimal solution (None if unknown)
            num_eq_constraints: Number of equality constraints
            num_ineq_constraints: Number of inequality constraints

        Raises:
            ValueError: If bounds are inconsistent or dimensions don't match
        """
        # Validate inputs
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError("bounds must be a tuple of (lower_bounds, upper_bounds)")

        lower_bounds, upper_bounds = bounds

        if len(lower_bounds) != dim or len(upper_bounds) != dim:
            raise ValueError(
                f"Bounds dimensions ({len(lower_bounds)}, {len(upper_bounds)}) "
                f"must match problem dimension ({dim})"
            )

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError(
                "Lower bounds must be strictly less than upper bounds in all dimensions"
            )

        if optimum_location is not None and len(optimum_location) != dim:
            raise ValueError(
                f"Optimum location dimension ({len(optimum_location)}) "
                f"must match problem dimension ({dim})"
            )

        # Store metadata
        self.name = name
        self.dim = dim
        self.bounds = (
            np.array(lower_bounds, dtype=np.float64),
            np.array(upper_bounds, dtype=np.float64),
        )
        self.optimum_value = optimum_value
        self.optimum_location = (
            np.array(optimum_location, dtype=np.float64)
            if optimum_location is not None
            else None
        )
        self.num_eq_constraints = num_eq_constraints
        self.num_ineq_constraints = num_ineq_constraints

    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function at point x.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Objective function value (scalar)

        Raises:
            ValueError: If x has incorrect dimensions
        """
        pass

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate equality constraints at point x.

        Equality constraints are of the form h(x) = 0.
        A solution is feasible if all |h_i(x)| ≈ 0.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Array of equality constraint values of shape (num_eq_constraints,)
            Returns empty array if no equality constraints exist.

        Note:
            Override this method in subclasses that have equality constraints.
        """
        return np.array([])

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate inequality constraints at point x.

        Inequality constraints are of the form g(x) ≤ 0.
        A solution is feasible if all g_i(x) ≤ 0.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Array of inequality constraint values of shape (num_ineq_constraints,)
            Returns empty array if no inequality constraints exist.

        Note:
            Override this method in subclasses that have inequality constraints.
        """
        return np.array([])

    def is_feasible(self, x: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Check if a solution is feasible.

        A solution is feasible if:
        1. It satisfies all bound constraints
        2. All equality constraints are satisfied within tolerance
        3. All inequality constraints are satisfied

        Args:
            x: Solution vector of shape (dim,)
            tol: Tolerance for equality constraints

        Returns:
            True if solution is feasible, False otherwise
        """
        # Check bounds
        lower, upper = self.bounds
        if np.any(x < lower) or np.any(x > upper):
            return False

        # Check equality constraints
        eq_violations = self.constraint_eq(x)
        if len(eq_violations) > 0 and np.any(np.abs(eq_violations) > tol):
            return False

        # Check inequality constraints
        ineq_violations = self.constraint_ineq(x)
        if len(ineq_violations) > 0 and np.any(ineq_violations > tol):
            return False

        return True

    def compute_violation(self, x: np.ndarray) -> float:
        """
        Compute total constraint violation for a solution.

        Total violation is computed as:
        - Bound violations: sum of distances outside bounds
        - Equality violations: sum of absolute values
        - Inequality violations: sum of positive parts

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Total constraint violation (0.0 if feasible)
        """
        total_violation = 0.0

        # Bound violations
        lower, upper = self.bounds
        total_violation += np.sum(np.maximum(0.0, lower - x))  # Lower bound violations
        total_violation += np.sum(np.maximum(0.0, x - upper))  # Upper bound violations

        # Equality constraint violations
        eq_violations = self.constraint_eq(x)
        if len(eq_violations) > 0:
            total_violation += np.sum(np.abs(eq_violations))

        # Inequality constraint violations
        ineq_violations = self.constraint_ineq(x)
        if len(ineq_violations) > 0:
            total_violation += np.sum(np.maximum(0.0, ineq_violations))

        return total_violation

    def __str__(self) -> str:
        """String representation of the problem."""
        return (
            f"{self.name} (dim={self.dim}, "
            f"eq_constraints={self.num_eq_constraints}, "
            f"ineq_constraints={self.num_ineq_constraints})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"OptimizationProblem(name='{self.name}', dim={self.dim}, "
            f"bounds={self.bounds}, optimum_value={self.optimum_value})"
        )
