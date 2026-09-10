"""
Unconstrained Benchmark Problems

This module implements standard unconstrained optimization test problems
used for benchmarking and validating optimization algorithms.

References:
    - Moré, J. J., Garbow, B. S., & Hillstrom, K. E. (1981). Testing unconstrained
      optimization software. ACM Transactions on Mathematical Software, 7(1), 17-41.
    - Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions
      for global optimization problems. International Journal of Mathematical
      Modelling and Numerical Optimisation, 4(2), 150-194.
"""

import numpy as np
from nelder_mead.benchmarks.problem_suite import OptimizationProblem


class Sphere(OptimizationProblem):
    """
    Sphere function (also known as De Jong's function 1).

    A simple unimodal function with a single global minimum at the origin.
    Often used as a basic test for optimization algorithms.

    Mathematical formulation:
        f(x) = Σ(x_i^2) for i=1 to n

    Properties:
        - Unimodal (single minimum)
        - Convex
        - Separable
        - Differentiable
        - Global minimum: f(0, 0, ..., 0) = 0

    Args:
        dim: Problem dimensionality (default: 10)
        bounds_range: Search space range [-bounds_range, bounds_range] (default: 100.0)

    Example:
        >>> problem = Sphere(dim=5)
        >>> x = np.zeros(5)
        >>> problem.objective(x)
        0.0
    """

    def __init__(self, dim: int = 10, bounds_range: float = 100.0):
        super().__init__(
            name="Sphere",
            dim=dim,
            bounds=(np.full(dim, -bounds_range), np.full(dim, bounds_range)),
            optimum_value=0.0,
            optimum_location=np.zeros(dim),
        )

    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate the Sphere function.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Objective function value
        """
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")
        return np.sum(x**2)


class Rosenbrock(OptimizationProblem):
    """
    Rosenbrock function (also known as Rosenbrock's valley or banana function).

    A classic test function with a narrow, parabolic valley. The global minimum
    lies in a long, narrow valley that is easy to find but difficult to converge to.

    Mathematical formulation:
        f(x) = Σ[100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2] for i=1 to n-1

    Properties:
        - Unimodal (single minimum)
        - Non-convex
        - Non-separable
        - Differentiable
        - Global minimum: f(1, 1, ..., 1) = 0
        - Narrow valley makes convergence challenging

    Args:
        dim: Problem dimensionality (default: 10)
        bounds_range: Search space range [-bounds_range, bounds_range] (default: 30.0)

    Example:
        >>> problem = Rosenbrock(dim=5)
        >>> x = np.ones(5)
        >>> problem.objective(x)
        0.0
    """

    def __init__(self, dim: int = 10, bounds_range: float = 30.0):
        super().__init__(
            name="Rosenbrock",
            dim=dim,
            bounds=(np.full(dim, -bounds_range), np.full(dim, bounds_range)),
            optimum_value=0.0,
            optimum_location=np.ones(dim),
        )

    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate the Rosenbrock function.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Objective function value
        """
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        # Sum of 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class Rastrigin(OptimizationProblem):
    """
    Rastrigin function.

    A highly multimodal function with many local minima arranged in a regular pattern.
    The function is based on the Sphere function with added cosine modulation to
    create local minima.

    Mathematical formulation:
        f(x) = 10n + Σ[x_i^2 - 10cos(2πx_i)] for i=1 to n

    Properties:
        - Highly multimodal (many local minima)
        - Non-convex
        - Separable
        - Differentiable
        - Global minimum: f(0, 0, ..., 0) = 0
        - Regular pattern of local minima

    Args:
        dim: Problem dimensionality (default: 10)
        bounds_range: Search space range [-bounds_range, bounds_range] (default: 5.12)

    Example:
        >>> problem = Rastrigin(dim=5)
        >>> x = np.zeros(5)
        >>> problem.objective(x)
        0.0
    """

    def __init__(self, dim: int = 10, bounds_range: float = 5.12):
        super().__init__(
            name="Rastrigin",
            dim=dim,
            bounds=(np.full(dim, -bounds_range), np.full(dim, bounds_range)),
            optimum_value=0.0,
            optimum_location=np.zeros(dim),
        )

    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate the Rastrigin function.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Objective function value
        """
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        A = 10.0
        return A * self.dim + np.sum(x**2 - A * np.cos(2 * np.pi * x))


class Ackley(OptimizationProblem):
    """
    Ackley function.

    A multimodal function with many local minima, characterized by a nearly flat
    outer region and a large hole at the center. The function is widely used for
    testing optimization algorithms.

    Mathematical formulation:
        f(x) = -a*exp(-b*sqrt(1/n * Σx_i^2)) - exp(1/n * Σcos(c*x_i)) + a + exp(1)
        where a=20, b=0.2, c=2π

    Properties:
        - Multimodal (many local minima)
        - Non-convex
        - Non-separable
        - Differentiable
        - Global minimum: f(0, 0, ..., 0) = 0
        - Nearly flat outer region with central hole

    Args:
        dim: Problem dimensionality (default: 10)
        bounds_range: Search space range [-bounds_range, bounds_range] (default: 32.768)

    Example:
        >>> problem = Ackley(dim=5)
        >>> x = np.zeros(5)
        >>> abs(problem.objective(x)) < 1e-10
        True
    """

    def __init__(self, dim: int = 10, bounds_range: float = 32.768):
        super().__init__(
            name="Ackley",
            dim=dim,
            bounds=(np.full(dim, -bounds_range), np.full(dim, bounds_range)),
            optimum_value=0.0,
            optimum_location=np.zeros(dim),
        )
        # Standard Ackley parameters
        self.a = 20.0
        self.b = 0.2
        self.c = 2.0 * np.pi

    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate the Ackley function.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Objective function value
        """
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / self.dim))
        term2 = -np.exp(sum_cos / self.dim)

        return term1 + term2 + self.a + np.e


class Griewank(OptimizationProblem):
    """
    Griewank function.

    A multimodal function with many widespread local minima. The function has
    a product term that introduces interdependence between variables, making
    it more challenging than separable functions.

    Mathematical formulation:
        f(x) = 1 + (1/4000)*Σx_i^2 - Π[cos(x_i/sqrt(i))] for i=1 to n

    Properties:
        - Multimodal (many local minima)
        - Non-convex
        - Non-separable (due to product term)
        - Differentiable
        - Global minimum: f(0, 0, ..., 0) = 0
        - Local minima become less pronounced as dimension increases

    Args:
        dim: Problem dimensionality (default: 10)
        bounds_range: Search space range [-bounds_range, bounds_range] (default: 600.0)

    Example:
        >>> problem = Griewank(dim=5)
        >>> x = np.zeros(5)
        >>> abs(problem.objective(x)) < 1e-10
        True
    """

    def __init__(self, dim: int = 10, bounds_range: float = 600.0):
        super().__init__(
            name="Griewank",
            dim=dim,
            bounds=(np.full(dim, -bounds_range), np.full(dim, bounds_range)),
            optimum_value=0.0,
            optimum_location=np.zeros(dim),
        )

    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate the Griewank function.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Objective function value
        """
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        sum_term = np.sum(x**2) / 4000.0

        # Product term: Π[cos(x_i/sqrt(i+1))] where i is 0-indexed
        indices = np.arange(1, self.dim + 1)
        prod_term = np.prod(np.cos(x / np.sqrt(indices)))

        return 1.0 + sum_term - prod_term


class Schwefel(OptimizationProblem):
    """
    Schwefel function.

    A deceptive multimodal function where the global minimum is geometrically
    distant from the next best local minima. This makes it challenging for
    optimization algorithms that can get trapped in local optima far from
    the global optimum.

    Mathematical formulation:
        f(x) = 418.9829n - Σ[x_i * sin(sqrt(|x_i|))] for i=1 to n

    Properties:
        - Highly multimodal (many local minima)
        - Non-convex
        - Separable
        - Differentiable (except at x_i = 0)
        - Global minimum: f(420.9687, ..., 420.9687) ≈ 0
        - Deceptive: global minimum far from next best local minima

    Args:
        dim: Problem dimensionality (default: 10)
        bounds_range: Search space range [-bounds_range, bounds_range] (default: 500.0)

    Example:
        >>> problem = Schwefel(dim=5)
        >>> x = np.full(5, 420.9687)
        >>> abs(problem.objective(x)) < 1.0
        True
    """

    def __init__(self, dim: int = 10, bounds_range: float = 500.0):
        super().__init__(
            name="Schwefel",
            dim=dim,
            bounds=(np.full(dim, -bounds_range), np.full(dim, bounds_range)),
            optimum_value=0.0,
            optimum_location=np.full(dim, 420.9687),
        )

    def objective(self, x: np.ndarray) -> float:
        """
        Evaluate the Schwefel function.

        Args:
            x: Solution vector of shape (dim,)

        Returns:
            Objective function value
        """
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return 418.9829 * self.dim - np.sum(x * np.sin(np.sqrt(np.abs(x))))
