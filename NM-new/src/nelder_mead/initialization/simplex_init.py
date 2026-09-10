"""
Simplex initialization strategies for Nelder-Mead algorithms.

This module provides various methods for constructing initial simplices,
including uniform random, Gaussian, and structured geometric approaches.

References:
    - Spendley, W., Hext, G. R., & Himsworth, F. R. (1962). Sequential application
      of simplex designs in optimisation and evolutionary operation.
      Technometrics, 4(4), 441-461.
    - Lagarias, J. C., Reeds, J. A., Wright, M. H., & Wright, P. E. (1998).
      Convergence properties of the Nelder-Mead simplex method in low dimensions.
      SIAM Journal on Optimization, 9(1), 112-147.
    - Gao, F., & Han, L. (2012). Implementing the Nelder-Mead simplex algorithm with
      adaptive parameters. Computational Optimization and Applications, 51(1), 259-277.
"""

import numpy as np
from typing import Optional
from ..core.simplex_operations import enforce_bounds


class SimplexInitializer:
    """
    Factory class for simplex initialization strategies.

    Provides static methods for generating initial simplices using various strategies:
    - Uniform random initialization
    - Gaussian random initialization
    - Spendley regular simplex
    - Pfeffer simplex (Lagarias specification)
    - Adaptive simplex (dimension-dependent step sizes)
    - Minimal positive basis (well-poised, for simplex-gradient methods)
    """

    @staticmethod
    def uniform(
        num_solutions: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dim: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate solutions uniformly within bounds.

        Each solution is sampled independently from a uniform distribution
        over the box constraints [lower_bounds, upper_bounds].

        Args:
            num_solutions: Number of solutions to generate (typically dim + 1)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            dim: Problem dimension
            rng: Random number generator (if None, uses default)

        Returns:
            Array of shape (num_solutions, dim) with solutions within bounds

        Example:
            >>> lower = np.array([0.0, 0.0])
            >>> upper = np.array([1.0, 1.0])
            >>> solutions = SimplexInitializer.uniform(3, lower, upper, 2)
            >>> solutions.shape
            (3, 2)
            >>> np.all((solutions >= lower) & (solutions <= upper))
            True
        """
        if rng is None:
            rng = np.random.default_rng()

        # Generate uniform random solutions
        solutions = np.zeros((num_solutions, dim))
        for i in range(num_solutions):
            solutions[i] = lower_bounds + rng.random(dim) * (
                upper_bounds - lower_bounds
            )

        return solutions

    @staticmethod
    def gaussian(
        num_solutions: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dim: int,
        rng: Optional[np.random.Generator] = None,
        center: Optional[np.ndarray] = None,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """
        Generate solutions from a Gaussian distribution.

        Solutions are sampled from N(center, σ²I) and then clipped to bounds.
        If no center is provided, uses a random point within bounds.

        Args:
            num_solutions: Number of solutions to generate (typically dim + 1)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            dim: Problem dimension
            rng: Random number generator (if None, uses default)
            center: Center of Gaussian distribution (if None, random within bounds)
            sigma: Standard deviation of Gaussian distribution

        Returns:
            Array of shape (num_solutions, dim) with solutions within bounds

        Example:
            >>> lower = np.array([0.0, 0.0])
            >>> upper = np.array([1.0, 1.0])
            >>> center = np.array([0.5, 0.5])
            >>> solutions = SimplexInitializer.gaussian(3, lower, upper, 2, center=center)
            >>> solutions.shape
            (3, 2)
            >>> np.all((solutions >= lower) & (solutions <= upper))
            True
        """
        if rng is None:
            rng = np.random.default_rng()

        # If no center provided, use random point within bounds
        if center is None:
            center = lower_bounds + rng.random(dim) * (upper_bounds - lower_bounds)

        # Generate Gaussian random solutions
        solutions = np.zeros((num_solutions, dim))
        for i in range(num_solutions):
            point = center + rng.normal(0, sigma, dim)
            solutions[i] = enforce_bounds(point, lower_bounds, upper_bounds)

        return solutions

    @staticmethod
    def spendley_simplex(
        num_solutions: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dim: int,
        x0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Construct a regular simplex using Spendley's method.

        Creates a simplex where all edges from the first vertex to other vertices
        have equal length. This is a geometrically regular simplex.

        The construction uses:
        - p = (1/(n√2)) * (√(n+1) - 1)
        - q = (1/(n√2)) * (√(n+1) + n - 1)

        Args:
            num_solutions: Number of solutions (must be dim + 1)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            dim: Problem dimension
            x0: Starting point (if None, uses center of bounds)
            rng: Random number generator (unused, for interface consistency)

        Returns:
            Array of shape (num_solutions, dim) forming a regular simplex

        Raises:
            ValueError: If num_solutions != dim + 1

        Example:
            >>> lower = np.array([0.0, 0.0])
            >>> upper = np.array([1.0, 1.0])
            >>> simplex = SimplexInitializer.spendley_simplex(3, lower, upper, 2)
            >>> simplex.shape
            (3, 2)
        """
        if num_solutions != dim + 1:
            raise ValueError(
                "Spendley simplex requires num_solutions = dim + 1, "
                f"got {num_solutions} for dimension {dim}"
            )

        # Use center of bounds if no starting point provided
        if x0 is None:
            x0 = (lower_bounds + upper_bounds) / 2.0

        # Compute step sizes for regular simplex
        # p and q are chosen so all edges from x0 have equal length
        p = (1.0 / (dim * np.sqrt(2))) * (np.sqrt(dim + 1) - 1)
        q = (1.0 / (dim * np.sqrt(2))) * (np.sqrt(dim + 1) + dim - 1)

        # Scale by search space size
        scale = np.min(upper_bounds - lower_bounds)
        p *= scale
        q *= scale

        # Construct simplex
        simplex = np.zeros((num_solutions, dim))
        simplex[0] = x0.copy()

        for i in range(1, num_solutions):
            point = x0.copy()
            # Add q to dimension i-1, add p to all other dimensions
            for j in range(dim):
                if j == i - 1:
                    point[j] += q
                else:
                    point[j] += p

            # Enforce bounds
            simplex[i] = enforce_bounds(point, lower_bounds, upper_bounds)

        return simplex

    @staticmethod
    def pfeffer_simplex(
        num_solutions: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dim: int,
        x0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Construct a simplex using Pfeffer's method (Lagarias 1995).

        This method constructs a simplex by adding scaled unit vectors to the
        starting point. The scaling is 0.05 for non-zero coordinates and 0.00025
        for zero coordinates.

        Args:
            num_solutions: Number of solutions (must be dim + 1)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            dim: Problem dimension
            x0: Starting point (if None, uses center of bounds)
            rng: Random number generator (unused, for interface consistency)

        Returns:
            Array of shape (num_solutions, dim) forming Pfeffer simplex

        Raises:
            ValueError: If num_solutions != dim + 1

        Example:
            >>> lower = np.array([0.0, 0.0])
            >>> upper = np.array([1.0, 1.0])
            >>> simplex = SimplexInitializer.pfeffer_simplex(3, lower, upper, 2)
            >>> simplex.shape
            (3, 2)
        """
        if num_solutions != dim + 1:
            raise ValueError(
                "Pfeffer simplex requires num_solutions = dim + 1, "
                f"got {num_solutions} for dimension {dim}"
            )

        # Use center of bounds if no starting point provided
        if x0 is None:
            x0 = (lower_bounds + upper_bounds) / 2.0

        # Construct simplex
        simplex = np.zeros((num_solutions, dim))
        simplex[0] = x0.copy()

        for i in range(1, num_solutions):
            point = x0.copy()
            # Add scaled unit vector in dimension i-1
            if x0[i - 1] != 0:
                point[i - 1] = x0[i - 1] * (1.0 + 0.05)
            else:
                point[i - 1] = 0.00025

            # Enforce bounds
            simplex[i] = enforce_bounds(point, lower_bounds, upper_bounds)

        return simplex

    @staticmethod
    def adaptive_simplex(
        num_solutions: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dim: int,
        x0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Construct a simplex with dimension-dependent step sizes (Gao & Han 2012).

        Uses adaptive step size: σ = min(max(||x₀||∞, 1), 10)
        This scales the simplex based on the magnitude of the starting point.

        Args:
            num_solutions: Number of solutions (must be dim + 1)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            dim: Problem dimension
            x0: Starting point (if None, uses center of bounds)
            rng: Random number generator (unused, for interface consistency)

        Returns:
            Array of shape (num_solutions, dim) with adaptive step sizes

        Raises:
            ValueError: If num_solutions != dim + 1

        Example:
            >>> lower = np.array([0.0, 0.0])
            >>> upper = np.array([1.0, 1.0])
            >>> simplex = SimplexInitializer.adaptive_simplex(3, lower, upper, 2)
            >>> simplex.shape
            (3, 2)
        """
        if num_solutions != dim + 1:
            raise ValueError(
                "Adaptive simplex requires num_solutions = dim + 1, "
                f"got {num_solutions} for dimension {dim}"
            )

        # Use center of bounds if no starting point provided
        if x0 is None:
            x0 = (lower_bounds + upper_bounds) / 2.0

        # Compute adaptive step size based on starting point magnitude
        sigma = min(max(np.linalg.norm(x0, ord=np.inf), 1.0), 10.0)

        # Construct simplex
        simplex = np.zeros((num_solutions, dim))
        simplex[0] = x0.copy()

        for i in range(1, num_solutions):
            point = x0.copy()
            # Add sigma in dimension i-1
            point[i - 1] += sigma

            # Enforce bounds
            simplex[i] = enforce_bounds(point, lower_bounds, upper_bounds)

        return simplex

    @staticmethod
    def minimal_positive_basis(
        num_solutions: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dim: int,
        x0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Construct a simplex from a minimal positive spanning basis.

        Formula: W = Z B⁻,   B⁻ = [e₁ e₂ ... eₙ  -Σᵢ eᵢ] ∈ ℝ^(n×(n+1))

        where Z = diag(z⁰) and z⁰ is sampled uniformly from the search space
        (or supplied as x0). B⁻ is the minimal positive spanning set of ℝⁿ,
        so W has full row rank and the resulting sample set is well poised —
        which is what makes the simplex gradient computed from it a meaningful
        approximation of the true gradient.

        The n+1 vertices are z⁰ᵢeᵢ for i = 1..n, plus -z⁰.

        Note:
            This construction assumes a search box containing the origin (the
            usual [-b, b]ⁿ convention). On a box that excludes the origin,
            clipping vertices back into bounds can collapse the simplex; callers
            should check `simplex_volume` is positive in that case.

        Args:
            num_solutions: Number of solutions (must be dim + 1)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            dim: Problem dimension
            x0: Diagonal z⁰ of Z (if None, sampled uniformly within bounds)
            rng: Random number generator (if None, uses default)

        Returns:
            Array of shape (num_solutions, dim) forming a positive-basis simplex

        Raises:
            ValueError: If num_solutions != dim + 1

        Example:
            >>> lower = np.array([-1.0, -1.0])
            >>> upper = np.array([1.0, 1.0])
            >>> x0 = np.array([1.0, 1.0])
            >>> SimplexInitializer.minimal_positive_basis(3, lower, upper, 2, x0=x0)
            array([[ 1.,  0.],
                   [ 0.,  1.],
                   [-1., -1.]])
        """
        if num_solutions != dim + 1:
            raise ValueError(
                "Minimal positive basis requires num_solutions = dim + 1, "
                f"got {num_solutions} for dimension {dim}"
            )

        if rng is None:
            rng = np.random.default_rng()

        # z⁰ sampled uniformly within bounds unless supplied
        if x0 is None:
            z0 = lower_bounds + rng.random(dim) * (upper_bounds - lower_bounds)
        else:
            z0 = np.asarray(x0, dtype=float).copy()

        # A zero entry in z⁰ makes Z singular and collapses the simplex.
        # Substitute the smallest non-degenerate magnitude available.
        degenerate = z0 == 0.0
        if np.any(degenerate):
            span = upper_bounds - lower_bounds
            z0[degenerate] = 1e-4 * span[degenerate]

        # B⁻ = [e₁ ... eₙ  -Σᵢeᵢ], the minimal positive spanning set
        b_minus = np.hstack([np.eye(dim), -np.ones((dim, 1))])

        # W = Z B⁻; columns are the vertices, so transpose to (n+1, n)
        simplex = (np.diag(z0) @ b_minus).T

        # Enforce bounds, matching the convention of the other initializers
        for i in range(num_solutions):
            simplex[i] = enforce_bounds(simplex[i], lower_bounds, upper_bounds)

        return simplex

    @staticmethod
    def initialize(
        method: str,
        num_solutions: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        dim: int,
        x0: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Initialize simplex using specified method.

        This is a convenience method that dispatches to the appropriate
        initialization strategy based on the method name.

        Args:
            method: Initialization method name
                   ('uniform', 'gaussian', 'spendleySimplex', 'pfefferSimplex',
                    'adaptiveSimplex', 'minimalPositiveBasis')
            num_solutions: Number of solutions to generate
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            dim: Problem dimension
            x0: Starting point for structured methods (optional)
            rng: Random number generator (optional)

        Returns:
            Array of shape (num_solutions, dim) with initialized solutions

        Raises:
            ValueError: If method is not recognized

        Example:
            >>> lower = np.array([0.0, 0.0])
            >>> upper = np.array([1.0, 1.0])
            >>> simplex = SimplexInitializer.initialize('uniform', 3, lower, upper, 2)
            >>> simplex.shape
            (3, 2)
        """
        method_map = {
            "uniform": SimplexInitializer.uniform,
            "gaussian": SimplexInitializer.gaussian,
            "spendleySimplex": SimplexInitializer.spendley_simplex,
            "pfefferSimplex": SimplexInitializer.pfeffer_simplex,
            "adaptiveSimplex": SimplexInitializer.adaptive_simplex,
            "minimalPositiveBasis": SimplexInitializer.minimal_positive_basis,
        }

        if method not in method_map:
            valid_methods = ", ".join(method_map.keys())
            raise ValueError(
                f"'{method}' is not a valid initialization method. "
                f"Available: {valid_methods}"
            )

        init_func = method_map[method]

        # Call appropriate initialization function
        if method in ["uniform", "gaussian"]:
            return init_func(num_solutions, lower_bounds, upper_bounds, dim, rng)
        else:
            # Structured methods (Spendley, Pfeffer, Adaptive)
            return init_func(num_solutions, lower_bounds, upper_bounds, dim, x0, rng)
