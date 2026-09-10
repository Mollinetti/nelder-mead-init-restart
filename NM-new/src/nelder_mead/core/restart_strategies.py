"""
Restart strategies for Nelder-Mead algorithms.

This module provides strategies for reinitializing the simplex when convergence
is detected. Restart mechanisms help escape local optima and continue searching.

Each strategy generates new points that are then enforced to be within bounds.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class RestartStrategy(ABC):
    """
    Abstract base class for restart strategies.

    A restart strategy generates new points for reinitializing the simplex
    when a stopping criterion is met.
    """

    def __init__(
        self, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize restart strategy.

        Args:
            seed: Random seed for reproducibility (optional, ignored if rng is provided)
            rng: Random number generator (optional, if None creates new one from seed)
        """
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    @abstractmethod
    def generate_point(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        best_solution: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate a new point for restart.

        Args:
            dim: Dimensionality of the problem
            bounds: Tuple of (lower_bounds, upper_bounds)
            best_solution: Current best solution (optional, used by some strategies)

        Returns:
            New point as array of shape (dim,)
        """
        pass

    def set_rng(self, rng: np.random.Generator):
        """Update the random number generator."""
        self.rng = rng


class UniformRestart(RestartStrategy):
    """
    Generate restart points uniformly within bounds.

    This strategy samples points uniformly from the box defined by the bounds.
    It provides good exploration of the entire search space.

    Example:
        >>> strategy = UniformRestart(seed=42)
        >>> bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        >>> point = strategy.generate_point(2, bounds)
        >>> np.all((point >= 0.0) & (point <= 1.0))
        True
    """

    def generate_point(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        best_solution: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate point uniformly within bounds.

        Args:
            dim: Dimensionality of the problem
            bounds: Tuple of (lower_bounds, upper_bounds)
            best_solution: Ignored by this strategy

        Returns:
            Point sampled uniformly from [lower, upper]
        """
        lower_bounds, upper_bounds = bounds
        # Use Generator API: random() returns values in [0, 1)
        return lower_bounds + self.rng.random(dim) * (upper_bounds - lower_bounds)


class GaussianRestart(RestartStrategy):
    """
    Generate restart points from a Gaussian distribution centered at origin.

    This strategy samples from N(0, I) and then enforces bounds by clipping.
    It provides exploration with a bias toward the center of the coordinate system.

    Example:
        >>> strategy = GaussianRestart(seed=42)
        >>> bounds = (np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
        >>> point = strategy.generate_point(2, bounds)
        >>> np.all((point >= -5.0) & (point <= 5.0))
        True
    """

    def generate_point(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        best_solution: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate point from Gaussian distribution centered at origin.

        Args:
            dim: Dimensionality of the problem
            bounds: Tuple of (lower_bounds, upper_bounds)
            best_solution: Ignored by this strategy

        Returns:
            Point sampled from N(0, I), clipped to bounds
        """
        lower_bounds, upper_bounds = bounds
        # Sample from standard normal distribution
        point = self.rng.standard_normal(dim)
        # Clip to bounds
        return np.clip(point, lower_bounds, upper_bounds)


class GaussianBestRestart(RestartStrategy):
    """
    Generate restart points from a Gaussian distribution centered at best solution.

    This strategy samples from N(best, I) and then enforces bounds by clipping.
    It provides local exploration around the current best solution while allowing
    for jumps to escape local optima.

    Example:
        >>> strategy = GaussianBestRestart(seed=42)
        >>> bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        >>> best = np.array([5.0, 5.0])
        >>> point = strategy.generate_point(2, bounds, best_solution=best)
        >>> np.all((point >= 0.0) & (point <= 10.0))
        True
    """

    def generate_point(
        self,
        dim: int,
        bounds: Tuple[np.ndarray, np.ndarray],
        best_solution: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate point from Gaussian distribution centered at best solution.

        Args:
            dim: Dimensionality of the problem
            bounds: Tuple of (lower_bounds, upper_bounds)
            best_solution: Current best solution to center distribution around

        Returns:
            Point sampled from N(best, I), clipped to bounds

        Raises:
            ValueError: If best_solution is None
        """
        if best_solution is None:
            raise ValueError(
                "GaussianBestRestart requires best_solution to be provided"
            )

        lower_bounds, upper_bounds = bounds
        # Sample from normal distribution centered at best solution
        point = best_solution + self.rng.standard_normal(dim)
        # Clip to bounds
        return np.clip(point, lower_bounds, upper_bounds)


def create_restart_strategy(
    strategy_name: str,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> RestartStrategy:
    """
    Factory function to create restart strategy by name.

    Args:
        strategy_name: Name of strategy ("uniform", "gaussian", or "gaussian_best")
        seed: Random seed for reproducibility (ignored if rng is provided)
        rng: Random number generator (optional, if None creates new one from seed)

    Returns:
        RestartStrategy instance

    Raises:
        ValueError: If strategy_name is not recognized

    Example:
        >>> strategy = create_restart_strategy("uniform", seed=42)
        >>> isinstance(strategy, UniformRestart)
        True
    """
    strategy_name = strategy_name.lower()

    strategies = {
        "uniform": UniformRestart,
        "gaussian": GaussianRestart,
        "gaussian_best": GaussianBestRestart,
    }

    if strategy_name not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(
            f"'{strategy_name}' is not a valid restart strategy. "
            f"Available strategies: {available}"
        )

    return strategies[strategy_name](seed=seed, rng=rng)
