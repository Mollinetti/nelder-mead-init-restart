"""
Stopping criteria for detecting convergence in Nelder-Mead algorithms.

This module provides functions that detect various convergence conditions.
Each function returns a boolean indicating whether the criterion is met.

References:
    - Lagarias, J. C., et al. (1998). Convergence properties of the Nelder-Mead
      simplex method in low dimensions. SIAM Journal on Optimization, 9(1), 112-147.
"""

import numpy as np
from typing import Tuple

from .simplex_operations import simplex_conditioning


def check_oriented_length(simplex: np.ndarray, epsilon: float = 1e-4) -> bool:
    """
    Check oriented length criterion for convergence.

    Measures maximum edge length normalized by the position of the first vertex.
    Formula: (1/max(1, ||v₀||)) * max(||vᵢ - v₀||) ≤ ε

    This criterion detects when the simplex has become very small relative to its
    position in the search space. The normalization by max(1, ||v₀||) ensures that
    the criterion works well both near the origin and far from it. Without this
    normalization, a simplex far from the origin might appear large even when it
    has effectively converged.

    Reference: Lagarias et al. (1998), Section 3.2

    Args:
        simplex: Array of shape (n+1, n) containing simplex vertices
        epsilon: Tolerance threshold (default: 1e-4)

    Returns:
        True if criterion is met (simplex has converged), False otherwise

    Example:
        >>> simplex = np.array([[1.0, 1.0], [1.001, 1.0], [1.0, 1.001]])
        >>> check_oriented_length(simplex, epsilon=1e-2)
        True  # All edges are very small
    """
    v0 = simplex[0]
    v0_norm = np.linalg.norm(v0)  # ||v₀|| - Euclidean norm of first vertex

    # Compute all edge lengths from first vertex: ||vᵢ - v₀|| for i = 1, ..., n
    edge_lengths = np.array(
        [np.linalg.norm(simplex[i] - v0) for i in range(1, len(simplex))]
    )
    max_edge = np.max(edge_lengths)  # Maximum edge length

    # Normalize by max(1, ||v₀||) to handle both near-origin and far-from-origin cases
    normalized_length = max_edge / max(1.0, v0_norm)

    return normalized_length <= epsilon


def check_std_dev(fitness_values: np.ndarray, epsilon: float = 1e-7) -> bool:
    """
    Check standard deviation of fitness values for convergence.

    Measures variance of objective function values across the simplex.
    Formula: (1/(n+1)) * Σ(fᵢ - f̄)² ≤ ε

    This criterion detects when all vertices have nearly identical objective values,
    indicating the simplex has converged to a region where the function is flat or
    the simplex has collapsed. The variance formula is the mean squared deviation
    from the average fitness.

    Reference: Lagarias et al. (1998), Section 3.2

    Args:
        fitness_values: Array of fitness values for all simplex vertices
        epsilon: Tolerance threshold (default: 1e-7)

    Returns:
        True if criterion is met (fitness values are nearly identical), False otherwise

    Example:
        >>> fitness = np.array([1.0, 1.0001, 1.0002])
        >>> check_std_dev(fitness, epsilon=1e-6)
        False  # Variance is too large
    """
    n = len(fitness_values)
    f_mean = np.mean(fitness_values)  # f̄ - mean fitness

    # Compute variance: (1/n) * Σ(fᵢ - f̄)²
    # This is the mean squared deviation from the average
    variance = np.sum((fitness_values - f_mean) ** 2) / n

    return variance <= epsilon


def check_small_simplex(
    simplex: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray], epsilon: float
) -> bool:
    """
    Check if simplex is too small relative to the search space.

    Measures maximum edge length normalized by the bound range in each dimension.

    Args:
        simplex: Array of shape (n+1, n) containing simplex vertices
        bounds: Tuple of (lower_bounds, upper_bounds)
        epsilon: Tolerance threshold

    Returns:
        True if criterion is met (simplex is too small), False otherwise

    Example:
        >>> simplex = np.array([[0.5, 0.5], [0.501, 0.5], [0.5, 0.501]])
        >>> bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        >>> check_small_simplex(simplex, bounds, epsilon=1e-2)
        True  # Simplex is tiny compared to [0, 10] bounds
    """
    lower_bounds, upper_bounds = bounds
    dim = simplex.shape[1]

    # Compute maximum edge length in each dimension, normalized by bound range
    max_normalized_edge = 0.0
    for d in range(dim):
        # Get all coordinates in dimension d
        coords = simplex[:, d]
        # Compute range in this dimension
        coord_range = np.max(coords) - np.min(coords)
        # Normalize by bound range
        bound_range = upper_bounds[d] - lower_bounds[d]
        if bound_range > 0:
            normalized = coord_range / bound_range
            max_normalized_edge = max(max_normalized_edge, normalized)

    return max_normalized_edge < epsilon


def check_flat_simplex(fitness_high: float, fitness_low: float, epsilon: float) -> bool:
    """
    Check if simplex is flat (all fitness values nearly identical).

    Formula: |f_max - f_min| < ε

    Args:
        fitness_high: Maximum fitness value in simplex
        fitness_low: Minimum fitness value in simplex
        epsilon: Tolerance threshold

    Returns:
        True if criterion is met (fitness values are nearly identical), False otherwise

    Example:
        >>> check_flat_simplex(1.0001, 1.0, epsilon=1e-3)
        True  # Difference is less than epsilon
    """
    return abs(fitness_high - fitness_low) < epsilon


def check_degenerate_simplex(
    simplex: np.ndarray, epsilon1: float, epsilon2: float
) -> bool:
    """
    Check whether the simplex has collapsed towards a lower-dimensional subspace.

    This is eq. (9) of Luersen & Le Riche, "Globalized Nelder-Mead method for
    engineering optimization", Computers & Structures 82(23):2251-2260, 2004.
    The simplex is degenerate when either conditioning measure falls below its
    tolerance:

        min_k ‖eᵏ‖ / max_k ‖eᵏ‖ < ε₁      (edges of wildly unequal length)
        |det[e]| / ∏_k ‖eᵏ‖     < ε₂      (edges nearly linearly dependent)

    where eᵏ are the edges radiating from the best vertex. Both quantities are
    translation- and scale-invariant, which is what makes them usable as a
    restart trigger: they say the simplex has the wrong *shape*, independently
    of where it sits or how large it is. Smallness is a separate test
    (`check_small_simplex`), and conflating the two is what Luersen & Le Riche
    report as making their tolerances hard to tune.

    Args:
        simplex: Array of shape (n+1, n) containing simplex vertices
        epsilon1: Tolerance on the edge-length ratio (their εs3)
        epsilon2: Tolerance on the Hadamard ratio (their εs4)

    Returns:
        True if the simplex is degenerate.

    Example:
        >>> # Nearly collinear in 2-D: fine edge lengths, no area
        >>> simplex = np.array([[0., 0.], [1., 0.], [2., 0.001]])
        >>> check_degenerate_simplex(simplex, epsilon1=0.1, epsilon2=0.1)
        True
    """
    edge_ratio, hadamard_ratio = simplex_conditioning(simplex)
    return edge_ratio < epsilon1 or hadamard_ratio < epsilon2

def check_fminsearch_fun(fitness_values: np.ndarray, tol_fun: float = 1e-12) -> bool:
    """
    Check MATLAB fminsearch-style function tolerance criterion.

    Formula: max(|fᵢ - f₀|) ≤ tol_fun for all i
    where f₀ is the fitness of the best vertex

    Args:
        fitness_values: Array of fitness values (assumed sorted, best first)
        tol_fun: Function tolerance (default: 1e-12)

    Returns:
        True if criterion is met, False otherwise

    Example:
        >>> fitness = np.array([1.0, 1.0, 1.0])
        >>> check_fminsearch_fun(fitness, tol_fun=1e-10)
        True  # All values identical
    """
    f_best = fitness_values[0]
    max_diff = np.max(np.abs(fitness_values - f_best))
    return max_diff <= tol_fun


def check_fminsearch_x(simplex: np.ndarray, tol_x: float = 1e-8) -> bool:
    """
    Check MATLAB fminsearch-style position tolerance criterion.

    Formula: max(||vᵢ - v₀||_∞) ≤ tol_x for all i
    where v₀ is the best vertex and ||·||_∞ is the infinity norm

    Args:
        simplex: Array of shape (n+1, n) containing simplex vertices (assumed sorted, best first)
        tol_x: Position tolerance (default: 1e-8)

    Returns:
        True if criterion is met, False otherwise

    Example:
        >>> simplex = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        >>> check_fminsearch_x(simplex, tol_x=1e-6)
        True  # All vertices at same position
    """
    v_best = simplex[0]
    max_dist = 0.0
    for i in range(1, len(simplex)):
        dist = np.linalg.norm(simplex[i] - v_best, ord=np.inf)
        max_dist = max(max_dist, dist)
    return max_dist <= tol_x
