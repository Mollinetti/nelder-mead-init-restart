"""
Core mathematical operations for Nelder-Mead algorithms.

This module contains pure functions for simplex geometric transformations,
stopping criteria, and restart strategies.
"""

from .simplex_operations import (
    compute_centroid,
    reflect,
    expand,
    contract_outside,
    contract_inside,
    shrink,
    enforce_bounds,
    edge_matrix,
    oriented_length,
    simplex_volume,
    simplex_gradient,
    merge_simplices,
)

from .stopping_criteria import (
    check_oriented_length,
    check_std_dev,
    check_small_simplex,
    check_flat_simplex,
    check_degenerate_simplex,
    check_fminsearch_fun,
    check_fminsearch_x,
)

from .restart_strategies import (
    RestartStrategy,
    UniformRestart,
    GaussianRestart,
    GaussianBestRestart,
)

__all__ = [
    # Simplex operations
    "compute_centroid",
    "reflect",
    "expand",
    "contract_outside",
    "contract_inside",
    "shrink",
    "enforce_bounds",
    "edge_matrix",
    "oriented_length",
    "simplex_volume",
    "simplex_gradient",
    "merge_simplices",
    # Stopping criteria
    "check_oriented_length",
    "check_std_dev",
    "check_small_simplex",
    "check_flat_simplex",
    "check_degenerate_simplex",
    "check_fminsearch_fun",
    "check_fminsearch_x",
    # Restart strategies
    "RestartStrategy",
    "UniformRestart",
    "GaussianRestart",
    "GaussianBestRestart",
]
