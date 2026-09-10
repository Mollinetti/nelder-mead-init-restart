#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Nelder-Mead Algorithm Implementation

This module implements the Adaptive Nelder-Mead algorithm with dimension-dependent
parameters as described by Gao & Han (2012). The adaptive variant adjusts the
simplex operation coefficients based on the problem dimension to improve
performance across different problem sizes.

The key difference from the classic Nelder-Mead algorithm is that the reflection,
expansion, contraction, and shrink coefficients are computed as functions of
the problem dimension n, rather than using fixed values.

References:
    - Gao, F., & Han, L. (2012). Implementing the Nelder-Mead simplex algorithm
      with adaptive parameters. Computational Optimization and Applications,
      51(1), 259-277.
"""

from typing import Callable, Optional, List
import numpy as np

from .nelder_mead import NelderMead
from ..constraints.barrier_base import Barrier


class AdaptiveNelderMead(NelderMead):
    """
    Adaptive Nelder-Mead algorithm with dimension-dependent parameters.

    This class inherits from the classic NelderMead implementation and overrides
    the simplex operation coefficients with dimension-dependent formulas:

    - Expansion coefficient: δₑ = 1 + 2/n
    - Outside contraction coefficient: δₒc = 0.75 - 1/(2n)
    - Inside contraction coefficient: δᵢc = -(0.75 - 1/(2n))
    - Shrink coefficient: γₛ = 1 - 1/n

    where n is the problem dimension.

    The reflection coefficient remains at the standard value of 1.0.

    These adaptive parameters have been shown to improve convergence properties
    across different problem dimensions compared to the fixed classical parameters.

    Attributes:
        All attributes inherited from NelderMead, with coefficients overridden
        based on dimension.
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
        restart_strategy: str = "gaussian_best",
        stopping_criteria: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the Adaptive Nelder-Mead algorithm.

        The simplex operation coefficients (delta_e, delta_oc, delta_ic, gamma_s)
        are automatically computed based on the problem dimension and cannot be
        manually specified. The reflection coefficient (delta_r) remains at 1.0.

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
            restart_strategy: Name of restart strategy (default: "gaussian_best")
            stopping_criteria: List of stopping criterion names (default: ["fminsearch_fun", "fminsearch_x"])

        Note:
            The delta_r, delta_e, delta_oc, delta_ic, and gamma_s parameters
            are NOT accepted as arguments since they are computed automatically
            based on the problem dimension.

        Validates: Requirements 6.1, 6.2, 6.3, 6.4
        """
        # Initialize parent class with default coefficients
        # (they will be overridden immediately after)
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
            delta_r=1.0,  # Reflection coefficient remains standard
            delta_e=2.0,  # Will be overridden
            delta_oc=0.5,  # Will be overridden
            delta_ic=-0.5,  # Will be overridden
            gamma_s=0.5,  # Will be overridden
            restart_strategy=restart_strategy,
            stopping_criteria=stopping_criteria,
            **kwargs,
        )

        # Override coefficients with dimension-dependent formulas
        # These formulas were derived by Gao & Han (2012) through theoretical analysis
        # and empirical testing across various problem dimensions
        n = self.dim

        # Requirement 6.1: Expansion coefficient = 1 + (2/n)
        # As dimension increases, expansion becomes more conservative
        # For n=2: δₑ = 2.0 (same as classical), for n=10: δₑ = 1.2
        # Rationale: In high dimensions, aggressive expansion is less effective
        self.delta_e = 1.0 + (2.0 / n)

        # Requirement 6.2: Outside contraction coefficient = 0.75 - (1/(2n))
        # As dimension increases, contraction becomes more aggressive
        # For n=2: δₒc = 0.5 (same as classical), for n=10: δₒc = 0.7
        # Rationale: In high dimensions, we need stronger contraction to reduce simplex size
        self.delta_oc = 0.75 - (1.0 / (2.0 * n))

        # Requirement 6.3: Inside contraction coefficient = -(0.75 - (1/(2n)))
        # This is the negative of the outside contraction coefficient
        # For n=2: δᵢc = -0.5 (same as classical), for n=10: δᵢc = -0.7
        # Rationale: Inside contraction should match outside contraction in magnitude
        self.delta_ic = -(0.75 - (1.0 / (2.0 * n)))

        # Requirement 6.4: Shrink coefficient = 1 - (1/n)
        # As dimension increases, shrink becomes less aggressive
        # For n=2: γₛ = 0.5 (same as classical), for n=10: γₛ = 0.9
        # Rationale: In high dimensions, shrinking too aggressively can cause premature
        # convergence. A larger γₛ preserves more of the simplex structure.
        self.gamma_s = 1.0 - (1.0 / n)
