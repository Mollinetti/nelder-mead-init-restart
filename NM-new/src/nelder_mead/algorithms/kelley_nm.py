#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nelder-Mead with Kelley's Oriented Restart

Kelley (1999) is the standard remedy for the stagnation McKinnon (1998) exposed.
It has two parts:

1. **A sufficient decrease test.** After each iteration the improvement in the best
   vertex is required to satisfy

       f̄_{k+1} - f̄_k  <  -α ‖∇ₛf(Y_k)‖²        (f̄ = mean of simplex values)

   where ∇ₛf is the simplex gradient. Failure of this test is the stagnation flag.

2. **An oriented restart.** On failure the simplex is reinitialized to a *smaller*
   one with orthogonal edges, oriented by an approximate steepest descent step from
   the current best vertex: vertex i becomes y⁰ + βᵢeᵢ with the sign of βᵢ chosen
   opposite to the i-th component of the simplex gradient, so the new simplex spans
   the descent direction.

This is the incumbent that a contraction-level remedy has to beat, so it is
implemented here as a comparator rather than as a contribution.

References:
    - Kelley, C. T. (1999). Detection and remediation of stagnation in the
      Nelder-Mead algorithm using a sufficient decrease condition.
      SIAM Journal on Optimization, 10(1), 43-55.
    - McKinnon, K. I. M. (1998). Convergence of the Nelder-Mead simplex method to a
      nonstationary point. SIAM Journal on Optimization, 9(1), 148-158.
"""

from typing import Callable, List, Optional

import numpy as np

from .nelder_mead import NelderMead
from ..constraints.barrier_base import Barrier
from ..core import simplex_operations as ops


class KelleyNelderMead(NelderMead):
    """
    Nelder-Mead with Kelley's sufficient-decrease test and oriented restart.

    Attributes:
        kelley_alpha (float): Coefficient α in the sufficient decrease test
        num_oriented_restarts (int): Oriented restarts performed
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
        kelley_alpha: float = 1e-4,
        **kwargs,
    ):
        """
        Initialize Nelder-Mead with Kelley's oriented restart.

        Args:
            objective_fn: Function to minimize
            constraint_eq_fn: Equality constraint function
            constraint_ineq_fn: Inequality constraint function
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            max_fes: Maximum number of function evaluations
            num_solutions: Number of solutions in simplex (default: dim+1)
            init_method: Initialization method name
            seed: Random seed for reproducibility
            barrier: Barrier method for constraint handling
            delta_r: Reflection coefficient
            delta_e: Expansion coefficient
            delta_oc: Outside contraction coefficient
            delta_ic: Inside contraction coefficient
            gamma_s: Shrink coefficient
            restart_strategy: Name of restart strategy
            stopping_criteria: List of stopping criterion names
            kelley_alpha: Coefficient α of the sufficient decrease test

        Raises:
            ValueError: If kelley_alpha is not positive
        """
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
            delta_r=delta_r,
            delta_e=delta_e,
            delta_oc=delta_oc,
            delta_ic=delta_ic,
            gamma_s=gamma_s,
            restart_strategy=restart_strategy,
            stopping_criteria=stopping_criteria,
            **kwargs,
        )

        if kelley_alpha <= 0.0:
            raise ValueError(f"kelley_alpha must be positive, got {kelley_alpha}")

        # Kelley p.138: the constructor argument is alpha_0. The alpha actually
        # used in (8.4) is rescaled once from the initial simplex,
        #     alpha = alpha_0 * sigma_+(S^0) / ||D^0 f||,
        # because "if the simplex diameter is much smaller than ||D_k f||, (8.4)
        # could fail on the first iterate".
        self.kelley_alpha_0 = kelley_alpha
        self.kelley_alpha = kelley_alpha
        self._alpha_scaled = False
        self.num_oriented_restarts = 0
        self._prev_mean = None
        self._restarting = False

    def _rescale_alpha(self) -> None:
        """
        Apply Kelley's one-time scaling alpha = alpha_0 * sigma_+(S^0) / ||D^0 f||.

        Without it, (8.4) can fail on the very first iterate whenever the initial
        simplex is small relative to the simplex gradient, which is exactly the
        regime a well-scaled initializer produces.
        """
        if self._alpha_scaled or self.simplex is None:
            return
        self._alpha_scaled = True
        grad_norm = float(
            np.linalg.norm(ops.simplex_gradient(self.simplex, self.fitness_values))
        )
        sigma_plus = ops.oriented_length(self.simplex)
        if grad_norm > 0.0 and sigma_plus > 0.0:
            self.kelley_alpha = self.kelley_alpha_0 * sigma_plus / grad_norm

    def _oriented_restart(self) -> None:
        """
        Rebuild the simplex as a smaller orthogonal one oriented along -∇ₛf.

        The best vertex is kept. Each other vertex becomes y⁰ + βᵢeᵢ, with |βᵢ| set
        to half the current oriented length and the sign opposite to the i-th
        component of the simplex gradient, so the new simplex contains an
        approximate steepest descent step.
        """
        self.num_oriented_restarts += 1

        self.simplex = ops.oriented_restart(
            self.simplex, self.fitness_values, self.lower_bounds, self.upper_bounds
        )
        (
            self.fitness_values,
            self.eq_violations_list,
            self.ineq_violations_list,
            self.total_violations,
        ) = self.evaluate_population(self.simplex)
        self._sort_simplex()

    def _sort_simplex(self) -> None:
        """
        Sort the simplex, then apply Kelley's sufficient decrease test.

        The test compares the improvement of the best vertex against
        -α‖∇ₛf‖²; failing it triggers an oriented restart.
        """
        super()._sort_simplex()

        # _oriented_restart re-sorts; do not re-enter the test from inside it
        if self._restarting:
            return

        if self.fitness_values is None or self._prev_mean is None:
            self._prev_mean = float(np.mean(self.fitness_values))
            self._rescale_alpha()
            return

        gradient = ops.simplex_gradient(self.simplex, self.fitness_values)
        required = self.kelley_alpha * float(np.dot(gradient, gradient))

        # The f^k of (8.4) is the *mean* simplex value, not the best vertex.
        # Kelley p.136 is explicit: "while a Nelder-Mead iterate may not result in
        # a reduction in the best function value, the average value ... will be
        # reduced", and Assumption 8.1.1 then requires f-bar^{k+1} < f-bar^k.
        # Reading f^k as the best vertex breaks the detector twice over: the best
        # is unchanged on a large share of iterations, which fails (8.4)
        # trivially and fires the restart on healthy progress; and on McKinnon's
        # own stagnation family the best vertex never moves at all, so the
        # detector can never fire on the failure it was designed for.
        current_mean = float(np.mean(self.fitness_values))
        improvement = current_mean - self._prev_mean

        # Kelley p.139: restart "when (8.4) fails but f^{k+1} - f^k < 0".
        stagnating = (
            improvement >= -required and improvement < 0.0 and required > 0.0
        )
        self._prev_mean = current_mean

        if stagnating and not self.should_terminate():
            self._restarting = True
            try:
                self._oriented_restart()
            finally:
                self._restarting = False
            self._prev_mean = float(np.mean(self.fitness_values))
