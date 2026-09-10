#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient-Based Nelder-Mead (g-NM) Algorithm Implementation

This module implements the g-NM algorithm: a Nelder-Mead variant in which the
shrink step is replaced by a *polling step* driven by simplex gradients.

g-NM is an instance of the **search-poll framework**: the Nelder-Mead geometric
transformations act as the *search step*, and a simplex-gradient-ordered poll
over a positive spanning set acts as the *poll step*. Framing it this way is what
makes the standard directional direct-search convergence results available,
since every one of them rests on the inequality obtained at an unsuccessful poll,

    ‖∇f(x_k)‖ ≤ (1/κ)·( ρ(α_k)/α_k + L·α_k/2 )

where κ > 0 is the cosine measure of the poll directions. That inequality needs
two properties, and the algorithm is built around them:

1. **The poll directions positively span ℝⁿ.** A poll along a single descent
   direction proves nothing: only a positive spanning set bounds the gradient.
   g-NM polls the minimal positive basis D = [e₁ ... eₙ  -Σᵢeᵢ], which costs
   n+1 evaluations, and uses the simplex gradient to *order* those directions so
   the best-aligned one is tried first — the idea of Custódio and Vicente.

2. **α_k → 0 along unsuccessful polls.** The step size is persistent state, not
   a per-poll constant: it grows by γ on a successful poll and shrinks by τ on
   an unsuccessful one. Once α falls below `alpha_min` the incumbent is declared
   a local minimizer and the simplex restarts.

Two further ingredients:

3. **Well-poised initialization.** The initial simplex is built from a minimal
   positive spanning basis W = Z B⁻ (see
   `SimplexInitializer.minimal_positive_basis`), which has full row rank and
   therefore makes the simplex gradient computed from it meaningful.

4. **A record set R with a merge/restart policy.** R holds previously accepted
   points and supplies the simplex gradient. If polling finds an improvement the
   simplex is merged toward a fresh simplex built from it; if polling exhausts
   the step size, the simplex is restarted from a uniformly random point.

Departures from the thesis text, all configurable:

- **The forcing function.** The thesis states the sufficient decrease condition
  as f(y*) < f(y) - ε "for a small enough ε". A *constant* ε does not support a
  convergence argument. Here ε = ρ(α) = c·α² is a genuine forcing function. Set
  `forcing_c=0.0` to recover simple decrease.

- **The poll set.** The thesis reads P_k = {y_k + α r' : r' ∈ R} with r' a
  single descent direction drawn from R. That is unprovable (one direction) and,
  read as a fan over all of R, costs O(|R|) evaluations per poll. See point 1.

- **A bounded record set.** The thesis lets R grow without bound, making memory
  O(k) and the least-squares solve grow with the iteration count. R is pruned to
  the `record_size` points *nearest the incumbent*, matching the fact that
  Λ-poisedness is defined on a ball around y⁰ — proximity, not recency, is what
  the theory asks for. Pass `record_size=None` for the unbounded original.

- **A scale-aware step size.** The thesis fixes α₀ = 1e-4 in absolute terms,
  unrelated to the simplex being polled: on a large simplex it banks a
  microscopic decrease and leaves the simplex uncontracted, so the search stalls
  (f ≈ 4.6 on a 2-D sphere over [-100, 100]² where classic Nelder-Mead reaches
  0). α is initialized to the oriented length σ⁺ and reset to it whenever the
  simplex is rebuilt. Pass `alpha_init=1e-4` for the thesis value.

Note on monotonicity: a restart at x₀ = U(l, u) can raise the objective value of
the simplex, so the iterate sequence is not monotone. Convergence statements
should therefore be made about the **best-so-far sequence**, which is monotone by
construction. `keep_incumbent_on_restart=True` enforces monotonicity in the code
instead, at the cost of biasing each restart toward the abandoned basin.

References:
    - Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization.
      The Computer Journal, 7(4), 308-313.
    - Custódio, A. L., & Vicente, L. N. (2007). Using sampling and simplex
      derivatives in pattern search methods. SIAM Journal on Optimization,
      17(4), 1236-1254.
    - Kolda, T. G., Lewis, R. M., & Torczon, V. (2003). Optimization by direct
      search: New perspectives on some classical and modern methods.
      SIAM Review, 45(3), 385-482.
    - Kelley, C. T. (1999). Detection and remediation of stagnation in the
      Nelder-Mead algorithm using a sufficient decrease condition.
      SIAM Journal on Optimization, 10(1), 43-55.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .nelder_mead import NelderMead
from ..constraints.barrier_base import Barrier
from ..core import simplex_operations as ops
from ..initialization.simplex_init import SimplexInitializer


class GradientNelderMead(NelderMead):
    """
    Nelder-Mead with a simplex-gradient polling step replacing the shrink step.

    Inherits the full Nelder-Mead transformation loop from :class:`NelderMead`
    and overrides three extension points:

    - :meth:`_on_accept` records every accepted vertex into R
    - :meth:`_shrink_simplex` polls instead of shrinking
    - :meth:`_restart_simplex` polls, then either merges or restarts

    Attributes:
        gamma_m (float): Merge step length γᵐ
        alpha_init (float): Initial polling step size α₀
        tau (float): Line search backtracking factor τ
        forcing_c (float): Coefficient c of the forcing function ρ(α) = c·α²
        max_backtracks (int): Maximum backtracking steps per polling call
        record_size (Optional[int]): Cap on |R| (None for unbounded)
        num_polls (int): Number of polling steps executed
        num_poll_successes (int): Polling steps that found a better point
        num_restarts (int): Restarts triggered by failed polling
        num_merges (int): Merge operations performed
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
        init_method: str = "minimalPositiveBasis",
        seed: int = 101,
        barrier: Optional[Barrier] = None,
        delta_r: float = 1.0,
        delta_e: float = 2.0,
        delta_oc: float = 0.5,
        delta_ic: float = -0.5,
        gamma_s: float = 0.5,
        restart_strategy: str = "gaussian_best",
        stopping_criteria: Optional[List[str]] = None,
        gamma_m: float = 0.8,
        alpha_init: Optional[float] = None,
        tau: float = 0.5,
        gamma_a: float = 2.0,
        alpha_min: float = 1e-10,
        forcing_c: float = 1e-4,
        record_size: Union[int, str, None] = "auto",
        keep_incumbent_on_restart: bool = False,
        **kwargs,
    ):
        """
        Initialize the gradient-based Nelder-Mead algorithm.

        Args:
            objective_fn: Function to minimize, takes np.ndarray and returns float
            constraint_eq_fn: Equality constraint function (returns violations)
            constraint_ineq_fn: Inequality constraint function (returns violations)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            max_fes: Maximum number of function evaluations
            num_solutions: Number of solutions in simplex (default: dim+1)
            init_method: Initialization method (default: "minimalPositiveBasis",
                         the well-poised construction g-NM assumes)
            seed: Random seed for reproducibility
            barrier: Barrier method for constraint handling
            delta_r: Reflection coefficient
            delta_e: Expansion coefficient
            delta_oc: Outside contraction coefficient
            delta_ic: Inside contraction coefficient
            gamma_s: Shrink coefficient (unused; polling replaces shrink)
            restart_strategy: Restart strategy name (inherited; g-NM restarts
                              from a uniform random point via init_method)
            stopping_criteria: List of stopping criterion names
            gamma_m: Merge step length γᵐ in [0, 1] (default: 0.8)
            alpha_init: Step size α₀, used at the start and after every simplex
                        rebuild. None (the default) uses the oriented length σ⁺
                        of the current simplex. Pass a float for a fixed
                        absolute step (the thesis uses 1e-4)
            tau: Contraction factor applied to α after an unsuccessful poll,
                 in (0, 1) (default: 0.5)
            gamma_a: Expansion factor applied to α after a successful poll,
                     at least 1.0 (default: 2.0)
            alpha_min: Step size below which the incumbent is declared a local
                       minimizer and the simplex restarts (default: 1e-10)
            forcing_c: Coefficient of the forcing function ρ(α) = c·α².
                       Use 0.0 for simple decrease (default: 1e-4)
            record_size: Cap on the size of the record set R. "auto" (the
                         default) uses 10(dim+1); an int sets it explicitly and
                         must be at least dim+1; None leaves R unbounded, as in
                         the thesis
            keep_incumbent_on_restart: If True, a restart keeps the best-so-far
                         solution as one vertex of the new simplex, enforcing
                         monotone descent in the code at the cost of biasing the
                         restart toward the abandoned basin. False (the default)
                         follows the thesis; state convergence results on the
                         best-so-far sequence instead

        Raises:
            ValueError: If any parameter is outside its valid range
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

        if not 0.0 <= gamma_m <= 1.0:
            raise ValueError(f"gamma_m must be in [0, 1], got {gamma_m}")
        if alpha_init is not None and alpha_init <= 0.0:
            raise ValueError(
                f"alpha_init must be positive or None, got {alpha_init}"
            )
        if not 0.0 < tau < 1.0:
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if gamma_a < 1.0:
            raise ValueError(f"gamma_a must be at least 1.0, got {gamma_a}")
        if alpha_min <= 0.0:
            raise ValueError(f"alpha_min must be positive, got {alpha_min}")
        if forcing_c < 0.0:
            raise ValueError(f"forcing_c must be non-negative, got {forcing_c}")

        if record_size == "auto":
            record_size = 10 * (self.dim + 1)
        elif record_size is not None and record_size < self.dim + 1:
            raise ValueError(
                f"record_size must be at least dim+1 = {self.dim + 1} for the "
                f"simplex gradient to be determined, got {record_size}"
            )

        self.gamma_m = gamma_m
        self.alpha_init = alpha_init
        self.tau = tau
        self.gamma_a = gamma_a
        self.alpha_min = alpha_min
        self.forcing_c = forcing_c
        self.record_size = record_size
        self.keep_incumbent_on_restart = keep_incumbent_on_restart

        # Record set R: points and their objective values, seeded with W
        self._record: Optional[np.ndarray] = None
        self._record_values: Optional[np.ndarray] = None

        # Persistent poll step size, initialized on first use
        self._alpha: Optional[float] = None

        # Poll directions: the minimal positive basis D = [e₁ ... eₙ  -Σᵢeᵢ],
        # normalized. Fixed for the run, reordered per poll by the gradient.
        basis = np.vstack([np.eye(self.dim), -np.ones(self.dim) / np.sqrt(self.dim)])
        self._poll_basis = basis

        # Diagnostics, so experiments can report how often polling actually fired
        self.num_polls = 0
        self.num_poll_successes = 0
        self.num_restarts = 0
        self.num_merges = 0

    # ------------------------------------------------------------------
    # Record set R
    # ------------------------------------------------------------------

    def evaluate_population(self, population: np.ndarray):
        """
        Evaluate a population, seeding the record set R on the first call.

        The first population evaluated is the initial simplex W, so this is
        where ``R ← W`` happens. Overriding here rather than in
        ``initialize_population`` means R is seeded with objective values
        already attached.

        Args:
            population: Array of shape (num_solutions, dim)

        Returns:
            Same tuple as :meth:`BaseAlgorithm.evaluate_population`
        """
        result = super().evaluate_population(population)

        if self._record is None:
            self._record = np.asarray(population, dtype=float).copy()
            self._record_values = np.asarray(result[0], dtype=float).copy()

        return result

    def _on_accept(self, point: np.ndarray, fitness: float) -> None:
        """
        Record a point accepted by a geometric transformation (``R ← R ∪ y``).

        Args:
            point: The accepted point
            fitness: Objective value of the accepted point
        """
        self._record_point(point, fitness)

    def _record_point(self, point: np.ndarray, fitness: float) -> None:
        """
        Append a point to the record set, enforcing the size cap.

        When capped, R is pruned to the points *nearest the incumbent* rather
        than the most recent ones. Λ-poisedness — the property that makes the
        simplex gradient a good approximation of ∇f — is defined on a ball
        around y⁰, so proximity is the criterion the theory asks for. An old
        nearby sample carries more information about the local model than a
        recent distant one.

        Args:
            point: Point to record
            fitness: Objective value of the point
        """
        if self._record is None:
            self._record = np.asarray([point], dtype=float)
            self._record_values = np.asarray([fitness], dtype=float)
            return

        self._record = np.vstack([self._record, np.asarray(point, dtype=float)])
        self._record_values = np.append(self._record_values, fitness)

        if self.record_size is None or len(self._record) <= self.record_size:
            return

        center = (
            self.best_solution
            if self.best_solution is not None
            else self._record[int(np.argmin(self._record_values))]
        )
        nearest = np.argsort(np.linalg.norm(self._record - center, axis=1))
        keep = np.sort(nearest[: self.record_size])

        self._record = self._record[keep]
        self._record_values = self._record_values[keep]

    def _in_record(self, point: np.ndarray, tol: float = 1e-12) -> bool:
        """
        Test whether a point already belongs to the record set (``y ∈ R``).

        Args:
            point: Candidate point
            tol: Absolute tolerance for the coordinate-wise comparison

        Returns:
            True if the point is already recorded
        """
        if self._record is None or len(self._record) == 0:
            return False

        return bool(
            np.any(np.all(np.abs(self._record - point) <= tol, axis=1))
        )

    def simplex_gradient(self) -> np.ndarray:
        """
        Compute the simplex gradient of f over the record set R.

        R is sorted by objective value (best first) so that the edge matrix
        L(R) = [r¹ - r⁰, ...] and the difference vector δ_f are built around the
        best recorded point, as equations (4.1) and (4.2) require.

        Returns:
            Simplex gradient as array of shape (dim,); zeros if R is too small
        """
        if self._record is None or len(self._record) < 2:
            return np.zeros(self.dim)

        order = np.argsort(self._record_values)
        return ops.simplex_gradient(
            self._record[order], self._record_values[order]
        )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_directions(self) -> np.ndarray:
        """
        Order the poll directions by the simplex gradient, best-aligned first.

        The direction *set* is always the full minimal positive basis
        D = [e₁ ... eₙ  -Σᵢeᵢ]: an unsuccessful poll only bounds ‖∇f‖ when the
        directions positively span ℝⁿ, so no direction may be dropped, however
        unpromising it looks. What the simplex gradient supplies is the *order*
        in which they are tried, so that opportunistic polling usually succeeds
        on the first evaluation. This is the role simplex derivatives play in
        Custódio and Vicente's pattern search.

        When the gradient carries no information the basis is returned in its
        natural order, which costs nothing but a worse expected ordering.

        Returns:
            Array of shape (dim+1, dim) of unit directions
        """
        gradient = self.simplex_gradient()
        grad_norm = np.linalg.norm(gradient)

        if grad_norm == 0.0 or not np.isfinite(grad_norm):
            return self._poll_basis

        # Ascending alignment: most negative dᵀg (steepest descent) goes first
        order = np.argsort(self._poll_basis @ gradient)
        return self._poll_basis[order]

    def _reset_alpha(self) -> None:
        """
        Reset the poll step size to α₀.

        Called at the start and whenever the simplex is rebuilt by a restart or
        a merge, since the new simplex may live at a completely different scale
        and a step size carried over from the old one would be meaningless.
        Within a restart epoch α is never reset, so it decreases monotonically
        across unsuccessful polls, which is what the convergence argument needs.
        """
        if self.alpha_init is not None:
            self._alpha = self.alpha_init
            return

        sigma = ops.oriented_length(self.simplex)
        self._alpha = (
            sigma
            if sigma > 0.0
            # Fully collapsed simplex: fall back to a fraction of the box width
            else 1e-4 * float(np.min(self.upper_bounds - self.lower_bounds))
        )

    def _polling_step(
        self, y_k: np.ndarray, f_k: float
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray, np.ndarray, float]]:
        """
        Poll the positive spanning set at the current step size.

        Evaluates P_k = {y_k + α d : d ∈ D} opportunistically, in the order
        given by :meth:`_poll_directions`, and accepts the first point y*
        satisfying the sufficient decrease condition

            f(y*) < f(y_k) - ρ(α),   ρ(α) = c·α²

        Cost is at most dim+1 evaluations, independent of |R|. The step size is
        updated in place: multiplied by γ on success, by τ on failure. Points
        already in R are skipped, since re-accepting one cannot make progress.

        Note:
            Poll points are clipped to the box, so a candidate on the boundary
            sits closer to y_k than α. The forcing function still uses the
            nominal α, which is conservative — it demands more decrease than the
            realized step earns.

        Args:
            y_k: Current iterate (the best simplex vertex)
            f_k: Objective value at y_k

        Returns:
            Tuple of (point, fitness, eq_violations, ineq_violations,
            total_violation) on success, None if the poll was unsuccessful
        """
        self.num_polls += 1

        if self._alpha is None:
            self._reset_alpha()

        alpha = self._alpha
        rho = self.forcing_c * alpha * alpha

        for direction in self._poll_directions():
            if self.should_terminate():
                return None

            candidate = ops.enforce_bounds(
                y_k + alpha * direction, self.lower_bounds, self.upper_bounds
            )

            if self._in_record(candidate):
                continue

            fitness, eq_vio, ineq_vio, total_vio = self.evaluate(candidate)

            if fitness < f_k - rho:
                # Successful poll: grow the step for the next one
                self._alpha = alpha * self.gamma_a
                self.num_poll_successes += 1
                return candidate, fitness, eq_vio, ineq_vio, total_vio

        # Unsuccessful poll: contract the step, so α → 0 over repeated failures
        self._alpha = alpha * self.tau
        return None

    # ------------------------------------------------------------------
    # Simplex construction
    # ------------------------------------------------------------------

    def _build_simplex(self, x0: Optional[np.ndarray]) -> np.ndarray:
        """
        Build a fresh simplex from the configured initialization heuristic.

        Args:
            x0: Starting point for the heuristic (None to let it choose)

        Returns:
            Array of shape (num_solutions, dim)
        """
        return SimplexInitializer.initialize(
            method=self.init_method,
            num_solutions=self.num_solutions,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            dim=self.dim,
            x0=x0,
            rng=self.rng,
        )

    def _adopt_simplex(self, simplex: np.ndarray) -> None:
        """
        Install a new simplex, clip it to bounds, re-evaluate and sort it.

        Args:
            simplex: Array of shape (num_solutions, dim)
        """
        for i in range(len(simplex)):
            simplex[i] = ops.enforce_bounds(
                simplex[i], self.lower_bounds, self.upper_bounds
            )

        self.simplex = simplex
        (
            self.fitness_values,
            self.eq_violations_list,
            self.ineq_violations_list,
            self.total_violations,
        ) = self.evaluate_population(self.simplex)
        self._sort_simplex()

    def _restart_from_random(self) -> None:
        """
        Restart the simplex from a uniformly random point (``x₀ = U(l, u)``).

        Polling failed, so y_k is treated as a local minimizer and the search
        moves elsewhere. Unless `keep_incumbent_on_restart` is False, the
        best-so-far solution is carried into the new simplex as one vertex:
        the thesis discards it, which lets the objective value of the simplex
        increase across a restart and breaks the monotone descent that a
        convergence argument relies on.

        The best vertex of the restarted simplex is added to R, so the
        abandoned basin is not re-explored by a later polling step.
        """
        self.num_restarts += 1

        x0 = self.lower_bounds + self.rng.random(self.dim) * (
            self.upper_bounds - self.lower_bounds
        )
        simplex = self._build_simplex(x0)

        if self.keep_incumbent_on_restart and self.best_solution is not None:
            # Overwrite the vertex farthest from the incumbent, so the restart
            # keeps its spread rather than collapsing toward the old basin
            distances = np.linalg.norm(simplex - self.best_solution, axis=1)
            simplex[int(np.argmax(distances))] = self.best_solution.copy()

        self._adopt_simplex(simplex)
        self._record_point(self.simplex[0], self.fitness_values[0])
        self._reset_alpha()

    # ------------------------------------------------------------------
    # Overridden Nelder-Mead extension points
    # ------------------------------------------------------------------

    def _step_exhausted(self) -> bool:
        """
        Report whether the poll step size has contracted below `alpha_min`.

        Repeated unsuccessful polls drive α to zero geometrically. Once it falls
        below the threshold the incumbent has been probed along a positive
        spanning set at an arbitrarily small scale without improvement, which is
        the practical statement of "y_k is a local minimizer" and the point at
        which the search should move elsewhere.

        Returns:
            True if α < alpha_min
        """
        return self._alpha is not None and self._alpha < self.alpha_min

    def _shrink_simplex(self) -> None:
        """
        Replace the shrink step with a polling step.

        Called when every geometric transformation has failed. A successful poll
        replaces the worst vertex. An unsuccessful poll contracts α and leaves
        the simplex untouched — an unsuccessful iteration in direct-search
        terms — and only once α is exhausted does the simplex restart.
        """
        y_k = self.simplex[0].copy()
        f_k = self.fitness_values[0]

        found = self._polling_step(y_k, f_k)

        if found is None:
            if self._step_exhausted():
                self._restart_from_random()
            return

        point, fitness, eq_vio, ineq_vio, total_vio = found
        self.simplex[-1] = point
        self.fitness_values[-1] = fitness
        self.eq_violations_list[-1] = eq_vio
        self.ineq_violations_list[-1] = ineq_vio
        self.total_violations[-1] = total_vio
        self._record_point(point, fitness)

    def _restart_simplex(self) -> None:
        """
        Handle a tripped termination test by polling, then merging or restarting.

        If polling finds a better point y*, a fresh simplex Y* is built from it
        and merged with the current simplex, shifting the search toward y*
        without discarding the incumbent. An unsuccessful poll contracts α; once
        α is exhausted, y_k is accepted as a local minimizer and the simplex
        restarts from a random point.
        """
        y_k = self.simplex[0].copy()
        f_k = self.fitness_values[0]

        found = self._polling_step(y_k, f_k)

        if found is None:
            if self._step_exhausted():
                self._restart_from_random()
            else:
                # Nothing changed, but the termination test will fire again on
                # the next pass; perturb the worst vertex so the loop makes
                # progress rather than re-polling an identical simplex.
                self._reseed_worst_vertex()
            return

        point, fitness = found[0], found[1]
        self.num_merges += 1

        star = self._build_simplex(point)
        merged = ops.merge_simplices(self.simplex, star, self.gamma_m)

        self._adopt_simplex(merged)
        self._record_point(point, fitness)
        self._reset_alpha()

    def _reseed_worst_vertex(self) -> None:
        """
        Replace the worst vertex with a point one step away from the incumbent.

        Used when the termination test fires but the poll step size is not yet
        exhausted: the simplex has collapsed, so it needs rebuilding, but the
        incumbent has not yet been proved a local minimizer and a full restart
        would discard a live search.
        """
        direction = self._poll_directions()[0]
        point = ops.enforce_bounds(
            self.simplex[0] + self._alpha * direction,
            self.lower_bounds,
            self.upper_bounds,
        )

        if self.should_terminate():
            return

        fitness, eq_vio, ineq_vio, total_vio = self.evaluate(point)
        self.simplex[-1] = point
        self.fitness_values[-1] = fitness
        self.eq_violations_list[-1] = eq_vio
        self.ineq_violations_list[-1] = ineq_vio
        self.total_violations[-1] = total_vio
        self._sort_simplex()
