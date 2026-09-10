#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Library baseline algorithms for benchmark comparison.

This module wraps well-established third-party optimizers as :class:`BaseAlgorithm`
subclasses so that :class:`~nelder_mead.testing.batch_runner.BatchRunner` can drive
them with no special-casing.

The wrapping is deliberate rather than calling SciPy directly from the experiment
scripts. Every baseline routes each candidate through ``self.evaluate``, so the
function-evaluation counter, the barrier penalty, Deb's feasibility ranking and the
convergence history are computed identically for the baselines and for the
Nelder-Mead variants. Any difference in the reported results is therefore a
difference between the search strategies, not between two accounting schemes.

The evaluation budget is enforced by raising :class:`_BudgetExhausted` from the
objective wrapper once ``max_fes`` is reached. Third-party optimizers stop on their
own iteration counters, which cannot be mapped onto a function-evaluation budget
exactly (SLSQP's finite-difference gradients and CMA-ES's restarts both spend a
variable number of evaluations per iteration), so the budget is enforced at the one
place every optimizer must pass through.

Note on the barrier: like :class:`~nelder_mead.algorithms.nelder_mead.NelderMead`,
these wrappers do not call ``barrier.update()``. The barrier therefore behaves
identically for every algorithm in a comparison.

References:
    - Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific
      computing in Python. Nature Methods, 17(3), 261-272.
    - Storn, R., & Price, K. (1997). Differential evolution - a simple and efficient
      heuristic for global optimization over continuous spaces. Journal of Global
      Optimization, 11(4), 341-359.
    - Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation
      in evolution strategies. Evolutionary Computation, 9(2), 159-195.
    - Kraft, D. (1988). A software package for sequential quadratic programming.
      DFVLR-FB 88-28, Institut fuer Dynamik der Flugsysteme.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .base_algorithm import BaseAlgorithm
from ..constraints.barrier_base import Barrier


class _BudgetExhausted(Exception):
    """Raised from the objective wrapper once the FES budget is spent."""


class ScipyBaseline(BaseAlgorithm):
    """
    Base class for third-party optimizer wrappers.

    Subclasses implement :meth:`_optimize`, which should call the wrapped optimizer
    with :meth:`_penalized` as its objective. Budget exhaustion and optimizer
    failures are handled here, so :meth:`_optimize` can be written as if the budget
    were unlimited.

    Attributes:
        Inherited from :class:`BaseAlgorithm`.
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
        init_method: str = "uniform",
        seed: int = 101,
        barrier: Optional[Barrier] = None,
    ):
        """
        Initialize the baseline wrapper.

        Args:
            objective_fn: Function to minimize
            constraint_eq_fn: Equality constraint function h(x), returns raw values
            constraint_ineq_fn: Inequality constraint function g(x) <= 0, raw values
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
            max_fes: Maximum number of function evaluations
            num_solutions: Population size where the optimizer accepts one
            init_method: Unused; accepted so the constructor signature matches
                         NelderMead and BatchRunner can drive both the same way
            seed: Random seed for reproducibility
            barrier: Barrier method for constraint handling
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
        )

    def _penalized(self, x: np.ndarray) -> float:
        """
        Objective seen by the wrapped optimizer.

        Args:
            x: Candidate solution

        Returns:
            Penalized objective value

        Raises:
            _BudgetExhausted: If the FES budget is already spent
        """
        if self.fes >= self.max_fes:
            raise _BudgetExhausted()
        return self.evaluate(np.asarray(x, dtype=float))[0]

    def _bounds_list(self) -> List[Tuple[float, float]]:
        """Return bounds in the (low, high) pair form SciPy expects."""
        return list(zip(self.lower_bounds, self.upper_bounds))

    def _random_start(self) -> np.ndarray:
        """Draw a uniformly random point inside the box constraints."""
        return self.lower_bounds + self.rng.random(self.dim) * (
            self.upper_bounds - self.lower_bounds
        )

    def _optimize(self) -> None:
        """Run the wrapped optimizer. Implemented by subclasses."""
        raise NotImplementedError

    def run(self) -> None:
        """
        Execute the wrapped optimizer until the evaluation budget is spent.

        Budget exhaustion is the expected exit path and is not an error. Other
        optimizer failures are downgraded to a warning so that a single failed run
        does not abort a batch experiment; the best solution found before the
        failure is retained.
        """
        try:
            self._optimize()
        except _BudgetExhausted:
            pass
        except Exception as exc:  # pragma: no cover - optimizer-specific failures
            warnings.warn(
                f"{type(self).__name__} terminated early: {exc}. "
                "Retaining the best solution found so far.",
                UserWarning,
            )


class ScipyDE(ScipyBaseline):
    """
    Differential Evolution via ``scipy.optimize.differential_evolution``.

    The population-based reference point for the comparison, and the closest
    library equivalent to the DE baseline used in the original experiment.

    ``maxiter`` is set high deliberately: the run is bounded by the FES budget in
    :meth:`_penalized`, not by a generation count. ``polish=False`` keeps the final
    L-BFGS-B refinement out of the picture, which would otherwise spend an
    uncounted, gradient-based budget that no other algorithm here receives.

    Example:
        >>> import numpy as np
        >>> alg = ScipyDE(
        ...     objective_fn=lambda x: float(np.sum(x ** 2)),
        ...     lower_bounds=np.array([-5.0, -5.0]),
        ...     upper_bounds=np.array([5.0, 5.0]),
        ...     max_fes=500,
        ...     seed=42,
        ... )
        >>> alg.run()
        >>> alg.best_fitness < 1.0
        True
    """

    def _optimize(self) -> None:
        """Run differential evolution until the budget is spent."""
        # popsize is a multiplier on dim in SciPy; target ~30 individuals to match
        # the population size used for the heuristics in the original experiment.
        popsize = max(5, round(30 / max(1, self.dim)))

        differential_evolution(
            self._penalized,
            self._bounds_list(),
            seed=self.seed,
            maxiter=10**9,
            popsize=popsize,
            polish=False,
            tol=0.0,
            init="sobol",
        )


class ScipySLSQP(ScipyBaseline):
    """
    Sequential Least Squares Programming via ``scipy.optimize.minimize``.

    A gradient-based reference point. SLSQP has no derivative information here (the
    penalized objective is a black box), so it builds finite-difference gradients,
    and every one of those evaluations is counted against the budget. That is the
    intended comparison: it shows what a derivative-based method costs on a problem
    class the paper argues is derivative-free.

    SLSQP converges to a local solution quickly, so it is restarted from a fresh
    random point whenever it returns, until the budget is spent.

    Example:
        >>> import numpy as np
        >>> alg = ScipySLSQP(
        ...     objective_fn=lambda x: float(np.sum(x ** 2)),
        ...     lower_bounds=np.array([-5.0, -5.0]),
        ...     upper_bounds=np.array([5.0, 5.0]),
        ...     max_fes=500,
        ...     seed=42,
        ... )
        >>> alg.run()
        >>> alg.best_fitness < 1e-3
        True
    """

    def _optimize(self) -> None:
        """Run SLSQP with random restarts until the budget is spent."""
        while True:
            minimize(
                self._penalized,
                self._random_start(),
                method="SLSQP",
                bounds=self._bounds_list(),
                options={"maxiter": 200},
            )


class ScipyNelderMead(ScipyBaseline):
    """
    Stock Nelder-Mead via ``scipy.optimize.minimize``.

    The control condition. It isolates what this package's Nelder-Mead adds over
    the textbook implementation (the barrier integration, the restart strategies
    and the simplex initializers), because both are handed the same penalized
    objective and the same budget.

    Restarted from a fresh random point on each convergence, matching
    :class:`ScipySLSQP`, so that the comparison is between search behaviour rather
    than between restart policies.

    Example:
        >>> import numpy as np
        >>> alg = ScipyNelderMead(
        ...     objective_fn=lambda x: float(np.sum(x ** 2)),
        ...     lower_bounds=np.array([-5.0, -5.0]),
        ...     upper_bounds=np.array([5.0, 5.0]),
        ...     max_fes=500,
        ...     seed=42,
        ... )
        >>> alg.run()
        >>> alg.best_fitness < 1e-3
        True
    """

    def _optimize(self) -> None:
        """Run stock Nelder-Mead with random restarts until the budget is spent."""
        while True:
            minimize(
                self._penalized,
                self._random_start(),
                method="Nelder-Mead",
                bounds=self._bounds_list(),
                options={"maxiter": 10**9, "maxfev": 10**9},
            )


class CMAES(ScipyBaseline):
    """
    CMA-ES via the ``cmaes`` package.

    The only genuinely different search family in the baseline set: it adapts a full
    covariance matrix rather than relying on differential vectors or a simplex, and
    it is the algorithm the original experiment's CMA-ES baseline refers to.

    The optimizer is restarted with a doubled population whenever it signals
    convergence (the IPOP restart scheme), so that the full budget is used.

    Raises:
        ImportError: If the optional ``cmaes`` package is not installed.

    Example:
        >>> import numpy as np
        >>> alg = CMAES(
        ...     objective_fn=lambda x: float(np.sum(x ** 2)),
        ...     lower_bounds=np.array([-5.0, -5.0]),
        ...     upper_bounds=np.array([5.0, 5.0]),
        ...     max_fes=500,
        ...     seed=42,
        ... )
        >>> alg.run()
        >>> alg.best_fitness < 1e-3
        True
    """

    def _optimize(self) -> None:
        """Run CMA-ES with IPOP restarts until the budget is spent."""
        try:
            from cmaes import CMA
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CMAES requires the 'cmaes' package. Install it with "
                "'pip install cmaes'."
            ) from exc

        bounds = np.column_stack([self.lower_bounds, self.upper_bounds])
        # A sigma of a quarter of the box width puts roughly two standard deviations
        # inside the bounds, the usual recommendation for a bounded search space.
        sigma = float(np.min(self.upper_bounds - self.lower_bounds)) / 4.0
        population_size = None

        while True:
            optimizer = CMA(
                mean=self._random_start(),
                sigma=sigma,
                bounds=bounds,
                seed=int(self.rng.integers(0, 2**31 - 1)),
                population_size=population_size,
            )

            while not optimizer.should_stop():
                solutions = []
                for _ in range(optimizer.population_size):
                    x = optimizer.ask()
                    solutions.append((x, self._penalized(x)))
                optimizer.tell(solutions)

            # IPOP restart: double the population on each restart
            population_size = optimizer.population_size * 2


def get_baseline_algorithms(include_cmaes: bool = True) -> List[Dict[str, Any]]:
    """
    Return the baseline algorithm configurations for BatchRunner.

    Args:
        include_cmaes: If True, include CMA-ES when the ``cmaes`` package is
                       importable. A warning is emitted and CMA-ES is dropped when
                       it is not, so that an experiment still runs without it.

    Returns:
        List of configuration dicts in the form
        :meth:`~nelder_mead.testing.batch_runner.BatchRunner.compare_algorithms`
        expects, i.e. ``{'class': ..., 'name': ...}``.

    Example:
        >>> configs = get_baseline_algorithms(include_cmaes=False)
        >>> [c["name"] for c in configs]
        ['DE', 'SLSQP', 'scipy-NM']
    """
    configs: List[Dict[str, Any]] = [
        {"class": ScipyDE, "name": "DE"},
        {"class": ScipySLSQP, "name": "SLSQP"},
        {"class": ScipyNelderMead, "name": "scipy-NM"},
    ]

    if include_cmaes:
        try:
            import cmaes  # noqa: F401

            configs.append({"class": CMAES, "name": "CMA-ES"})
        except ImportError:
            warnings.warn(
                "The 'cmaes' package is not installed; CMA-ES is excluded from the "
                "baseline set. Install it with 'pip install cmaes'.",
                UserWarning,
            )

    return configs
