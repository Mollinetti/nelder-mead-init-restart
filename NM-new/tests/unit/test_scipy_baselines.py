#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the library baseline wrappers.

Verifies that every baseline:
- respects the function evaluation budget exactly (never overshoots max_fes)
- returns a solution of the correct shape
- is reproducible: the same seed produces the same result
- routes constraint handling through the shared barrier machinery
"""

import numpy as np
import pytest

from nelder_mead.algorithms.scipy_baselines import (
    CMAES,
    ScipyDE,
    ScipyNelderMead,
    ScipySLSQP,
    get_baseline_algorithms,
)
from nelder_mead.constraints.deb_barrier import DebBarrier


ALL_BASELINES = [ScipyDE, ScipySLSQP, ScipyNelderMead, CMAES]

LOWER = np.array([-5.0, -5.0, -5.0])
UPPER = np.array([5.0, 5.0, 5.0])


def sphere(x):
    """Simple convex objective with optimum f(0) = 0."""
    return float(np.sum(x**2))


def offset_constraint(x):
    """g(x) <= 0 forcing the first variable to at least 1.0."""
    return np.array([1.0 - x[0]])


@pytest.mark.parametrize("algorithm_class", ALL_BASELINES)
class TestBaselineContract:
    """Each baseline must honour the BaseAlgorithm contract BatchRunner relies on."""

    def test_respects_budget(self, algorithm_class):
        """FES must never exceed max_fes, and the budget must actually be used."""
        max_fes = 400
        alg = algorithm_class(
            objective_fn=sphere,
            lower_bounds=LOWER,
            upper_bounds=UPPER,
            max_fes=max_fes,
            seed=42,
        )
        alg.run()

        assert 0 < alg.fes <= max_fes
        # A budget of 400 on a 3-D sphere should be substantially consumed;
        # stopping far short means the optimizer is exiting instead of restarting.
        assert alg.fes >= max_fes * 0.5

    def test_returns_valid_solution(self, algorithm_class):
        """The best solution must have the problem's dimension and be finite."""
        alg = algorithm_class(
            objective_fn=sphere,
            lower_bounds=LOWER,
            upper_bounds=UPPER,
            max_fes=300,
            seed=42,
        )
        alg.run()

        solution, fitness, _, _, _ = alg.get_best_solution()

        assert solution is not None
        assert solution.shape == (3,)
        assert np.isfinite(fitness)

    def test_reproducible(self, algorithm_class):
        """The same seed must produce the same result."""
        results = []
        for _ in range(2):
            alg = algorithm_class(
                objective_fn=sphere,
                lower_bounds=LOWER,
                upper_bounds=UPPER,
                max_fes=300,
                seed=7,
            )
            alg.run()
            results.append((alg.best_fitness, alg.fes))

        assert results[0] == results[1]

    def test_records_convergence_history(self, algorithm_class):
        """memory_global_best must be populated for the convergence plots."""
        alg = algorithm_class(
            objective_fn=sphere,
            lower_bounds=LOWER,
            upper_bounds=UPPER,
            max_fes=300,
            seed=42,
        )
        alg.run()

        assert len(alg.memory_global_best) > 0
        fes_values = [fes for fes, _ in alg.memory_global_best]
        assert fes_values == sorted(fes_values)

    def test_makes_progress(self, algorithm_class):
        """On a convex problem every baseline should get close to the optimum."""
        alg = algorithm_class(
            objective_fn=sphere,
            lower_bounds=LOWER,
            upper_bounds=UPPER,
            max_fes=2000,
            seed=42,
        )
        alg.run()

        assert alg.best_fitness < 1.0

    def test_handles_constraints(self, algorithm_class):
        """With a barrier, the constrained optimum must beat the unconstrained one."""
        barrier = DebBarrier(bounds=(LOWER, UPPER), num_solutions=4)
        alg = algorithm_class(
            objective_fn=sphere,
            constraint_ineq_fn=offset_constraint,
            lower_bounds=LOWER,
            upper_bounds=UPPER,
            max_fes=2000,
            seed=42,
            barrier=barrier,
        )
        alg.run()

        solution, _, _, _, total_violation = alg.get_best_solution()

        # The constraint x0 >= 1 must be respected to a loose tolerance
        assert total_violation < 1e-2
        assert solution[0] > 0.9


class TestBudgetEdgeCases:
    """The budget guard must hold at the boundaries, not just in the common case."""

    def test_minimal_budget(self):
        """A budget smaller than one optimizer iteration must still terminate."""
        alg = ScipyDE(
            objective_fn=sphere,
            lower_bounds=LOWER,
            upper_bounds=UPPER,
            max_fes=5,
            seed=42,
        )
        alg.run()

        assert 0 < alg.fes <= 5
        assert alg.best_solution is not None

    def test_failing_objective_does_not_abort(self):
        """An objective that raises must be absorbed by BaseAlgorithm's handler."""

        def unstable(x):
            if x[0] > 0:
                raise RuntimeError("simulated solver failure")
            return float(np.sum(x**2))

        alg = ScipyNelderMead(
            objective_fn=unstable,
            lower_bounds=LOWER,
            upper_bounds=UPPER,
            max_fes=200,
            seed=42,
        )
        alg.run()

        assert alg.best_solution is not None


class TestBaselineRegistry:
    """get_baseline_algorithms must produce BatchRunner-compatible configs."""

    def test_config_shape(self):
        """Each config needs a 'class' and a 'name' key and nothing surprising."""
        configs = get_baseline_algorithms()

        assert len(configs) >= 3
        for config in configs:
            assert "class" in config
            assert "name" in config
            assert set(config.keys()) == {"class", "name"}

    def test_names_are_unique(self):
        """Names key the results dict, so duplicates would silently drop results."""
        names = [c["name"] for c in get_baseline_algorithms()]

        assert len(names) == len(set(names))

    def test_cmaes_can_be_excluded(self):
        """The CMA-ES opt-out must work for environments without the package."""
        names = [c["name"] for c in get_baseline_algorithms(include_cmaes=False)]

        assert "CMA-ES" not in names


class TestRunIndependence:
    """The 30-run protocol requires the seed to actually change the run.

    The package's default simplex initializer, "spendleySimplex", builds a
    regular simplex around the centre of the box and never consults the random
    number generator. With it, a Nelder-Mead run is fully determined by the
    problem, and the seed reaches the algorithm only through the restart
    strategy, which fires only when a stopping criterion triggers. Every "run"
    is then the same run, and the reported standard deviation measures nothing.

    This was not hypothetical: on the black-box suite at 2,000 evaluations the
    adaptive variant produced one distinct result across all 30 seeds.
    """

    def test_spendley_initializer_ignores_the_seed(self):
        """Pins the property that makes the default unsuitable for the protocol."""
        from nelder_mead.initialization.simplex_init import SimplexInitializer

        lower, upper = np.zeros(8), np.ones(8)
        first = SimplexInitializer.initialize(
            "spendleySimplex", 9, lower, upper, 8, None, np.random.default_rng(1)
        )
        second = SimplexInitializer.initialize(
            "spendleySimplex", 9, lower, upper, 8, None, np.random.default_rng(999)
        )

        assert np.array_equal(first, second)

    def test_uniform_initializer_uses_the_seed(self):
        """The initializer the experiments use must depend on the seed."""
        from nelder_mead.initialization.simplex_init import SimplexInitializer

        lower, upper = np.zeros(8), np.ones(8)
        first = SimplexInitializer.initialize(
            "uniform", 9, lower, upper, 8, None, np.random.default_rng(1)
        )
        second = SimplexInitializer.initialize(
            "uniform", 9, lower, upper, 8, None, np.random.default_rng(999)
        )

        assert not np.array_equal(first, second)

    def test_nelder_mead_runs_differ_across_seeds_under_uniform_init(self):
        """End to end: distinct seeds must give distinct runs.

        This is the property the experiment protocol depends on, so it is
        asserted on the algorithm rather than only on the initializer.
        """
        from nelder_mead.algorithms.nelder_mead import NelderMead

        results = set()
        for seed in range(101, 107):
            alg = NelderMead(
                objective_fn=sphere,
                constraint_ineq_fn=offset_constraint,
                lower_bounds=LOWER,
                upper_bounds=UPPER,
                max_fes=500,
                seed=seed,
                init_method="uniform",
            )
            alg.run()
            results.add(round(alg.best_fitness, 12))

        assert len(results) > 1

    def test_experiment_runner_defaults_to_a_stochastic_initializer(self):
        """The runner must not silently fall back to the deterministic default."""
        import importlib.util
        import pathlib

        script = (
            pathlib.Path(__file__).parents[2]
            / "benchmarks" / "scripts" / "run_journal_suite.py"
        )
        spec = importlib.util.spec_from_file_location("run_journal_suite", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.DEFAULT_INIT_METHOD == "uniform"
        for config in module.algorithm_configs():
            if config["name"] in ("cNM", "aNM"):
                assert config["init_method"] == "uniform"
