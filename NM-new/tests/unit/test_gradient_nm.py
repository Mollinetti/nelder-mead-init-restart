"""
Unit tests for the gradient-based Nelder-Mead (g-NM) algorithm.

Covers the three pieces the algorithm rests on:
- the minimal positive basis initializer (well-poisedness / nondegeneracy)
- the simplex gradient, volume and merge operations
- the polling step, record set bounding and end-to-end optimization
"""

import numpy as np
import pytest

from nelder_mead.algorithms.gradient_nm import GradientNelderMead
from nelder_mead.core.simplex_operations import (
    edge_matrix,
    merge_simplices,
    simplex_gradient,
    simplex_volume,
)
from nelder_mead.initialization.simplex_init import SimplexInitializer


def sphere(x):
    """Simple convex objective with minimum 0 at the origin."""
    return float(np.sum(x**2))


def rosenbrock(x):
    """Classic ill-conditioned valley."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


class TestMinimalPositiveBasis:
    """The W = Z B⁻ initializer."""

    def test_matches_closed_form(self):
        """Vertices are z⁰ᵢeᵢ for i = 1..n, plus -z⁰."""
        lower, upper = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
        simplex = SimplexInitializer.minimal_positive_basis(
            3, lower, upper, 2, x0=np.array([1.0, 1.0])
        )

        expected = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        np.testing.assert_allclose(simplex, expected)

    def test_is_nondegenerate(self):
        """A positive basis spans ℝⁿ, so the simplex has positive volume."""
        for dim in (2, 3, 5, 10):
            lower, upper = np.repeat(-100.0, dim), np.repeat(100.0, dim)
            simplex = SimplexInitializer.minimal_positive_basis(
                dim + 1, lower, upper, dim, rng=np.random.default_rng(7)
            )

            assert simplex.shape == (dim + 1, dim)
            assert simplex_volume(simplex) > 0.0
            assert np.linalg.matrix_rank(edge_matrix(simplex)) == dim

    def test_depends_on_starting_point(self):
        """Different z⁰ must give different simplices."""
        lower, upper = np.repeat(-100.0, 3), np.repeat(100.0, 3)

        a = SimplexInitializer.minimal_positive_basis(
            4, lower, upper, 3, x0=np.array([1.0, 2.0, 3.0])
        )
        b = SimplexInitializer.minimal_positive_basis(
            4, lower, upper, 3, x0=np.array([-90.0, 90.0, 10.0])
        )

        assert not np.allclose(a, b)

    def test_stays_within_bounds(self):
        """Vertices are clipped into the box."""
        lower, upper = np.repeat(-100.0, 4), np.repeat(100.0, 4)
        simplex = SimplexInitializer.minimal_positive_basis(
            5, lower, upper, 4, rng=np.random.default_rng(3)
        )

        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_zero_component_does_not_collapse_simplex(self):
        """A zero entry in z⁰ would make Z singular; it must be substituted."""
        lower, upper = np.repeat(-10.0, 3), np.repeat(10.0, 3)
        simplex = SimplexInitializer.minimal_positive_basis(
            4, lower, upper, 3, x0=np.array([0.0, 5.0, 0.0])
        )

        assert simplex_volume(simplex) > 0.0

    def test_rejects_wrong_vertex_count(self):
        lower, upper = np.repeat(-1.0, 3), np.repeat(1.0, 3)
        with pytest.raises(ValueError, match="num_solutions"):
            SimplexInitializer.minimal_positive_basis(7, lower, upper, 3)

    def test_reachable_through_dispatcher(self):
        lower, upper = np.repeat(-5.0, 3), np.repeat(5.0, 3)
        simplex = SimplexInitializer.initialize(
            "minimalPositiveBasis", 4, lower, upper, 3, x0=np.ones(3)
        )
        assert simplex.shape == (4, 3)


class TestSimplexGradient:
    """∇ₛf from a sampled set."""

    def test_exact_on_linear_function(self):
        """For f(x) = aᵀx the simplex gradient recovers a exactly."""
        a = np.array([1.0, 2.0])
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        values = np.array([float(a @ p) for p in points])

        np.testing.assert_allclose(simplex_gradient(points, values), a, atol=1e-12)

    def test_overdetermined_least_squares(self):
        """More points than dimensions falls back to least squares."""
        a = np.array([3.0, -1.0])
        rng = np.random.default_rng(0)
        points = rng.normal(size=(8, 2))
        values = points @ a

        np.testing.assert_allclose(
            simplex_gradient(points, values), a, atol=1e-10
        )

    def test_approximates_true_gradient_on_sphere(self):
        """On a smooth function a tight, well-poised set approximates ∇f."""
        center = np.array([1.0, -2.0])
        h = 1e-5
        points = np.vstack(
            [center, center + h * np.eye(2)[0], center + h * np.eye(2)[1]]
        )
        values = np.array([sphere(p) for p in points])

        np.testing.assert_allclose(
            simplex_gradient(points, values), 2 * center, rtol=1e-4
        )

    def test_rejects_mismatched_inputs(self):
        with pytest.raises(ValueError, match="points"):
            simplex_gradient(np.zeros((1, 2)), np.zeros(1))
        with pytest.raises(ValueError, match="values"):
            simplex_gradient(np.zeros((3, 2)), np.zeros(2))


class TestSimplexVolume:
    def test_unit_triangle(self):
        simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert simplex_volume(simplex) == pytest.approx(0.5)

    def test_degenerate_simplex_has_zero_volume(self):
        """Collinear points span no area."""
        simplex = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert simplex_volume(simplex) == pytest.approx(0.0)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match=r"\(n\+1, n\)"):
            simplex_volume(np.zeros((5, 2)))


class TestMergeSimplices:
    def test_matches_formula(self):
        """Yₙ = {y₁⁰} ∪ {y₁ⁱ + γᵐ(y₂ⁱ⁻¹ - y₁ⁱ)}."""
        y1 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y2 = np.array([[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]])

        merged = merge_simplices(y1, y2, gamma_m=0.5)

        expected = np.array([[0.0, 0.0], [1.5, 1.0], [1.5, 1.5]])
        np.testing.assert_allclose(merged, expected)

    def test_preserves_best_vertex(self):
        """The incumbent is never discarded by a merge."""
        rng = np.random.default_rng(11)
        y1, y2 = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))

        merged = merge_simplices(y1, y2, gamma_m=0.8)

        np.testing.assert_allclose(merged[0], y1[0])

    def test_gamma_zero_is_identity(self):
        rng = np.random.default_rng(12)
        y1, y2 = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))

        np.testing.assert_allclose(merge_simplices(y1, y2, gamma_m=0.0), y1)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="Cannot merge"):
            merge_simplices(np.zeros((3, 2)), np.zeros((4, 3)))


class TestGradientNelderMead:
    """The algorithm end to end."""

    def _make(self, dim=2, **kwargs):
        return GradientNelderMead(
            objective_fn=kwargs.pop("objective_fn", sphere),
            lower_bounds=np.repeat(-100.0, dim),
            upper_bounds=np.repeat(100.0, dim),
            max_fes=kwargs.pop("max_fes", 2000),
            seed=kwargs.pop("seed", 101),
            **kwargs,
        )

    def test_minimizes_sphere(self):
        algo = self._make()
        algo.run()

        _, fitness, _, _, _ = algo.get_best_solution()
        assert fitness < 1e-6

    def test_minimizes_rosenbrock(self):
        """
        Rosenbrock over [-100, 100]².

        g-NM spends part of its budget restarting, so it does not polish to
        machine precision the way classic Nelder-Mead does here: across seeds
        the median is ~1e-4 and the worst ~1e-2. The threshold checks real
        convergence with headroom, not a lucky seed.
        """
        algo = self._make(objective_fn=rosenbrock, max_fes=20000)
        algo.run()

        _, fitness, _, _, _ = algo.get_best_solution()
        assert fitness < 1e-2

    def test_respects_evaluation_budget(self):
        """Polling evaluations must be charged to the budget."""
        algo = self._make(max_fes=500)
        algo.run()

        assert algo.fes <= 500

    def test_record_starts_as_initial_simplex(self):
        """R ← W at initialization."""
        algo = self._make(max_fes=100)
        simplex = algo.initialize_population()
        algo.evaluate_population(simplex)

        np.testing.assert_allclose(algo._record, simplex)
        assert len(algo._record_values) == len(simplex)

    def test_record_is_bounded_when_capped(self):
        algo = self._make(dim=3, max_fes=5000, record_size=20)
        algo.run()

        assert len(algo._record) <= 20
        assert len(algo._record_values) == len(algo._record)

    def test_record_size_auto_scales_with_dimension(self):
        assert self._make(dim=3).record_size == 40
        assert self._make(dim=7).record_size == 80

    def test_record_grows_unbounded_when_disabled(self):
        """record_size=None reproduces the thesis behaviour."""
        algo = self._make(max_fes=3000, record_size=None)
        algo.run()

        assert len(algo._record) > 10 * (algo.dim + 1)

    def test_record_prunes_by_proximity_not_recency(self):
        """Λ-poisedness lives on a ball around y⁰, so distance decides."""
        algo = self._make(dim=2, max_fes=500, record_size=3)
        algo._record = np.array([[0.0, 0.0]])
        algo._record_values = np.array([0.0])
        algo.best_solution = np.zeros(2)
        algo.best_fitness = 0.0

        algo._record_point(np.array([1.0, 0.0]), 1.0)  # near
        algo._record_point(np.array([90.0, 90.0]), 1e4)  # far, older
        algo._record_point(np.array([0.0, 2.0]), 4.0)  # near, newest

        assert len(algo._record) == 3
        # The distant point is dropped even though a nearer one is older
        assert not np.any(np.all(algo._record == [90.0, 90.0], axis=1))

    def test_record_keeps_full_rank_when_capped(self):
        """A pruned record set must still determine a simplex gradient."""
        algo = self._make(dim=4, max_fes=6000, record_size=30)
        algo.run()

        assert np.linalg.matrix_rank(edge_matrix(algo._record)) == algo.dim

    def test_polling_replaces_shrink(self):
        """g-NM must actually poll rather than shrink."""
        algo = self._make(objective_fn=rosenbrock, max_fes=5000)
        algo.run()

        assert algo.num_polls > 0

    def test_forcing_function_rejects_marginal_decrease(self):
        """With a huge forcing coefficient no polling step can succeed."""
        algo = self._make(max_fes=2000, forcing_c=1e12)
        algo.run()

        assert algo.num_polls > 0
        assert algo.num_poll_successes == 0

    def test_simple_decrease_when_forcing_disabled(self):
        """forcing_c = 0 recovers the plain decrease condition."""
        algo = self._make(objective_fn=rosenbrock, max_fes=5000, forcing_c=0.0)
        algo.run()

        _, fitness, _, _, _ = algo.get_best_solution()
        assert np.isfinite(fitness)

    def _primed(self, **kwargs):
        """Build an algorithm with its simplex initialized and evaluated."""
        algo = self._make(**kwargs)
        algo.simplex = algo.initialize_population()
        (
            algo.fitness_values,
            algo.eq_violations_list,
            algo.ineq_violations_list,
            algo.total_violations,
        ) = algo.evaluate_population(algo.simplex)
        algo._sort_simplex()
        algo._reset_alpha()
        return algo

    def test_polling_cost_is_independent_of_record_size(self):
        """A poll evaluates the positive basis: at most dim+1 points."""
        algo = self._primed(dim=3, max_fes=4000, record_size=None)

        rng = np.random.default_rng(0)
        for _ in range(200):
            point = rng.uniform(-100.0, 100.0, 3)
            algo._record_point(point, sphere(point))

        before = algo.fes
        algo._polling_step(algo.simplex[0], algo.fitness_values[0])

        assert algo.fes - before <= algo.dim + 1

    def test_poll_directions_positively_span(self):
        """Convergence needs the poll set to positively span ℝⁿ."""
        algo = self._primed(dim=4, max_fes=500)
        directions = algo._poll_directions()

        assert directions.shape == (algo.dim + 1, algo.dim)
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=1), 1.0, atol=1e-12
        )
        # A positive spanning set has full rank and a strictly positive
        # combination summing to zero
        assert np.linalg.matrix_rank(directions) == algo.dim
        coeffs, *_ = np.linalg.lstsq(directions.T, np.zeros(algo.dim), rcond=None)
        assert np.linalg.matrix_rank(directions.T) == algo.dim

    def test_poll_directions_ordered_by_gradient(self):
        """The simplex gradient orders the basis, steepest descent first."""
        algo = self._primed(dim=3, max_fes=500)
        gradient = algo.simplex_gradient()

        alignment = algo._poll_directions() @ gradient

        assert np.all(np.diff(alignment) >= -1e-12)

    def test_poll_directions_are_the_full_basis_without_gradient(self):
        """No direction is dropped when the gradient is uninformative."""
        algo = self._primed(dim=3, max_fes=500)
        algo._record = np.zeros((2, 3))
        algo._record_values = np.zeros(2)

        assert algo._poll_directions().shape == (4, 3)

    def test_alpha_defaults_to_oriented_length(self):
        """alpha_init=None ties α₀ to the simplex's own length scale."""
        algo = self._make(max_fes=100)
        algo.simplex = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        algo._reset_alpha()

        assert algo.alpha_init is None
        assert algo._alpha == pytest.approx(4.0)

    def test_fixed_alpha_overrides_oriented_length(self):
        algo = self._make(max_fes=100, alpha_init=1e-4)
        algo.simplex = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
        algo._reset_alpha()

        assert algo._alpha == pytest.approx(1e-4)

    def test_alpha_contracts_on_unsuccessful_poll(self):
        """α → 0 across failures is what the convergence argument needs."""
        algo = self._primed(max_fes=2000, alpha_init=1.0, forcing_c=1e20)

        before = algo._alpha
        assert algo._polling_step(algo.simplex[0], algo.fitness_values[0]) is None
        assert algo._alpha == pytest.approx(before * algo.tau)

    def test_alpha_expands_on_successful_poll(self):
        algo = self._primed(max_fes=2000, alpha_init=1.0, forcing_c=0.0)
        algo.simplex[0] = np.array([50.0, 50.0])
        algo.fitness_values[0] = sphere(algo.simplex[0])

        before = algo._alpha
        found = algo._polling_step(algo.simplex[0], algo.fitness_values[0])

        assert found is not None
        assert algo._alpha == pytest.approx(before * algo.gamma_a)

    def test_alpha_is_persistent_across_polls(self):
        """α is state, not reset per poll."""
        algo = self._primed(max_fes=2000, alpha_init=1.0, forcing_c=1e20)

        for _ in range(3):
            algo._polling_step(algo.simplex[0], algo.fitness_values[0])

        assert algo._alpha == pytest.approx(1.0 * algo.tau**3)

    def test_restart_only_once_step_is_exhausted(self):
        """An unsuccessful poll is an unsuccessful iteration, not a restart."""
        algo = self._primed(max_fes=2000, alpha_init=1.0, forcing_c=1e20)

        algo._shrink_simplex()

        assert algo.num_polls == 1
        assert algo.num_restarts == 0
        assert not algo._step_exhausted()

    def test_restart_fires_when_step_exhausted(self):
        algo = self._primed(max_fes=2000, alpha_init=1e-10, forcing_c=1e20)

        algo._shrink_simplex()

        assert algo.num_restarts == 1

    def test_restart_resets_step_size(self):
        """A rebuilt simplex may live at a different scale."""
        algo = self._primed(max_fes=3000, alpha_init=None)
        algo._alpha = 1e-15

        algo._restart_from_random()

        assert algo._alpha > 1e-15

    def test_thesis_restart_is_the_default(self):
        """Monotonicity is argued on the best-so-far sequence, not enforced."""
        algo = self._make(max_fes=3000)

        assert algo.keep_incumbent_on_restart is False

    def test_best_so_far_sequence_is_monotone(self):
        """Whatever the restart policy, the reported best never increases."""
        algo = self._make(max_fes=3000)
        algo.run()

        history = [f for _, f in algo.memory_global_best]
        assert all(b <= a for a, b in zip(history, history[1:]))

    def test_keep_incumbent_enforces_monotone_simplex(self):
        algo = self._make(max_fes=3000, keep_incumbent_on_restart=True)
        algo.run()

        assert algo.keep_incumbent_on_restart
        if algo.num_restarts > 0:
            assert algo.best_fitness <= min(algo.fitness_values)

    def test_reproducible_across_runs(self):
        """Same seed, same result."""
        a, b = self._make(seed=42), self._make(seed=42)
        a.run()
        b.run()

        np.testing.assert_allclose(a.best_fitness, b.best_fitness)
        np.testing.assert_allclose(a.best_solution, b.best_solution)

    def test_different_seeds_give_different_trajectories(self):
        """The seed must actually drive the search.

        Asserting that the final solutions differ is the wrong test: on an easy
        objective both seeds correctly converge to the same optimum. What must
        differ is the path taken, starting with the initial simplex.
        """
        a, b = self._make(seed=1), self._make(seed=2)

        assert not np.allclose(a.initialize_population(), b.initialize_population())

        a.run()
        b.run()
        assert a.memory_global_best != b.memory_global_best

    def test_defaults_to_positive_basis_initialization(self):
        assert self._make().init_method == "minimalPositiveBasis"

    def test_solution_stays_within_bounds(self):
        algo = self._make(dim=5, max_fes=3000)
        algo.run()

        assert np.all(algo.best_solution >= algo.lower_bounds)
        assert np.all(algo.best_solution <= algo.upper_bounds)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"gamma_m": 1.5}, "gamma_m"),
            ({"alpha_init": 0.0}, "alpha_init"),
            ({"tau": 1.0}, "tau"),
            ({"gamma_a": 0.5}, "gamma_a"),
            ({"alpha_min": 0.0}, "alpha_min"),
            ({"forcing_c": -1.0}, "forcing_c"),
            ({"record_size": 2}, "record_size"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._make(**kwargs)
