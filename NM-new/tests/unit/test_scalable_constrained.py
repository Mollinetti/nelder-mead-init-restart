#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the dimension-parameterized problem families.

The important tests here are the optimum checks: each family's recorded optimum
is a closed form derived by hand, and TestKnownOptima re-derives it numerically
with SLSQP. If a family's algebra were wrong, the scalability curves would be
measuring an arbitrary reference point rather than the distance to the solution.

The rest verify that the families stay well posed as the dimension grows: the
feasible set must remain non-empty, the constraint counts must scale as declared,
and rotations must remain orthogonal.
"""

import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint, minimize

from nelder_mead.benchmarks.problem_suite import OptimizationProblem
from nelder_mead.benchmarks.scalable_constrained import (
    DEFAULT_DIMENSIONS,
    SCALABLE_FAMILIES,
    ScalableActiveSetInequality,
    ScalableEqualityManifold,
    ScalableRotatedInequality,
    ScalableSphereInequality,
    _rotation_matrix,
    get_scalability_sweep,
)


ALL_CLASSES = [
    ScalableSphereInequality,
    ScalableRotatedInequality,
    ScalableEqualityManifold,
    ScalableActiveSetInequality,
]

SWEEP_DIMENSIONS = [2, 5, 10, 30]


def solve_numerically(problem, tries=40, seed=0):
    """Best feasible objective SLSQP can find from many random starts."""
    lower, upper = problem.bounds
    constraints = []
    if problem.num_ineq_constraints:
        constraints.append(
            NonlinearConstraint(problem.constraint_ineq, -np.inf, 0.0)
        )
    if problem.num_eq_constraints:
        constraints.append(NonlinearConstraint(problem.constraint_eq, 0.0, 0.0))

    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(tries):
        x0 = lower + rng.random(problem.dim) * (upper - lower)
        result = minimize(
            problem.objective,
            x0,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if result.success and problem.compute_violation(result.x) < 1e-8:
            best = min(best, result.fun)
    return best


class TestKnownOptima:
    """The recorded closed-form optima must match an independent numerical solve."""

    @pytest.mark.parametrize("dim", [4, 5, 10, 20, 30])
    def test_sphere_optimum_is_quarter_dimension(self, dim):
        """For n >= 4 the symmetric point x_i = 1/2 is optimal, giving f* = n/4."""
        problem = ScalableSphereInequality(dim=dim)

        assert problem.optimum_value == pytest.approx(dim / 4.0)
        assert solve_numerically(problem) == pytest.approx(dim / 4.0, rel=1e-6)

    @pytest.mark.parametrize("dim", [2, 3])
    def test_sphere_records_no_optimum_below_four_dimensions(self, dim):
        """Below n = 4 the exclusion constraint binds and the closed form fails."""
        problem = ScalableSphereInequality(dim=dim)

        assert problem.optimum_value is None
        # The closed form would claim n/4; the true optimum is strictly worse
        assert solve_numerically(problem) > dim / 4.0

    @pytest.mark.parametrize("dim", [2, 5, 10, 20, 30])
    def test_active_set_optimum_is_two_and_a_quarter_per_dimension(self, dim):
        """All constraints active at x_i = 1/2 gives f* = 2.25*n."""
        problem = ScalableActiveSetInequality(dim=dim)

        assert problem.optimum_value == pytest.approx(2.25 * dim)
        assert solve_numerically(problem) == pytest.approx(2.25 * dim, rel=1e-6)

    @pytest.mark.parametrize("dim,num_eq", [(5, 1), (10, 2), (20, 4), (30, 6)])
    def test_equality_optimum_is_sum_over_blocks(self, dim, num_eq):
        """Projecting the all-ones vector onto each block's unit sphere is optimal."""
        problem = ScalableEqualityManifold(dim=dim, num_eq=num_eq)
        expected = sum((np.sqrt(len(b)) - 1.0) ** 2 for b in problem.blocks)

        assert problem.optimum_value == pytest.approx(expected)
        assert solve_numerically(problem) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("dim", [5, 10, 30])
    def test_recorded_optimum_is_attained_at_recorded_location(self, dim):
        """optimum_location must actually evaluate to optimum_value and be feasible."""
        for problem in (
            ScalableSphereInequality(dim=dim),
            ScalableActiveSetInequality(dim=dim),
            ScalableEqualityManifold(dim=dim),
        ):
            assert problem.objective(problem.optimum_location) == pytest.approx(
                problem.optimum_value
            )
            assert problem.compute_violation(problem.optimum_location) < 1e-9

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_rotated_records_no_optimum(self, dim):
        """The rotated family's optimum has no closed form and must not be invented."""
        assert ScalableRotatedInequality(dim=dim).optimum_value is None


@pytest.mark.parametrize("problem_class", ALL_CLASSES)
@pytest.mark.parametrize("dim", SWEEP_DIMENSIONS)
class TestWellPosedAcrossDimensions:
    """Each family must remain a valid, solvable problem at every swept dimension."""

    def test_declared_counts_match_returned_arrays(self, problem_class, dim):
        """A miscounted constraint is silently dropped by BatchRunner."""
        problem = problem_class(dim=dim)
        x = np.zeros(dim)

        assert len(problem.constraint_eq(x)) == problem.num_eq_constraints
        assert len(problem.constraint_ineq(x)) == problem.num_ineq_constraints

    def test_feasible_region_is_non_empty(self, problem_class, dim):
        """A family with no feasible point would produce meaningless curves."""
        problem = problem_class(dim=dim)

        assert np.isfinite(solve_numerically(problem, tries=25))

    def test_objective_is_finite_in_the_box(self, problem_class, dim):
        """Random points inside the box must evaluate without overflow."""
        problem = problem_class(dim=dim)
        lower, upper = problem.bounds
        rng = np.random.default_rng(1)

        for _ in range(20):
            x = lower + rng.random(dim) * (upper - lower)
            assert np.isfinite(problem.objective(x))

    def test_rejects_wrong_dimension(self, problem_class, dim):
        """A wrong-length input must raise rather than broadcast."""
        problem = problem_class(dim=dim)

        with pytest.raises(ValueError, match="Expected"):
            problem.objective(np.ones(dim + 1))

    def test_is_an_optimization_problem(self, problem_class, dim):
        """BatchRunner requires the OptimizationProblem interface."""
        assert isinstance(problem_class(dim=dim), OptimizationProblem)

    def test_name_encodes_the_dimension(self, problem_class, dim):
        """Results are keyed by name, so two dimensions must not collide."""
        assert str(dim) in problem_class(dim=dim).name


class TestConstraintScaling:
    """Constraint counts must scale as each family documents."""

    @pytest.mark.parametrize("dim", SWEEP_DIMENSIONS)
    def test_fixed_constraint_families(self, dim):
        """Sphere and rotated hold their constraint count fixed at 2."""
        assert ScalableSphereInequality(dim=dim).num_ineq_constraints == 2
        assert ScalableRotatedInequality(dim=dim).num_ineq_constraints == 2

    @pytest.mark.parametrize("dim", SWEEP_DIMENSIONS)
    def test_active_set_scales_with_dimension(self, dim):
        """The active-set family keeps a constant constraint-to-variable ratio."""
        assert ScalableActiveSetInequality(dim=dim).num_ineq_constraints == dim
        assert (
            ScalableActiveSetInequality(dim=dim, constraints_per_dim=2)
            .num_ineq_constraints
            == 2 * dim
        )

    @pytest.mark.parametrize("dim", [10, 20, 30])
    def test_equality_blocks_partition_every_coordinate(self, dim):
        """Every coordinate must appear in exactly one block, or it is unconstrained."""
        problem = ScalableEqualityManifold(dim=dim)
        covered = np.concatenate(problem.blocks)

        assert sorted(covered.tolist()) == list(range(dim))

    def test_equality_default_count_scales(self):
        """The default block count grows with dimension so codimension scales."""
        assert ScalableEqualityManifold(dim=10).num_eq_constraints == 2
        assert ScalableEqualityManifold(dim=50).num_eq_constraints == 10


class TestRotation:
    """The rotation must be a genuine, reproducible orthogonal transform."""

    @pytest.mark.parametrize("dim", SWEEP_DIMENSIONS)
    def test_is_orthogonal(self, dim):
        """A non-orthogonal 'rotation' would change the problem's conditioning."""
        rotation = _rotation_matrix(dim, seed=7)

        assert np.allclose(rotation @ rotation.T, np.eye(dim), atol=1e-12)
        assert abs(abs(np.linalg.det(rotation)) - 1.0) < 1e-12

    def test_is_reproducible(self):
        """The same seed must give the same rotation across runs and processes."""
        assert np.array_equal(_rotation_matrix(8, seed=3), _rotation_matrix(8, seed=3))

    def test_different_seeds_differ(self):
        """Otherwise the seed parameter is a no-op."""
        assert not np.array_equal(
            _rotation_matrix(8, seed=3), _rotation_matrix(8, seed=4)
        )

    def test_rotation_makes_the_objective_non_separable(self):
        """The point of the family: coordinates must actually couple.

        A separable objective f(x) = sum_i phi_i(x_i) has a zero mixed second
        difference in any two coordinates. A non-zero one is therefore exactly the
        evidence that the rotation coupled them.
        """
        problem = ScalableRotatedInequality(dim=5)
        base = np.zeros(5)
        step_x0 = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        step_x1 = np.array([0.0, 0.5, 0.0, 0.0, 0.0])

        mixed_second_difference = (
            problem.objective(base + step_x0 + step_x1)
            - problem.objective(base + step_x0)
            - problem.objective(base + step_x1)
            + problem.objective(base)
        )

        assert abs(mixed_second_difference) > 1e-9

    def test_unrotated_family_is_separable(self):
        """The control: the same measurement must vanish without the rotation."""
        problem = ScalableSphereInequality(dim=5)
        base = np.zeros(5)
        step_x0 = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        step_x1 = np.array([0.0, 0.5, 0.0, 0.0, 0.0])

        mixed_second_difference = (
            problem.objective(base + step_x0 + step_x1)
            - problem.objective(base + step_x0)
            - problem.objective(base + step_x1)
            + problem.objective(base)
        )

        assert abs(mixed_second_difference) < 1e-12


class TestValidation:
    """Constructor guards, so a malformed sweep fails at setup rather than mid-run."""

    @pytest.mark.parametrize("problem_class", ALL_CLASSES)
    def test_rejects_dimension_below_two(self, problem_class):
        """A one-dimensional 'simplex sweep' is not meaningful."""
        with pytest.raises(ValueError, match="at least 2"):
            problem_class(dim=1)

    def test_rejects_too_many_equality_blocks(self):
        """More blocks than coordinates cannot partition anything."""
        with pytest.raises(ValueError, match="num_eq must be between"):
            ScalableEqualityManifold(dim=5, num_eq=6)

    def test_rejects_zero_equality_blocks(self):
        """Zero equality constraints would silently make the family unconstrained."""
        with pytest.raises(ValueError, match="num_eq must be between"):
            ScalableEqualityManifold(dim=5, num_eq=0)

    def test_rejects_non_positive_constraints_per_dim(self):
        """Guarding the multiplier keeps the declared count honest."""
        with pytest.raises(ValueError, match="at least 1"):
            ScalableActiveSetInequality(dim=5, constraints_per_dim=0)


class TestSweepBuilder:
    """get_scalability_sweep must produce the grid the experiment script consumes."""

    def test_default_grid_shape(self):
        """Every family at every default dimension."""
        sweep = get_scalability_sweep()

        assert set(sweep) == set(SCALABLE_FAMILIES)
        for problems in sweep.values():
            assert len(problems) == len(DEFAULT_DIMENSIONS)

    def test_orders_by_dimension(self):
        """Curves are plotted against dimension, so the order must be monotone."""
        sweep = get_scalability_sweep(dimensions=(30, 2, 10))

        for problems in sweep.values():
            dims = [dim for dim, _ in problems]
            assert dims == sorted(dims)

    def test_family_selection(self):
        """A partial sweep must be possible; the full grid is expensive."""
        sweep = get_scalability_sweep(dimensions=(2, 5), families=["sphere"])

        assert set(sweep) == {"sphere"}
        assert [dim for dim, _ in sweep["sphere"]] == [2, 5]

    def test_rejects_unknown_family(self):
        """A typo must fail loudly rather than silently shrinking the sweep."""
        with pytest.raises(ValueError, match="Unknown scalable families"):
            get_scalability_sweep(families=["nonexistent"])

    def test_default_dimensions_bracket_the_classic_suite(self):
        """The sweep must cover the paper's n <= 7 regime and extend well past it."""
        assert min(DEFAULT_DIMENSIONS) <= 2
        assert any(d <= 7 for d in DEFAULT_DIMENSIONS)
        assert max(DEFAULT_DIMENSIONS) >= 50

    def test_problem_names_are_unique(self):
        """Results are keyed by name across the whole sweep."""
        sweep = get_scalability_sweep()
        names = [p.name for problems in sweep.values() for _, p in problems]

        assert len(names) == len(set(names))
