#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the CEC2020 real-world constrained problem subset.

The load-bearing test here is TestAgainstOfficialReference: it compares every
objective and constraint against values derived from the official competition
implementation (cec20_func.m / Cal_par.m). A mistranscribed coefficient in one of
these problems would silently invalidate every downstream table and statistical
test, and would not be caught by any structural check, so it is pinned against
the authoritative source rather than against our own output.

The remaining tests cover the problem metadata, bounds and input validation.
"""

import json
import pathlib

import numpy as np
import pytest

from nelder_mead.benchmarks.cec2020_rw import (
    RC01, RC02, RC03, RC04, RC05, RC06, RC08, RC14, RC15, RC16, RC17, RC19,
    get_cec2020_rw_by_dimension,
    get_cec2020_rw_problems,
)
from nelder_mead.benchmarks.problem_suite import OptimizationProblem


FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "cec2020_rw_reference.json"

ALL_CLASSES = [RC01, RC02, RC03, RC04, RC05, RC06, RC08, RC14, RC15, RC16, RC17, RC19]

# Dimension, equality count, inequality count, taken from Cal_par.m. RC17 is the
# documented exception: Cal_par.m says 3 inequalities, cec20_func.m computes 4.
EXPECTED_SHAPE = {
    "RC01": (9, 8, 0),
    "RC02": (11, 9, 0),
    "RC03": (7, 0, 14),
    "RC04": (6, 4, 1),
    "RC05": (9, 4, 2),
    "RC06": (38, 32, 0),
    "RC08": (2, 0, 2),
    "RC14": (10, 0, 10),
    "RC15": (7, 0, 11),
    "RC16": (14, 0, 15),
    "RC17": (3, 0, 4),
    "RC19": (4, 0, 5),
}


@pytest.fixture(scope="module")
def reference():
    """Reference values derived from the official MATLAB implementation."""
    with open(FIXTURE) as fh:
        return json.load(fh)


class TestAgainstOfficialReference:
    """Every coefficient must match the official competition implementation."""

    def test_fixture_covers_every_problem(self, reference):
        """A problem missing from the fixture would go silently unverified."""
        assert set(reference["problems"]) == set(get_cec2020_rw_problems())

    @pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
    def test_objective_matches(self, reference, name):
        """f(x) must agree with the official implementation at every sample."""
        problem = get_cec2020_rw_problems()[name]
        for sample in reference["problems"][name]["samples"]:
            x = np.array(sample["x"])
            assert problem.objective(x) == pytest.approx(sample["f"], rel=1e-12)

    @pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
    def test_inequality_constraints_match(self, reference, name):
        """g(x) must agree elementwise with the official implementation."""
        problem = get_cec2020_rw_problems()[name]
        for sample in reference["problems"][name]["samples"]:
            x = np.array(sample["x"])
            actual = problem.constraint_ineq(x)
            assert len(actual) == len(sample["g"])
            for got, want in zip(actual, sample["g"]):
                assert got == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
    def test_equality_constraints_match(self, reference, name):
        """h(x) must agree elementwise with the official implementation."""
        problem = get_cec2020_rw_problems()[name]
        for sample in reference["problems"][name]["samples"]:
            x = np.array(sample["x"])
            actual = problem.constraint_eq(x)
            assert len(actual) == len(sample["h"])
            for got, want in zip(actual, sample["h"]):
                assert got == pytest.approx(want, rel=1e-12)

    @pytest.mark.parametrize("name", sorted(EXPECTED_SHAPE))
    def test_bounds_match(self, reference, name):
        """Bounds must match Cal_par.m exactly; a wrong box changes the problem."""
        problem = get_cec2020_rw_problems()[name]
        expected = reference["problems"][name]

        assert np.allclose(problem.bounds[0], expected["lower_bounds"], rtol=0, atol=0)
        assert np.allclose(problem.bounds[1], expected["upper_bounds"], rtol=0, atol=0)


@pytest.mark.parametrize("problem_class", ALL_CLASSES)
class TestProblemContract:
    """Structural properties BatchRunner and the barrier machinery depend on."""

    def test_declared_shape(self, problem_class):
        """Dimension and constraint counts must match the official parameters."""
        problem = problem_class()
        dim, num_eq, num_ineq = EXPECTED_SHAPE[problem.name]

        assert problem.dim == dim
        assert problem.num_eq_constraints == num_eq
        assert problem.num_ineq_constraints == num_ineq

    def test_constraint_counts_are_honest(self, problem_class):
        """The declared counts must match what the functions actually return.

        BatchRunner attaches a constraint function only when its count is
        non-zero, so a count that disagrees with the returned array silently
        drops constraints from the optimization.
        """
        problem = problem_class()
        lower, upper = problem.bounds
        x = lower + 0.5 * (upper - lower)

        assert len(problem.constraint_eq(x)) == problem.num_eq_constraints
        assert len(problem.constraint_ineq(x)) == problem.num_ineq_constraints

    def test_bounds_are_ordered_and_finite(self, problem_class):
        """Every box must be finite and non-degenerate."""
        problem = problem_class()
        lower, upper = problem.bounds

        assert len(lower) == len(upper) == problem.dim
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all(lower < upper)

    def test_objective_is_finite_inside_the_box(self, problem_class):
        """A random feasible-box point must produce a finite objective."""
        problem = problem_class()
        lower, upper = problem.bounds
        rng = np.random.default_rng(3)

        for _ in range(20):
            x = lower + rng.random(problem.dim) * (upper - lower)
            assert np.isfinite(problem.objective(x))

    def test_rejects_wrong_dimension(self, problem_class):
        """A wrong-length input must raise rather than broadcast silently."""
        problem = problem_class()
        wrong = np.ones(problem.dim + 1)

        with pytest.raises(ValueError, match="Expected"):
            problem.objective(wrong)

    def test_is_an_optimization_problem(self, problem_class):
        """BatchRunner requires the OptimizationProblem interface."""
        assert isinstance(problem_class(), OptimizationProblem)


class TestRegistry:
    """The hand-maintained registry must stay in step with the module."""

    def test_contains_every_class(self):
        """A class missing from the registry would never be benchmarked."""
        registry = get_cec2020_rw_problems()

        assert len(registry) == len(ALL_CLASSES)
        assert set(registry) == {cls.__name__ for cls in ALL_CLASSES}

    def test_returns_fresh_instances(self):
        """Problems are stateless, but sharing instances across runs invites bugs."""
        assert get_cec2020_rw_problems()["RC01"] is not get_cec2020_rw_problems()["RC01"]

    def test_names_match_keys(self):
        """The key and the problem's own name must agree; results are keyed by name."""
        for key, problem in get_cec2020_rw_problems().items():
            assert key == problem.name

    def test_dimension_ordering_spans_the_intended_range(self):
        """The subset exists to widen the dimension range beyond the classic suite."""
        ordered = get_cec2020_rw_by_dimension()
        dims = [dim for dim, _ in ordered]

        assert dims == sorted(dims)
        assert min(dims) == 2
        assert max(dims) == 38
        # The paper's classic suite tops out at 7 variables; this must go well past
        assert sum(1 for d in dims if d > 7) >= 5


class TestDocumentedDiscrepancies:
    """Pins the two inconsistencies found in the official sources."""

    def test_rc17_implements_four_constraints(self):
        """Cal_par.m declares 3, cec20_func.m computes 4; we follow the code.

        The fourth constraint is the outer-diameter limit (x1 + x2)/1.5 - 1 <= 0,
        which appears in every literature formulation of this problem.
        """
        problem = RC17()
        x = np.array([0.05, 0.25, 2.0])

        assert problem.num_ineq_constraints == 4
        assert problem.constraint_ineq(x)[3] == pytest.approx((0.05 + 0.25) / 1.5 - 1.0)

    def test_integer_variables_are_rounded(self):
        """RC08 and RC14 round inside the functions, as the official code does."""
        problem = RC08()

        # x2 = 0.4 and x2 = 0.4999 both round to 0, so the objective must agree
        assert problem.objective(np.array([1.0, 0.4])) == pytest.approx(
            problem.objective(np.array([1.0, 0.4999]))
        )
        # and 0.6 rounds to 1, which must differ
        assert problem.objective(np.array([1.0, 0.6])) != pytest.approx(
            problem.objective(np.array([1.0, 0.4]))
        )
