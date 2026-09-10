#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the simulation-based black-box problem.

Two properties carry the weight here, because they are the entire justification
for the problem's existence:

- TestIsGenuinelyConstrained: the minimum-mass design must be infeasible. If it
  were feasible the answer would be the lower-bound corner and the simulation
  would be decoration.
- TestNonSmoothness: the stress constraint must actually have kinks. If it were
  smooth, this problem would not substantiate the paper's derivative-free
  motivation any better than the analytic benchmarks already do.

The rest verify the physics behaves sensibly, that the memo does not corrupt
results, and that the instrumentation reports real costs.
"""

import numpy as np
import pytest

from nelder_mead.benchmarks.blackbox import (
    CantileverBracketFEA,
    get_blackbox_problems,
)
from nelder_mead.benchmarks.problem_suite import OptimizationProblem


@pytest.fixture(scope="module")
def problem():
    """A small bracket, kept coarse so the suite stays fast."""
    return CantileverBracketFEA(num_groups=6, nx=18, ny=6)


class TestIsGenuinelyConstrained:
    """The constraints must shape the solution, or the problem is pointless."""

    def test_minimum_mass_design_is_infeasible(self, problem):
        """The lower-bound corner must violate the stress allowable."""
        lower, _ = problem.bounds

        assert problem.compute_violation(lower) > 0
        assert problem.constraint_ineq(lower)[0] > 0

    def test_maximum_mass_design_is_feasible(self, problem):
        """The upper-bound corner must be feasible, or nothing is."""
        _, upper = problem.bounds

        assert np.all(problem.constraint_ineq(upper) <= 0)

    def test_optimum_lies_strictly_above_the_minimum_mass(self, problem):
        """Since the cheapest design is infeasible, the answer must cost more."""
        lower, _ = problem.bounds
        minimum_mass = problem.objective(lower)

        # A feasible design must be heavier than the infeasible lower-bound corner
        _, upper = problem.bounds
        assert problem.objective(upper) > minimum_mass


class TestNonSmoothness:
    """The stress constraint must be non-differentiable, not merely nonlinear."""

    def test_critical_element_changes_across_the_domain(self, problem):
        """A kink exists exactly where the argmax over elements switches.

        If the same element were always critical, the max would be a single smooth
        function and the constraint would be differentiable.
        """
        lower, upper = problem.bounds
        rng = np.random.default_rng(0)

        critical_elements = set()
        for _ in range(40):
            x = lower + rng.random(problem.dim) * (upper - lower)
            problem._cache_key = None  # force a genuine solve for every sample
            problem._simulate(x)
            critical_elements.add(problem.critical_element)

        assert len(critical_elements) > 1

    def test_taper_constraints_have_absolute_value_kinks(self, problem):
        """|t_i - t_{i+1}| is non-differentiable where the difference changes sign."""
        base = np.full(problem.dim, 0.02)

        # Sweep the first strip through the point where t0 - t1 crosses zero
        deltas = np.linspace(-0.004, 0.004, 9)
        values = []
        for delta in deltas:
            x = base.copy()
            x[0] += delta
            values.append(problem.constraint_ineq(x)[2])

        # A V shape: the minimum is interior and both arms rise away from it
        values = np.array(values)
        assert np.argmin(values) not in (0, len(values) - 1)
        assert values[0] > values[len(values) // 2]
        assert values[-1] > values[len(values) // 2]


class TestPhysics:
    """The simulation must behave like a cantilever, not like a random function."""

    def test_mass_increases_with_thickness(self, problem):
        """Mass is a positive integral of thickness."""
        thin = problem.objective(np.full(problem.dim, 0.01))
        thick = problem.objective(np.full(problem.dim, 0.03))

        assert thick > thin

    def test_stiffer_beam_deflects_less(self, problem):
        """Tripling the thickness must reduce the tip deflection."""
        _, _, thin_deflection = problem._simulate(np.full(problem.dim, 0.01))
        _, _, thick_deflection = problem._simulate(np.full(problem.dim, 0.03))

        assert thick_deflection < thin_deflection

    def test_thicker_beam_carries_less_stress(self, problem):
        """Peak von Mises stress must fall as material is added."""
        _, thin_stress, _ = problem._simulate(np.full(problem.dim, 0.01))
        _, thick_stress, _ = problem._simulate(np.full(problem.dim, 0.03))

        assert thick_stress < thin_stress

    def test_results_are_finite_across_the_box(self, problem):
        """No thickness in range may produce a singular or divergent solve."""
        lower, upper = problem.bounds
        rng = np.random.default_rng(2)

        for _ in range(15):
            x = lower + rng.random(problem.dim) * (upper - lower)
            mass, stress, deflection = problem._simulate(x)
            assert np.isfinite(mass) and mass > 0
            assert np.isfinite(stress) and stress > 0
            assert np.isfinite(deflection) and deflection > 0

    def test_deterministic(self, problem):
        """The same design must give the same answer; runs are compared across seeds."""
        x = np.full(problem.dim, 0.017)
        first = problem._simulate(x)
        problem._cache_key = None  # defeat the memo to force a genuine re-solve
        second = problem._simulate(x)

        assert first == second


class TestEvaluationCost:
    """Instrumentation must report the real cost, which is the problem's point."""

    def test_memo_prevents_double_solving(self):
        """objective() and constraint_ineq() at one point must cost one solve."""
        problem = CantileverBracketFEA(num_groups=4, nx=12, ny=4)
        x = np.full(4, 0.02)

        problem.objective(x)
        problem.constraint_ineq(x)

        assert problem.solve_count == 1

    def test_distinct_points_each_cost_a_solve(self):
        """The memo must not return stale results for a different design."""
        problem = CantileverBracketFEA(num_groups=4, nx=12, ny=4)

        for value in (0.01, 0.02, 0.03):
            problem.objective(np.full(4, value))

        assert problem.solve_count == 3

    def test_memo_returns_correct_values_after_a_switch(self):
        """Alternating between two designs must not leak one's answer into the other."""
        problem = CantileverBracketFEA(num_groups=4, nx=12, ny=4)
        thin = np.full(4, 0.01)
        thick = np.full(4, 0.03)

        first_thin = problem.objective(thin)
        problem.objective(thick)
        second_thin = problem.objective(thin)

        assert first_thin == second_thin

    def test_reports_solve_time(self):
        """A non-zero mean solve time is what makes this an 'expensive' problem."""
        problem = CantileverBracketFEA(num_groups=4, nx=12, ny=4)
        problem.objective(np.full(4, 0.02))

        assert problem.solve_count == 1
        assert problem.mean_solve_time > 0.0

    def test_instrumentation_resets(self):
        """Between runs the counters must be zeroable."""
        problem = CantileverBracketFEA(num_groups=4, nx=12, ny=4)
        problem.objective(np.full(4, 0.02))
        problem.reset_instrumentation()

        assert problem.solve_count == 0
        assert problem.mean_solve_time == 0.0


class TestProblemContract:
    """Structural requirements BatchRunner depends on."""

    def test_is_an_optimization_problem(self, problem):
        """BatchRunner requires the OptimizationProblem interface."""
        assert isinstance(problem, OptimizationProblem)

    def test_declared_constraint_count(self, problem):
        """Stress, deflection, and one taper limit per adjacent pair."""
        x = np.full(problem.dim, 0.02)

        assert problem.num_ineq_constraints == problem.dim + 1
        assert len(problem.constraint_ineq(x)) == problem.num_ineq_constraints
        assert len(problem.constraint_eq(x)) == 0

    def test_no_optimum_is_claimed(self, problem):
        """The solution has no closed form and must not be invented."""
        assert problem.optimum_value is None

    def test_rejects_wrong_dimension(self, problem):
        """A wrong-length input must raise rather than misindex the mesh."""
        with pytest.raises(ValueError, match="Expected"):
            problem.objective(np.ones(problem.dim + 1))

    def test_name_encodes_dimension(self, problem):
        """Results are keyed by name."""
        assert problem.name == "CantileverBracketFEA-6D"


class TestValidation:
    """Constructor guards, so a bad configuration fails at setup."""

    def test_rejects_too_few_groups(self):
        """With one strip there are no taper constraints and no sizing decision."""
        with pytest.raises(ValueError, match="at least 2"):
            CantileverBracketFEA(num_groups=1)

    def test_rejects_mesh_coarser_than_the_grouping(self):
        """Every strip needs at least one element column or it has no effect."""
        with pytest.raises(ValueError, match="at least num_groups"):
            CantileverBracketFEA(num_groups=10, nx=5)

    def test_rejects_inverted_thickness_bounds(self):
        """An inverted box would be caught later and less clearly."""
        with pytest.raises(ValueError, match="less than max_thickness"):
            CantileverBracketFEA(min_thickness=0.05, max_thickness=0.005)


class TestRegistry:
    """The registry the experiment script consumes."""

    def test_contains_the_default_problem(self):
        """The default configuration is the one reported in the paper."""
        problems = get_blackbox_problems()

        assert "CantileverBracketFEA-8D" in problems
        assert problems["CantileverBracketFEA-8D"].dim == 8
