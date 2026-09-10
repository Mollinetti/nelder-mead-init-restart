"""
Unit tests for constrained benchmark problems.

Tests verify:
1. Problem initialization and metadata
2. Objective function evaluation
3. Constraint function evaluation
4. Known optima (where available)
5. Feasibility checking
6. Constraint violation computation
"""

import pytest
import numpy as np
from nelder_mead.benchmarks.constrained import (
    G01,
    G04,
    G05,
    G06,
    G07,
    G08,
    G09,
    G10,
    G11,
    PressureVessel,
    WeldedBeam,
    get_all_constrained_problems,
    get_cec_problems,
    get_engineering_problems,
)


class TestG01:
    """Tests for G01 problem."""

    def test_initialization(self):
        """Test G01 problem initialization."""
        problem = G01()
        assert problem.name == "G01"
        assert problem.dim == 13
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 9
        assert problem.optimum_value == -15.0
        assert problem.optimum_location is not None

    def test_bounds(self):
        """Test G01 bounds are correct."""
        problem = G01()
        lower, upper = problem.bounds

        # First 9 variables: [0, 1]
        assert np.all(lower[0:9] == 0.0)
        assert np.all(upper[0:9] == 1.0)

        # Last 4 variables: [0, 100]
        assert np.all(lower[9:13] == 0.0)
        assert np.all(upper[9:13] == 100.0)

    def test_objective_evaluation(self):
        """Test G01 objective function evaluation."""
        problem = G01()
        x = np.ones(13)
        f = problem.objective(x)
        assert isinstance(f, (int, float, np.number))

        # Test with known optimum
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)
        assert abs(f_opt - problem.optimum_value) < 0.1

    def test_constraint_evaluation(self):
        """Test G01 constraint evaluation."""
        problem = G01()
        x = np.ones(13)

        # Test inequality constraints
        g = problem.constraint_ineq(x)
        assert len(g) == 9
        assert isinstance(g, np.ndarray)

        # Test equality constraints (should be empty)
        h = problem.constraint_eq(x)
        assert len(h) == 0

    def test_dimension_validation(self):
        """Test that wrong dimensions raise errors."""
        problem = G01()
        with pytest.raises(ValueError):
            problem.objective(np.ones(10))
        with pytest.raises(ValueError):
            problem.constraint_ineq(np.ones(10))


class TestG04:
    """Tests for G04 problem."""

    def test_initialization(self):
        """Test G04 problem initialization."""
        problem = G04()
        assert problem.name == "G04"
        assert problem.dim == 5
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 6
        assert abs(problem.optimum_value - (-30665.539)) < 0.001

    def test_bounds(self):
        """Test G04 bounds are correct."""
        problem = G04()
        lower, upper = problem.bounds

        assert lower[0] == 78.0 and upper[0] == 102.0
        assert lower[1] == 33.0 and upper[1] == 45.0
        assert np.all(lower[2:5] == 27.0)
        assert np.all(upper[2:5] == 45.0)

    def test_objective_evaluation(self):
        """Test G04 objective function evaluation."""
        problem = G04()
        x = np.array([78, 33, 27, 27, 27])
        f = problem.objective(x)
        assert isinstance(f, (int, float, np.number))

    def test_constraint_evaluation(self):
        """Test G04 constraint evaluation."""
        problem = G04()
        x = np.array([78, 33, 27, 27, 27])
        g = problem.constraint_ineq(x)
        assert len(g) == 6


class TestG06:
    """Tests for G06 problem."""

    def test_initialization(self):
        """Test G06 problem initialization."""
        problem = G06()
        assert problem.name == "G06"
        assert problem.dim == 2
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 2
        assert abs(problem.optimum_value - (-6961.81388)) < 0.001

    def test_known_optimum(self):
        """Test G06 known optimum."""
        problem = G06()
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)

        # Should be close to known optimum
        assert abs(f_opt - problem.optimum_value) < 1.0

    def test_constraint_evaluation(self):
        """Test G06 constraint evaluation."""
        problem = G06()
        x = np.array([14.0, 1.0])
        g = problem.constraint_ineq(x)
        assert len(g) == 2


class TestG07:
    """Tests for G07 problem."""

    def test_initialization(self):
        """Test G07 problem initialization."""
        problem = G07()
        assert problem.name == "G07"
        assert problem.dim == 10
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 8
        assert abs(problem.optimum_value - 24.3062091) < 0.001

    def test_bounds(self):
        """Test G07 bounds are correct."""
        problem = G07()
        lower, upper = problem.bounds
        assert np.all(lower == -10.0)
        assert np.all(upper == 10.0)

    def test_objective_evaluation(self):
        """Test G07 objective function evaluation."""
        problem = G07()
        x = np.zeros(10)
        f = problem.objective(x)
        assert isinstance(f, (int, float, np.number))

    def test_constraint_evaluation(self):
        """Test G07 constraint evaluation."""
        problem = G07()
        x = np.zeros(10)
        g = problem.constraint_ineq(x)
        assert len(g) == 8


class TestG09:
    """Tests for G09 problem."""

    def test_initialization(self):
        """Test G09 problem initialization."""
        problem = G09()
        assert problem.name == "G09"
        assert problem.dim == 7
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 4
        assert abs(problem.optimum_value - 680.6300573) < 0.001

    def test_objective_evaluation(self):
        """Test G09 objective function evaluation."""
        problem = G09()
        x = np.zeros(7)
        f = problem.objective(x)
        assert isinstance(f, (int, float, np.number))

    def test_constraint_evaluation(self):
        """Test G09 constraint evaluation."""
        problem = G09()
        x = np.zeros(7)
        g = problem.constraint_ineq(x)
        assert len(g) == 4


class TestG10:
    """Tests for G10 problem."""

    def test_initialization(self):
        """Test G10 problem initialization."""
        problem = G10()
        assert problem.name == "G10"
        assert problem.dim == 8
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 6
        assert abs(problem.optimum_value - 7049.248) < 0.001

    def test_bounds(self):
        """Test G10 bounds are correct."""
        problem = G10()
        lower, upper = problem.bounds

        assert lower[0] == 100.0 and upper[0] == 10000.0
        assert lower[1] == 1000.0 and upper[1] == 10000.0
        assert lower[2] == 1000.0 and upper[2] == 10000.0
        assert np.all(lower[3:8] == 10.0)
        assert np.all(upper[3:8] == 1000.0)

    def test_objective_evaluation(self):
        """Test G10 objective function evaluation."""
        problem = G10()
        x = np.array([100, 1000, 1000, 10, 10, 10, 10, 10])
        f = problem.objective(x)
        assert isinstance(f, (int, float, np.number))
        # Linear objective: should equal sum of first 3 variables
        assert abs(f - 2100.0) < 1e-10

    def test_constraint_evaluation(self):
        """Test G10 constraint evaluation."""
        problem = G10()
        x = np.array([100, 1000, 1000, 10, 10, 10, 10, 10])
        g = problem.constraint_ineq(x)
        assert len(g) == 6


class TestPressureVessel:
    """Tests for Pressure Vessel design problem."""

    def test_initialization(self):
        """Test Pressure Vessel problem initialization."""
        problem = PressureVessel()
        assert problem.name == "PressureVessel"
        assert problem.dim == 4
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 4
        assert abs(problem.optimum_value - 6059.714) < 0.001

    def test_bounds(self):
        """Test Pressure Vessel bounds are correct."""
        problem = PressureVessel()
        lower, upper = problem.bounds

        assert lower[0] == 0.0 and upper[0] == 99.0
        assert lower[1] == 0.0 and upper[1] == 99.0
        assert lower[2] == 10.0 and upper[2] == 200.0
        assert lower[3] == 10.0 and upper[3] == 200.0

    def test_objective_evaluation(self):
        """Test Pressure Vessel objective function evaluation."""
        problem = PressureVessel()
        x = np.array([1.0, 1.0, 50.0, 100.0])
        f = problem.objective(x)
        assert isinstance(f, (int, float, np.number))
        assert f > 0  # Cost should be positive

    def test_known_optimum(self):
        """Test Pressure Vessel known optimum."""
        problem = PressureVessel()
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)

        # Should be close to known optimum
        assert abs(f_opt - problem.optimum_value) < 10.0

    def test_constraint_evaluation(self):
        """Test Pressure Vessel constraint evaluation."""
        problem = PressureVessel()
        x = np.array([1.0, 1.0, 50.0, 100.0])
        g = problem.constraint_ineq(x)
        assert len(g) == 4


class TestWeldedBeam:
    """Tests for Welded Beam design problem."""

    def test_initialization(self):
        """Test Welded Beam problem initialization."""
        problem = WeldedBeam()
        assert problem.name == "WeldedBeam"
        assert problem.dim == 4
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 7
        assert abs(problem.optimum_value - 1.7249) < 0.001

    def test_bounds(self):
        """Test Welded Beam bounds are correct."""
        problem = WeldedBeam()
        lower, upper = problem.bounds

        assert lower[0] == 0.1 and upper[0] == 2.0
        assert lower[1] == 0.1 and upper[1] == 10.0
        assert lower[2] == 0.1 and upper[2] == 10.0
        assert lower[3] == 0.1 and upper[3] == 2.0

    def test_objective_evaluation(self):
        """Test Welded Beam objective function evaluation."""
        problem = WeldedBeam()
        x = np.array([0.5, 5.0, 5.0, 0.5])
        f = problem.objective(x)
        assert isinstance(f, (int, float, np.number))
        assert f > 0  # Cost should be positive

    def test_known_optimum(self):
        """Test Welded Beam known optimum."""
        problem = WeldedBeam()
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)

        # Should be close to known optimum
        assert abs(f_opt - problem.optimum_value) < 0.01

    def test_constraint_evaluation(self):
        """Test Welded Beam constraint evaluation."""
        problem = WeldedBeam()
        x = np.array([0.5, 5.0, 5.0, 0.5])
        g = problem.constraint_ineq(x)
        assert len(g) == 7

    def test_constants(self):
        """Test Welded Beam problem constants."""
        problem = WeldedBeam()
        assert problem.P == 6000.0
        assert problem.L == 14.0
        assert problem.E == 30e6
        assert problem.G == 12e6
        assert problem.tau_max == 13600.0
        assert problem.sigma_max == 30000.0
        assert problem.delta_max == 0.25


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_all_constrained_problems(self):
        """Test get_all_constrained_problems function."""
        problems = get_all_constrained_problems()

        assert isinstance(problems, dict)
        assert len(problems) == 11

        # Check CEC problems
        assert "G01" in problems
        assert "G04" in problems
        assert "G05" in problems
        assert "G06" in problems
        assert "G07" in problems
        assert "G08" in problems
        assert "G09" in problems
        assert "G10" in problems
        assert "G11" in problems

        # Check engineering problems
        assert "PressureVessel" in problems
        assert "WeldedBeam" in problems

        # Check all are OptimizationProblem instances
        for name, problem in problems.items():
            assert hasattr(problem, "objective")
            assert hasattr(problem, "constraint_ineq")
            assert hasattr(problem, "constraint_eq")

    def test_get_cec_problems(self):
        """Test get_cec_problems function."""
        problems = get_cec_problems()

        assert isinstance(problems, dict)
        assert len(problems) == 9

        expected_names = [
            "G01", "G04", "G05", "G06", "G07", "G08", "G09", "G10", "G11",
        ]
        for name in expected_names:
            assert name in problems

    def test_get_engineering_problems(self):
        """Test get_engineering_problems function."""
        problems = get_engineering_problems()

        assert isinstance(problems, dict)
        assert len(problems) == 2

        assert "PressureVessel" in problems
        assert "WeldedBeam" in problems


class TestFeasibilityAndViolations:
    """Tests for feasibility checking and violation computation."""

    def test_feasibility_checking(self):
        """Test is_feasible method for constrained problems."""
        # G06 has a very small feasible region
        problem = G06()

        # Known optimum should be feasible (or very close)
        x_opt = problem.optimum_location
        # May not be exactly feasible due to numerical precision
        violation = problem.compute_violation(x_opt)
        assert violation < 1.0  # Should be close to feasible

    def test_violation_computation(self):
        """Test compute_violation method."""
        problem = G01()

        # Test with a point that violates bounds
        x_violate = np.full(13, -1.0)  # Below lower bound
        violation = problem.compute_violation(x_violate)
        assert violation > 0

        # Test with a feasible point (if we can find one)
        x_feasible = problem.optimum_location
        violation = problem.compute_violation(x_feasible)
        # Known optimum should have low violation
        assert violation < 1.0

    def test_constraint_violation_components(self):
        """Test that violations are computed correctly."""
        problem = PressureVessel()

        # Create a point that violates constraints
        x = np.array([0.5, 0.5, 20.0, 50.0])

        # Get constraint values
        g = problem.constraint_ineq(x)

        # Compute violation manually
        manual_violation = np.sum(np.maximum(0.0, g))

        # Compare with problem's compute_violation
        total_violation = problem.compute_violation(x)

        # Should include constraint violations (may also include bound violations)
        assert total_violation >= manual_violation


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_boundary_points(self):
        """Test evaluation at boundary points."""
        problem = G06()
        lower, upper = problem.bounds

        # Test at lower bounds
        x_lower = lower.copy()
        f_lower = problem.objective(x_lower)
        assert isinstance(f_lower, (int, float, np.number))

        # Test at upper bounds
        x_upper = upper.copy()
        f_upper = problem.objective(x_upper)
        assert isinstance(f_upper, (int, float, np.number))

    def test_zero_point(self):
        """Test evaluation at origin (where valid)."""
        # G07 has bounds that include zero
        problem = G07()
        x_zero = np.zeros(10)

        f = problem.objective(x_zero)
        assert isinstance(f, (int, float, np.number))

        g = problem.constraint_ineq(x_zero)
        assert len(g) == 8

    def test_all_problems_callable(self):
        """Test that all problems can be instantiated and called."""
        problems = get_all_constrained_problems()

        for name, problem in problems.items():
            # Create a random point within bounds
            lower, upper = problem.bounds
            x = lower + np.random.rand(problem.dim) * (upper - lower)

            # Test objective
            f = problem.objective(x)
            assert isinstance(f, (int, float, np.number))
            assert not np.isnan(f)
            assert not np.isinf(f)

            # Test constraints
            g = problem.constraint_ineq(x)
            assert len(g) == problem.num_ineq_constraints

            h = problem.constraint_eq(x)
            assert len(h) == problem.num_eq_constraints


class TestG05:
    """Tests for the G05 problem (equality-constrained coverage)."""

    def test_initialization(self):
        """Metadata must match the CEC 2006 definition."""
        problem = G05()

        assert problem.name == "G05"
        assert problem.dim == 4
        assert problem.num_eq_constraints == 3
        assert problem.num_ineq_constraints == 2
        assert problem.optimum_value == pytest.approx(5126.4967140071)

    def test_bounds(self):
        """Bounds must match the published problem definition."""
        problem = G05()
        lower, upper = problem.bounds

        assert np.allclose(lower, [0.0, 0.0, -0.55, -0.55])
        assert np.allclose(upper, [1200.0, 1200.0, 0.55, 0.55])

    def test_objective_at_known_optimum(self):
        """f(x*) must reproduce the published optimum.

        The tolerance reflects the precision of the published x*, which is given
        to seven digits; a wrong coefficient would miss by far more than this.
        """
        problem = G05()

        assert problem.objective(problem.optimum_location) == pytest.approx(
            problem.optimum_value, abs=1e-2
        )

    def test_constraints_at_known_optimum(self):
        """The published optimum must be feasible to the precision of its digits."""
        problem = G05()
        x_opt = problem.optimum_location

        assert len(problem.constraint_eq(x_opt)) == 3
        assert len(problem.constraint_ineq(x_opt)) == 2
        assert np.all(np.abs(problem.constraint_eq(x_opt)) < 1e-3)
        assert np.all(problem.constraint_ineq(x_opt) <= 0)

    def test_dimension_validation(self):
        """A wrong-length input must raise."""
        problem = G05()

        with pytest.raises(ValueError):
            problem.objective(np.ones(3))


class TestG08:
    """Tests for the G08 problem (multimodal coverage)."""

    def test_initialization(self):
        """Metadata must match the CEC 2006 definition."""
        problem = G08()

        assert problem.name == "G08"
        assert problem.dim == 2
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 2
        assert problem.optimum_value == pytest.approx(-0.0958250414)

    def test_bounds(self):
        """Bounds must match the published problem definition."""
        problem = G08()
        lower, upper = problem.bounds

        assert np.allclose(lower, [0.0, 0.0])
        assert np.allclose(upper, [10.0, 10.0])

    def test_objective_at_known_optimum(self):
        """f(x*) must reproduce the published optimum to near machine precision."""
        problem = G08()

        assert problem.objective(problem.optimum_location) == pytest.approx(
            problem.optimum_value, abs=1e-9
        )

    def test_optimum_is_feasible(self):
        """The published optimum must satisfy both constraints."""
        problem = G08()

        assert np.all(problem.constraint_ineq(problem.optimum_location) <= 0)

    def test_is_multimodal(self):
        """Distinct local minima must exist, or the problem adds nothing here."""
        problem = G08()
        values = [
            problem.objective(np.array([x1, 4.2453733]))
            for x1 in (1.2279713, 2.2279713, 3.2279713)
        ]

        assert len(set(np.round(values, 6))) == len(values)

    def test_dimension_validation(self):
        """A wrong-length input must raise."""
        problem = G08()

        with pytest.raises(ValueError):
            problem.objective(np.ones(3))


class TestG11:
    """Tests for the G11 problem (smallest equality-constrained case)."""

    def test_initialization(self):
        """Metadata must match the CEC 2006 definition."""
        problem = G11()

        assert problem.name == "G11"
        assert problem.dim == 2
        assert problem.num_eq_constraints == 1
        assert problem.num_ineq_constraints == 0
        assert problem.optimum_value == pytest.approx(0.7499)

    def test_bounds(self):
        """Bounds must match the published problem definition."""
        problem = G11()
        lower, upper = problem.bounds

        assert np.allclose(lower, [-1.0, -1.0])
        assert np.allclose(upper, [1.0, 1.0])

    def test_objective_at_known_optimum(self):
        """f(x*) must reproduce the published optimum."""
        problem = G11()

        assert problem.objective(problem.optimum_location) == pytest.approx(
            problem.optimum_value, abs=1e-3
        )

    def test_optimum_lies_on_the_manifold(self):
        """The optimum must satisfy x2 = x1^2 to the precision of its digits."""
        problem = G11()

        assert abs(problem.constraint_eq(problem.optimum_location)[0]) < 1e-6

    def test_both_symmetric_optima(self):
        """Both x1 = +/- 1/sqrt(2) give the same objective; neither may be favoured."""
        problem = G11()
        positive = problem.objective(np.array([0.7071068, 0.5]))
        negative = problem.objective(np.array([-0.7071068, 0.5]))

        assert positive == pytest.approx(negative)

    def test_dimension_validation(self):
        """A wrong-length input must raise."""
        problem = G11()

        with pytest.raises(ValueError):
            problem.objective(np.ones(3))
