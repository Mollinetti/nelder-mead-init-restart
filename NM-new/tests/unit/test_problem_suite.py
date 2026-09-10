"""
Unit tests for OptimizationProblem base class.

Tests cover:
- Problem initialization and metadata storage
- Constraint interface (objective, equality, inequality)
- Feasibility checking
- Violation computation
- Input validation and error handling
"""

import pytest
import numpy as np
from nelder_mead.benchmarks.problem_suite import OptimizationProblem


class SimpleSphere(OptimizationProblem):
    """Simple sphere function for testing."""

    def __init__(self, dim=5):
        super().__init__(
            name="SimpleSphere",
            dim=dim,
            bounds=(np.full(dim, -10.0), np.full(dim, 10.0)),
            optimum_value=0.0,
            optimum_location=np.zeros(dim),
        )

    def objective(self, x):
        return np.sum(x**2)


class ConstrainedProblem(OptimizationProblem):
    """Problem with both equality and inequality constraints for testing."""

    def __init__(self):
        super().__init__(
            name="ConstrainedProblem",
            dim=3,
            bounds=(np.array([-5.0, -5.0, -5.0]), np.array([5.0, 5.0, 5.0])),
            optimum_value=None,
            optimum_location=None,
            num_eq_constraints=1,
            num_ineq_constraints=2,
        )

    def objective(self, x):
        return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

    def constraint_eq(self, x):
        # Equality: x[0] + x[1] + x[2] = 1
        return np.array([x[0] + x[1] + x[2] - 1.0])

    def constraint_ineq(self, x):
        # Inequality 1: x[0] - x[1] <= 0
        # Inequality 2: x[1] - x[2] <= 0
        return np.array([x[0] - x[1], x[1] - x[2]])


class TestOptimizationProblemInitialization:
    """Test problem initialization and metadata storage."""

    def test_basic_initialization(self):
        """Test basic problem initialization with required parameters."""
        problem = SimpleSphere(dim=5)

        assert problem.name == "SimpleSphere"
        assert problem.dim == 5
        assert len(problem.bounds[0]) == 5
        assert len(problem.bounds[1]) == 5
        assert np.allclose(problem.bounds[0], -10.0)
        assert np.allclose(problem.bounds[1], 10.0)
        assert problem.optimum_value == 0.0
        assert np.allclose(problem.optimum_location, 0.0)
        assert problem.num_eq_constraints == 0
        assert problem.num_ineq_constraints == 0

    def test_initialization_with_constraints(self):
        """Test initialization with constraint counts."""
        problem = ConstrainedProblem()

        assert problem.dim == 3
        assert problem.num_eq_constraints == 1
        assert problem.num_ineq_constraints == 2

    def test_initialization_without_optimum(self):
        """Test initialization when optimum is unknown."""

        class TestProblem(OptimizationProblem):
            def objective(self, x):
                return np.sum(x**2)

        problem = TestProblem(
            name="TestProblem",
            dim=3,
            bounds=(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
            optimum_value=None,
            optimum_location=None,
        )

        assert problem.optimum_value is None
        assert problem.optimum_location is None

    def test_bounds_stored_as_numpy_arrays(self):
        """Test that bounds are converted to numpy arrays."""
        problem = SimpleSphere(dim=3)

        assert isinstance(problem.bounds[0], np.ndarray)
        assert isinstance(problem.bounds[1], np.ndarray)
        assert problem.bounds[0].dtype == np.float64
        assert problem.bounds[1].dtype == np.float64


class TestOptimizationProblemValidation:
    """Test input validation and error handling."""

    def test_invalid_bounds_format(self):
        """Test that invalid bounds format raises error."""
        with pytest.raises(ValueError, match="bounds must be a tuple"):

            class TestProblem(OptimizationProblem):
                def objective(self, x):
                    return 0.0

            TestProblem(
                name="Test",
                dim=3,
                bounds=[np.zeros(3), np.ones(3)],  # List instead of tuple
                optimum_value=None,
                optimum_location=None,
            )

    def test_bounds_dimension_mismatch(self):
        """Test that bounds dimension mismatch raises error."""
        with pytest.raises(
            ValueError, match="Bounds dimensions.*must match problem dimension"
        ):

            class TestProblem(OptimizationProblem):
                def objective(self, x):
                    return 0.0

            TestProblem(
                name="Test",
                dim=5,
                bounds=(np.zeros(3), np.ones(3)),  # Wrong dimension
                optimum_value=None,
                optimum_location=None,
            )

    def test_inconsistent_bounds(self):
        """Test that lower >= upper bounds raises error."""
        with pytest.raises(
            ValueError, match="Lower bounds must be strictly less than upper bounds"
        ):

            class TestProblem(OptimizationProblem):
                def objective(self, x):
                    return 0.0

            TestProblem(
                name="Test",
                dim=3,
                bounds=(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([0.0, 2.0, 4.0]),
                ),  # First lower > upper
                optimum_value=None,
                optimum_location=None,
            )

    def test_optimum_location_dimension_mismatch(self):
        """Test that optimum location dimension mismatch raises error."""
        with pytest.raises(
            ValueError, match="Optimum location dimension.*must match problem dimension"
        ):

            class TestProblem(OptimizationProblem):
                def objective(self, x):
                    return 0.0

            TestProblem(
                name="Test",
                dim=5,
                bounds=(np.zeros(5), np.ones(5)),
                optimum_value=0.0,
                optimum_location=np.zeros(3),  # Wrong dimension
            )


class TestObjectiveFunction:
    """Test objective function interface."""

    def test_objective_evaluation(self):
        """Test that objective function can be evaluated."""
        problem = SimpleSphere(dim=3)
        x = np.array([1.0, 2.0, 3.0])

        result = problem.objective(x)
        expected = 1.0 + 4.0 + 9.0  # 1^2 + 2^2 + 3^2

        assert np.isclose(result, expected)

    def test_objective_at_optimum(self):
        """Test that objective at known optimum matches optimum value."""
        problem = SimpleSphere(dim=5)

        result = problem.objective(problem.optimum_location)

        assert np.isclose(result, problem.optimum_value)


class TestConstraintInterface:
    """Test constraint function interfaces."""

    def test_default_no_equality_constraints(self):
        """Test that default implementation returns empty array for equality constraints."""
        problem = SimpleSphere(dim=3)
        x = np.array([1.0, 2.0, 3.0])

        eq_violations = problem.constraint_eq(x)

        assert isinstance(eq_violations, np.ndarray)
        assert len(eq_violations) == 0

    def test_default_no_inequality_constraints(self):
        """Test that default implementation returns empty array for inequality constraints."""
        problem = SimpleSphere(dim=3)
        x = np.array([1.0, 2.0, 3.0])

        ineq_violations = problem.constraint_ineq(x)

        assert isinstance(ineq_violations, np.ndarray)
        assert len(ineq_violations) == 0

    def test_equality_constraint_evaluation(self):
        """Test evaluation of equality constraints."""
        problem = ConstrainedProblem()
        x = np.array([0.5, 0.3, 0.2])  # Sum = 1.0, satisfies equality

        eq_violations = problem.constraint_eq(x)

        assert len(eq_violations) == 1
        assert np.isclose(eq_violations[0], 0.0, atol=1e-10)

    def test_inequality_constraint_evaluation(self):
        """Test evaluation of inequality constraints."""
        problem = ConstrainedProblem()
        x = np.array([1.0, 2.0, 3.0])  # x[0] < x[1] < x[2], satisfies inequalities

        ineq_violations = problem.constraint_ineq(x)

        assert len(ineq_violations) == 2
        assert ineq_violations[0] < 0  # x[0] - x[1] = 1 - 2 = -1 < 0
        assert ineq_violations[1] < 0  # x[1] - x[2] = 2 - 3 = -1 < 0


class TestFeasibilityChecking:
    """Test feasibility checking functionality."""

    def test_feasible_unconstrained_solution(self):
        """Test that feasible unconstrained solution is recognized."""
        problem = SimpleSphere(dim=3)
        x = np.array([0.0, 0.0, 0.0])  # Within bounds, no constraints

        assert problem.is_feasible(x)

    def test_infeasible_lower_bound_violation(self):
        """Test that lower bound violation is detected."""
        problem = SimpleSphere(dim=3)
        x = np.array([-11.0, 0.0, 0.0])  # Below lower bound of -10

        assert not problem.is_feasible(x)

    def test_infeasible_upper_bound_violation(self):
        """Test that upper bound violation is detected."""
        problem = SimpleSphere(dim=3)
        x = np.array([0.0, 11.0, 0.0])  # Above upper bound of 10

        assert not problem.is_feasible(x)

    def test_feasible_constrained_solution(self):
        """Test that feasible constrained solution is recognized."""
        problem = ConstrainedProblem()
        x = np.array([0.2, 0.3, 0.5])  # Sum = 1.0, x[0] < x[1] < x[2]

        assert problem.is_feasible(x)

    def test_infeasible_equality_constraint_violation(self):
        """Test that equality constraint violation is detected."""
        problem = ConstrainedProblem()
        x = np.array([1.0, 1.0, 1.0])  # Sum = 3.0, violates equality

        assert not problem.is_feasible(x)

    def test_infeasible_inequality_constraint_violation(self):
        """Test that inequality constraint violation is detected."""
        problem = ConstrainedProblem()
        x = np.array([0.5, 0.3, 0.2])  # Sum = 1.0 (OK), but x[0] > x[1] (violates ineq)

        assert not problem.is_feasible(x)

    def test_feasibility_with_tolerance(self):
        """Test that feasibility checking respects tolerance for equality constraints."""
        problem = ConstrainedProblem()
        x = np.array([0.333, 0.333, 0.333])  # Sum ≈ 0.999, close to 1.0

        # Should be feasible with default tolerance
        assert problem.is_feasible(x, tol=1e-2)

        # Should be infeasible with very strict tolerance
        assert not problem.is_feasible(x, tol=1e-10)


class TestViolationComputation:
    """Test constraint violation computation."""

    def test_zero_violation_for_feasible_solution(self):
        """Test that feasible solution has zero violation."""
        problem = SimpleSphere(dim=3)
        x = np.array([0.0, 0.0, 0.0])

        violation = problem.compute_violation(x)

        assert np.isclose(violation, 0.0)

    def test_lower_bound_violation_computation(self):
        """Test computation of lower bound violations."""
        problem = SimpleSphere(dim=3)
        x = np.array([-12.0, 0.0, 0.0])  # 2 units below lower bound of -10

        violation = problem.compute_violation(x)

        assert np.isclose(violation, 2.0)

    def test_upper_bound_violation_computation(self):
        """Test computation of upper bound violations."""
        problem = SimpleSphere(dim=3)
        x = np.array([0.0, 13.0, 0.0])  # 3 units above upper bound of 10

        violation = problem.compute_violation(x)

        assert np.isclose(violation, 3.0)

    def test_multiple_bound_violations(self):
        """Test computation of multiple bound violations."""
        problem = SimpleSphere(dim=3)
        x = np.array([-12.0, 13.0, 0.0])  # 2 below lower, 3 above upper

        violation = problem.compute_violation(x)

        assert np.isclose(violation, 5.0)  # 2 + 3

    def test_equality_constraint_violation_computation(self):
        """Test computation of equality constraint violations."""
        problem = ConstrainedProblem()
        x = np.array([1.0, 1.0, 1.0])  # Sum = 3.0, violation = |3 - 1| = 2

        violation = problem.compute_violation(x)

        assert np.isclose(violation, 2.0)

    def test_inequality_constraint_violation_computation(self):
        """Test computation of inequality constraint violations."""
        problem = ConstrainedProblem()
        x = np.array([2.0, 1.0, 0.0])  # x[0] - x[1] = 1 > 0, x[1] - x[2] = 1 > 0

        violation = problem.compute_violation(x)

        # Equality violation: |2 + 1 + 0 - 1| = 2
        # Inequality violations: max(0, 1) + max(0, 1) = 2
        # Total: 2 + 2 = 4
        assert np.isclose(violation, 4.0)

    def test_combined_violations(self):
        """Test computation of combined bound, equality, and inequality violations."""
        problem = ConstrainedProblem()
        x = np.array(
            [6.0, 5.0, 4.0]
        )  # Above upper bound, violates equality and inequalities

        violation = problem.compute_violation(x)

        # Bound violations: (6-5) + (5-5) + (4-5) = 1 + 0 + 0 = 1
        # Equality violation: |6 + 5 + 4 - 1| = 14
        # Inequality violations: max(0, 6-5) + max(0, 5-4) = 1 + 1 = 2
        # Total: 1 + 14 + 2 = 17
        assert np.isclose(violation, 17.0)


class TestStringRepresentation:
    """Test string representation methods."""

    def test_str_representation(self):
        """Test __str__ method."""
        problem = ConstrainedProblem()

        result = str(problem)

        assert "ConstrainedProblem" in result
        assert "dim=3" in result
        assert "eq_constraints=1" in result
        assert "ineq_constraints=2" in result

    def test_repr_representation(self):
        """Test __repr__ method."""
        problem = SimpleSphere(dim=5)

        result = repr(problem)

        assert "OptimizationProblem" in result
        assert "name='SimpleSphere'" in result
        assert "dim=5" in result
        assert "optimum_value=0.0" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
