"""
Unit tests for unconstrained benchmark problems.

Tests verify:
1. Known optima are correctly specified
2. Objective functions compute correct values
3. Bounds are correctly specified
4. Problem metadata is correct
5. Edge cases and error handling
"""

import pytest
import numpy as np
from nelder_mead.benchmarks.unconstrained import (
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
    Griewank,
    Schwefel,
)


class TestSphere:
    """Tests for Sphere function."""

    def test_optimum_value(self):
        """Test that the global minimum is at the origin with value 0."""
        problem = Sphere(dim=5)
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)
        assert np.allclose(f_opt, 0.0, atol=1e-10), "Sphere optimum should be 0"

    def test_optimum_location(self):
        """Test that the optimum location is at the origin."""
        problem = Sphere(dim=5)
        assert np.allclose(
            problem.optimum_location, np.zeros(5)
        ), "Optimum should be at origin"

    def test_positive_values(self):
        """Test that function values are non-negative."""
        problem = Sphere(dim=5)
        x = np.random.randn(5)
        f = problem.objective(x)
        assert f >= 0.0, "Sphere function should be non-negative"

    def test_known_value(self):
        """Test a known function value."""
        problem = Sphere(dim=3)
        x = np.array([1.0, 2.0, 3.0])
        f = problem.objective(x)
        expected = 1.0 + 4.0 + 9.0  # 1^2 + 2^2 + 3^2
        assert np.allclose(f, expected), f"Expected {expected}, got {f}"

    def test_bounds(self):
        """Test that bounds are correctly set."""
        problem = Sphere(dim=5, bounds_range=50.0)
        lower, upper = problem.bounds
        assert np.allclose(lower, -50.0), "Lower bounds should be -50"
        assert np.allclose(upper, 50.0), "Upper bounds should be 50"

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        problem = Sphere(dim=5)
        x = np.array([1.0, 2.0, 3.0])  # Wrong dimension
        with pytest.raises(ValueError, match="Expected 5 dimensions"):
            problem.objective(x)

    def test_no_constraints(self):
        """Test that Sphere has no constraints."""
        problem = Sphere(dim=5)
        x = np.zeros(5)
        assert len(problem.constraint_eq(x)) == 0, "Should have no equality constraints"
        assert (
            len(problem.constraint_ineq(x)) == 0
        ), "Should have no inequality constraints"


class TestRosenbrock:
    """Tests for Rosenbrock function."""

    def test_optimum_value(self):
        """Test that the global minimum is at (1, 1, ..., 1) with value 0."""
        problem = Rosenbrock(dim=5)
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)
        assert np.allclose(f_opt, 0.0, atol=1e-10), "Rosenbrock optimum should be 0"

    def test_optimum_location(self):
        """Test that the optimum location is at (1, 1, ..., 1)."""
        problem = Rosenbrock(dim=5)
        assert np.allclose(
            problem.optimum_location, np.ones(5)
        ), "Optimum should be at (1,1,...,1)"

    def test_positive_values(self):
        """Test that function values are non-negative."""
        problem = Rosenbrock(dim=5)
        x = np.random.randn(5)
        f = problem.objective(x)
        assert f >= 0.0, "Rosenbrock function should be non-negative"

    def test_known_value_2d(self):
        """Test a known function value in 2D."""
        problem = Rosenbrock(dim=2)
        x = np.array([0.0, 0.0])
        f = problem.objective(x)
        # f(0,0) = 100*(0 - 0^2)^2 + (1 - 0)^2 = 0 + 1 = 1
        expected = 1.0
        assert np.allclose(f, expected), f"Expected {expected}, got {f}"

    def test_known_value_3d(self):
        """Test a known function value in 3D."""
        problem = Rosenbrock(dim=3)
        x = np.array([1.0, 1.0, 2.0])
        f = problem.objective(x)
        # f(1,1,2) = 100*(1-1^2)^2 + (1-1)^2 + 100*(2-1^2)^2 + (1-1)^2
        #          = 0 + 0 + 100*1 + 0 = 100
        expected = 100.0
        assert np.allclose(f, expected), f"Expected {expected}, got {f}"

    def test_bounds(self):
        """Test that bounds are correctly set."""
        problem = Rosenbrock(dim=5, bounds_range=10.0)
        lower, upper = problem.bounds
        assert np.allclose(lower, -10.0), "Lower bounds should be -10"
        assert np.allclose(upper, 10.0), "Upper bounds should be 10"

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        problem = Rosenbrock(dim=5)
        x = np.array([1.0, 2.0, 3.0])  # Wrong dimension
        with pytest.raises(ValueError, match="Expected 5 dimensions"):
            problem.objective(x)


class TestRastrigin:
    """Tests for Rastrigin function."""

    def test_optimum_value(self):
        """Test that the global minimum is at the origin with value 0."""
        problem = Rastrigin(dim=5)
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)
        assert np.allclose(f_opt, 0.0, atol=1e-10), "Rastrigin optimum should be 0"

    def test_optimum_location(self):
        """Test that the optimum location is at the origin."""
        problem = Rastrigin(dim=5)
        assert np.allclose(
            problem.optimum_location, np.zeros(5)
        ), "Optimum should be at origin"

    def test_positive_values(self):
        """Test that function values are non-negative."""
        problem = Rastrigin(dim=5)
        x = problem.optimum_location
        f = problem.objective(x)
        assert f >= 0.0, "Rastrigin function should be non-negative at optimum"

    def test_known_value(self):
        """Test a known function value."""
        problem = Rastrigin(dim=2)
        x = np.array([1.0, 1.0])
        f = problem.objective(x)
        # f(1,1) = 10*2 + (1^2 - 10*cos(2π*1)) + (1^2 - 10*cos(2π*1))
        #        = 20 + (1 - 10*1) + (1 - 10*1) = 20 + (-9) + (-9) = 2
        expected = 20.0 + 2 * (1.0 - 10.0 * np.cos(2 * np.pi * 1.0))
        assert np.allclose(f, expected), f"Expected {expected}, got {f}"

    def test_bounds(self):
        """Test that bounds are correctly set."""
        problem = Rastrigin(dim=5, bounds_range=5.12)
        lower, upper = problem.bounds
        assert np.allclose(lower, -5.12), "Lower bounds should be -5.12"
        assert np.allclose(upper, 5.12), "Upper bounds should be 5.12"

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        problem = Rastrigin(dim=5)
        x = np.array([1.0, 2.0, 3.0])  # Wrong dimension
        with pytest.raises(ValueError, match="Expected 5 dimensions"):
            problem.objective(x)


class TestAckley:
    """Tests for Ackley function."""

    def test_optimum_value(self):
        """Test that the global minimum is at the origin with value ~0."""
        problem = Ackley(dim=5)
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)
        assert np.allclose(f_opt, 0.0, atol=1e-10), "Ackley optimum should be ~0"

    def test_optimum_location(self):
        """Test that the optimum location is at the origin."""
        problem = Ackley(dim=5)
        assert np.allclose(
            problem.optimum_location, np.zeros(5)
        ), "Optimum should be at origin"

    def test_positive_values(self):
        """Test that function values are non-negative."""
        problem = Ackley(dim=5)
        x = problem.optimum_location
        f = problem.objective(x)
        assert f >= -1e-10, "Ackley function should be non-negative at optimum"

    def test_known_value(self):
        """Test a known function value."""
        problem = Ackley(dim=2)
        x = np.array([1.0, 1.0])
        f = problem.objective(x)

        # Manual calculation
        a, b, c = 20.0, 0.2, 2.0 * np.pi
        sum_sq = 2.0  # 1^2 + 1^2
        sum_cos = 2 * np.cos(c * 1.0)
        term1 = -a * np.exp(-b * np.sqrt(sum_sq / 2))
        term2 = -np.exp(sum_cos / 2)
        expected = term1 + term2 + a + np.e

        assert np.allclose(f, expected, atol=1e-10), f"Expected {expected}, got {f}"

    def test_bounds(self):
        """Test that bounds are correctly set."""
        problem = Ackley(dim=5, bounds_range=32.768)
        lower, upper = problem.bounds
        assert np.allclose(lower, -32.768), "Lower bounds should be -32.768"
        assert np.allclose(upper, 32.768), "Upper bounds should be 32.768"

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        problem = Ackley(dim=5)
        x = np.array([1.0, 2.0, 3.0])  # Wrong dimension
        with pytest.raises(ValueError, match="Expected 5 dimensions"):
            problem.objective(x)


class TestGriewank:
    """Tests for Griewank function."""

    def test_optimum_value(self):
        """Test that the global minimum is at the origin with value 0."""
        problem = Griewank(dim=5)
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)
        assert np.allclose(f_opt, 0.0, atol=1e-10), "Griewank optimum should be 0"

    def test_optimum_location(self):
        """Test that the optimum location is at the origin."""
        problem = Griewank(dim=5)
        assert np.allclose(
            problem.optimum_location, np.zeros(5)
        ), "Optimum should be at origin"

    def test_positive_values(self):
        """Test that function values are non-negative at optimum."""
        problem = Griewank(dim=5)
        x = problem.optimum_location
        f = problem.objective(x)
        assert f >= -1e-10, "Griewank function should be ~0 at optimum"

    def test_known_value(self):
        """Test a known function value."""
        problem = Griewank(dim=2)
        x = np.array([1.0, 1.0])
        f = problem.objective(x)

        # Manual calculation
        sum_term = (1.0 + 1.0) / 4000.0  # (1^2 + 1^2) / 4000
        prod_term = np.cos(1.0 / np.sqrt(1)) * np.cos(1.0 / np.sqrt(2))
        expected = 1.0 + sum_term - prod_term

        assert np.allclose(f, expected, atol=1e-10), f"Expected {expected}, got {f}"

    def test_bounds(self):
        """Test that bounds are correctly set."""
        problem = Griewank(dim=5, bounds_range=600.0)
        lower, upper = problem.bounds
        assert np.allclose(lower, -600.0), "Lower bounds should be -600"
        assert np.allclose(upper, 600.0), "Upper bounds should be 600"

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        problem = Griewank(dim=5)
        x = np.array([1.0, 2.0, 3.0])  # Wrong dimension
        with pytest.raises(ValueError, match="Expected 5 dimensions"):
            problem.objective(x)


class TestSchwefel:
    """Tests for Schwefel function."""

    def test_optimum_value(self):
        """Test that the global minimum is near 0."""
        problem = Schwefel(dim=5)
        x_opt = problem.optimum_location
        f_opt = problem.objective(x_opt)
        # Schwefel optimum is approximately 0, but not exactly due to numerical precision
        assert np.allclose(f_opt, 0.0, atol=1.0), "Schwefel optimum should be ~0"

    def test_optimum_location(self):
        """Test that the optimum location is at (420.9687, ..., 420.9687)."""
        problem = Schwefel(dim=5)
        expected = np.full(5, 420.9687)
        assert np.allclose(
            problem.optimum_location, expected
        ), "Optimum should be at (420.9687,...)"

    def test_known_value_at_origin(self):
        """Test function value at origin."""
        problem = Schwefel(dim=2)
        x = np.array([0.0, 0.0])
        f = problem.objective(x)
        # f(0,0) = 418.9829*2 - 0 = 837.9658
        expected = 418.9829 * 2
        assert np.allclose(f, expected, atol=1e-4), f"Expected {expected}, got {f}"

    def test_known_value(self):
        """Test a known function value."""
        problem = Schwefel(dim=2)
        x = np.array([100.0, 100.0])
        f = problem.objective(x)
        # f(100,100) = 418.9829*2 - 2*100*sin(sqrt(100))
        expected = 418.9829 * 2 - 2 * 100.0 * np.sin(np.sqrt(100.0))
        assert np.allclose(f, expected, atol=1e-10), f"Expected {expected}, got {f}"

    def test_bounds(self):
        """Test that bounds are correctly set."""
        problem = Schwefel(dim=5, bounds_range=500.0)
        lower, upper = problem.bounds
        assert np.allclose(lower, -500.0), "Lower bounds should be -500"
        assert np.allclose(upper, 500.0), "Upper bounds should be 500"

    def test_dimension_mismatch(self):
        """Test that dimension mismatch raises error."""
        problem = Schwefel(dim=5)
        x = np.array([1.0, 2.0, 3.0])  # Wrong dimension
        with pytest.raises(ValueError, match="Expected 5 dimensions"):
            problem.objective(x)


class TestProblemMetadata:
    """Tests for problem metadata and common properties."""

    def test_all_problems_have_names(self):
        """Test that all problems have proper names."""
        problems = [
            Sphere(dim=5),
            Rosenbrock(dim=5),
            Rastrigin(dim=5),
            Ackley(dim=5),
            Griewank(dim=5),
            Schwefel(dim=5),
        ]

        expected_names = [
            "Sphere",
            "Rosenbrock",
            "Rastrigin",
            "Ackley",
            "Griewank",
            "Schwefel",
        ]

        for problem, expected_name in zip(problems, expected_names):
            assert (
                problem.name == expected_name
            ), f"Problem name should be {expected_name}"

    def test_all_problems_have_correct_dimension(self):
        """Test that all problems have correct dimension."""
        dim = 7
        problems = [
            Sphere(dim=dim),
            Rosenbrock(dim=dim),
            Rastrigin(dim=dim),
            Ackley(dim=dim),
            Griewank(dim=dim),
            Schwefel(dim=dim),
        ]

        for problem in problems:
            assert problem.dim == dim, f"{problem.name} should have dimension {dim}"

    def test_all_problems_have_optimum_value(self):
        """Test that all problems have known optimum values."""
        problems = [
            Sphere(dim=5),
            Rosenbrock(dim=5),
            Rastrigin(dim=5),
            Ackley(dim=5),
            Griewank(dim=5),
            Schwefel(dim=5),
        ]

        for problem in problems:
            assert (
                problem.optimum_value is not None
            ), f"{problem.name} should have optimum value"
            assert problem.optimum_value == 0.0, f"{problem.name} optimum should be 0"

    def test_all_problems_have_optimum_location(self):
        """Test that all problems have known optimum locations."""
        problems = [
            Sphere(dim=5),
            Rosenbrock(dim=5),
            Rastrigin(dim=5),
            Ackley(dim=5),
            Griewank(dim=5),
            Schwefel(dim=5),
        ]

        for problem in problems:
            assert (
                problem.optimum_location is not None
            ), f"{problem.name} should have optimum location"
            assert (
                len(problem.optimum_location) == problem.dim
            ), f"{problem.name} optimum location should have correct dimension"

    def test_all_problems_are_unconstrained(self):
        """Test that all problems have no constraints."""
        problems = [
            Sphere(dim=5),
            Rosenbrock(dim=5),
            Rastrigin(dim=5),
            Ackley(dim=5),
            Griewank(dim=5),
            Schwefel(dim=5),
        ]

        x = np.zeros(5)
        for problem in problems:
            assert (
                problem.num_eq_constraints == 0
            ), f"{problem.name} should have no equality constraints"
            assert (
                problem.num_ineq_constraints == 0
            ), f"{problem.name} should have no inequality constraints"
            assert (
                len(problem.constraint_eq(x)) == 0
            ), f"{problem.name} should return empty eq constraints"
            assert (
                len(problem.constraint_ineq(x)) == 0
            ), f"{problem.name} should return empty ineq constraints"

    def test_all_problems_have_valid_bounds(self):
        """Test that all problems have valid bounds."""
        problems = [
            Sphere(dim=5),
            Rosenbrock(dim=5),
            Rastrigin(dim=5),
            Ackley(dim=5),
            Griewank(dim=5),
            Schwefel(dim=5),
        ]

        for problem in problems:
            lower, upper = problem.bounds
            assert (
                len(lower) == problem.dim
            ), f"{problem.name} lower bounds should match dimension"
            assert (
                len(upper) == problem.dim
            ), f"{problem.name} upper bounds should match dimension"
            assert np.all(
                lower < upper
            ), f"{problem.name} lower bounds should be less than upper bounds"

    def test_optimum_is_feasible(self):
        """Test that the optimum location is within bounds."""
        problems = [
            Sphere(dim=5),
            Rosenbrock(dim=5),
            Rastrigin(dim=5),
            Ackley(dim=5),
            Griewank(dim=5),
            Schwefel(dim=5),
        ]

        for problem in problems:
            x_opt = problem.optimum_location
            lower, upper = problem.bounds
            assert np.all(
                x_opt >= lower
            ), f"{problem.name} optimum should be within lower bounds"
            assert np.all(
                x_opt <= upper
            ), f"{problem.name} optimum should be within upper bounds"
            assert problem.is_feasible(
                x_opt
            ), f"{problem.name} optimum should be feasible"

    def test_string_representation(self):
        """Test string representation of problems."""
        problem = Sphere(dim=5)
        str_repr = str(problem)
        assert "Sphere" in str_repr, "String representation should contain problem name"
        assert "dim=5" in str_repr, "String representation should contain dimension"
