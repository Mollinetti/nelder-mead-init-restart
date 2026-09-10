"""
Unit tests for simplex initialization strategies.

Tests specific examples, edge cases, and error conditions for all initialization methods.
"""

import numpy as np
import pytest
from nelder_mead.initialization import SimplexInitializer


class TestUniformInitialization:
    """Tests for uniform random initialization."""

    def test_uniform_generates_correct_shape(self):
        """Uniform initialization should generate correct number of solutions."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        rng = np.random.default_rng(42)

        solutions = SimplexInitializer.uniform(3, lower, upper, 2, rng)

        assert solutions.shape == (3, 2)

    def test_uniform_respects_bounds(self):
        """All uniform solutions should be within bounds."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(42)

        solutions = SimplexInitializer.uniform(10, lower, upper, 3, rng)

        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

    def test_uniform_is_reproducible(self):
        """Uniform initialization with same seed should produce same results."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        rng1 = np.random.default_rng(42)
        solutions1 = SimplexInitializer.uniform(3, lower, upper, 2, rng1)

        rng2 = np.random.default_rng(42)
        solutions2 = SimplexInitializer.uniform(3, lower, upper, 2, rng2)

        np.testing.assert_array_equal(solutions1, solutions2)


class TestGaussianInitialization:
    """Tests for Gaussian random initialization."""

    def test_gaussian_generates_correct_shape(self):
        """Gaussian initialization should generate correct number of solutions."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        center = np.array([0.5, 0.5])
        rng = np.random.default_rng(42)

        solutions = SimplexInitializer.gaussian(3, lower, upper, 2, rng, center)

        assert solutions.shape == (3, 2)

    def test_gaussian_respects_bounds(self):
        """All Gaussian solutions should be clipped to bounds."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        center = np.array([0.5, 0.5])
        rng = np.random.default_rng(42)

        solutions = SimplexInitializer.gaussian(
            10, lower, upper, 2, rng, center, sigma=2.0
        )

        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

    def test_gaussian_uses_random_center_if_none(self):
        """Gaussian should use random center if none provided."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        rng = np.random.default_rng(42)

        solutions = SimplexInitializer.gaussian(3, lower, upper, 2, rng)

        assert solutions.shape == (3, 2)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

    def test_gaussian_is_reproducible(self):
        """Gaussian initialization with same seed should produce same results."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        center = np.array([0.5, 0.5])

        rng1 = np.random.default_rng(42)
        solutions1 = SimplexInitializer.gaussian(3, lower, upper, 2, rng1, center)

        rng2 = np.random.default_rng(42)
        solutions2 = SimplexInitializer.gaussian(3, lower, upper, 2, rng2, center)

        np.testing.assert_array_equal(solutions1, solutions2)


class TestSpendleySimplex:
    """Tests for Spendley regular simplex initialization."""

    def test_spendley_generates_correct_shape(self):
        """Spendley simplex should generate dim+1 solutions."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        simplex = SimplexInitializer.spendley_simplex(3, lower, upper, 2)

        assert simplex.shape == (3, 2)

    def test_spendley_requires_dim_plus_one(self):
        """Spendley simplex should raise error if num_solutions != dim+1."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="requires num_solutions = dim \\+ 1"):
            SimplexInitializer.spendley_simplex(5, lower, upper, 2)

    def test_spendley_respects_bounds(self):
        """All Spendley simplex vertices should be within bounds."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        simplex = SimplexInitializer.spendley_simplex(4, lower, upper, 3)

        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_spendley_uses_center_if_no_x0(self):
        """Spendley should use center of bounds if no x0 provided."""
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 2.0])

        simplex = SimplexInitializer.spendley_simplex(3, lower, upper, 2)

        # First vertex should be at or near center
        expected_center = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(simplex[0], expected_center, decimal=1)

    def test_spendley_uses_provided_x0(self):
        """Spendley should use provided starting point."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        x0 = np.array([0.3, 0.7])

        simplex = SimplexInitializer.spendley_simplex(3, lower, upper, 2, x0)

        # First vertex should be x0 (or clipped to bounds)
        assert np.allclose(simplex[0], x0, atol=0.1)

    def test_spendley_has_regular_geometry(self):
        """Spendley simplex should have approximately equal edge lengths from first vertex."""
        lower = np.array([-10.0, -10.0])
        upper = np.array([10.0, 10.0])
        x0 = np.array([0.0, 0.0])

        simplex = SimplexInitializer.spendley_simplex(3, lower, upper, 2, x0)

        # Compute distances from first vertex to others
        dist1 = np.linalg.norm(simplex[1] - simplex[0])
        dist2 = np.linalg.norm(simplex[2] - simplex[0])

        # Distances should be approximately equal
        assert np.abs(dist1 - dist2) < 0.1


class TestPfefferSimplex:
    """Tests for Pfeffer simplex initialization."""

    def test_pfeffer_generates_correct_shape(self):
        """Pfeffer simplex should generate dim+1 solutions."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        simplex = SimplexInitializer.pfeffer_simplex(3, lower, upper, 2)

        assert simplex.shape == (3, 2)

    def test_pfeffer_requires_dim_plus_one(self):
        """Pfeffer simplex should raise error if num_solutions != dim+1."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="requires num_solutions = dim \\+ 1"):
            SimplexInitializer.pfeffer_simplex(5, lower, upper, 2)

    def test_pfeffer_respects_bounds(self):
        """All Pfeffer simplex vertices should be within bounds."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        simplex = SimplexInitializer.pfeffer_simplex(4, lower, upper, 3)

        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_pfeffer_uses_center_if_no_x0(self):
        """Pfeffer should use center of bounds if no x0 provided."""
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 2.0])

        simplex = SimplexInitializer.pfeffer_simplex(3, lower, upper, 2)

        # First vertex should be at center
        expected_center = np.array([1.0, 1.0])
        np.testing.assert_array_equal(simplex[0], expected_center)

    def test_pfeffer_scaling_for_nonzero_coordinates(self):
        """Pfeffer should scale by 1.05 for non-zero coordinates."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        x0 = np.array([2.0, 3.0])

        simplex = SimplexInitializer.pfeffer_simplex(3, lower, upper, 2, x0)

        # Second vertex should have first coordinate scaled by 1.05
        assert np.isclose(simplex[1, 0], 2.0 * 1.05)
        assert simplex[1, 1] == 3.0

        # Third vertex should have second coordinate scaled by 1.05
        assert simplex[2, 0] == 2.0
        assert np.isclose(simplex[2, 1], 3.0 * 1.05)

    def test_pfeffer_scaling_for_zero_coordinates(self):
        """Pfeffer should use 0.00025 for zero coordinates."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        x0 = np.array([0.0, 0.0])

        simplex = SimplexInitializer.pfeffer_simplex(3, lower, upper, 2, x0)

        # Second vertex should have first coordinate set to 0.00025
        assert simplex[1, 0] == 0.00025
        assert simplex[1, 1] == 0.0

        # Third vertex should have second coordinate set to 0.00025
        assert simplex[2, 0] == 0.0
        assert simplex[2, 1] == 0.00025


class TestAdaptiveSimplex:
    """Tests for adaptive simplex initialization."""

    def test_adaptive_generates_correct_shape(self):
        """Adaptive simplex should generate dim+1 solutions."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        simplex = SimplexInitializer.adaptive_simplex(3, lower, upper, 2)

        assert simplex.shape == (3, 2)

    def test_adaptive_requires_dim_plus_one(self):
        """Adaptive simplex should raise error if num_solutions != dim+1."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="requires num_solutions = dim \\+ 1"):
            SimplexInitializer.adaptive_simplex(5, lower, upper, 2)

    def test_adaptive_respects_bounds(self):
        """All adaptive simplex vertices should be within bounds."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        simplex = SimplexInitializer.adaptive_simplex(4, lower, upper, 3)

        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_adaptive_uses_center_if_no_x0(self):
        """Adaptive should use center of bounds if no x0 provided."""
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 2.0])

        simplex = SimplexInitializer.adaptive_simplex(3, lower, upper, 2)

        # First vertex should be at center
        expected_center = np.array([1.0, 1.0])
        np.testing.assert_array_equal(simplex[0], expected_center)

    def test_adaptive_step_size_for_small_x0(self):
        """Adaptive should use sigma=1 for small x0."""
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        x0 = np.array([0.1, 0.2])  # ||x0||_inf = 0.2 < 1

        simplex = SimplexInitializer.adaptive_simplex(3, lower, upper, 2, x0)

        # Step size should be 1.0 (max of ||x0||_inf and 1)
        assert np.isclose(simplex[1, 0], x0[0] + 1.0)
        assert simplex[1, 1] == x0[1]

    def test_adaptive_step_size_for_medium_x0(self):
        """Adaptive should use sigma=||x0||_inf for medium x0."""
        lower = np.array([0.0, 0.0])
        upper = np.array([20.0, 20.0])
        x0 = np.array([3.0, 5.0])  # ||x0||_inf = 5.0

        simplex = SimplexInitializer.adaptive_simplex(3, lower, upper, 2, x0)

        # Step size should be 5.0 (||x0||_inf)
        assert np.isclose(simplex[1, 0], x0[0] + 5.0)
        assert simplex[1, 1] == x0[1]

    def test_adaptive_step_size_capped_at_10(self):
        """Adaptive should cap sigma at 10 for large x0."""
        lower = np.array([0.0, 0.0])
        upper = np.array([100.0, 100.0])
        x0 = np.array([50.0, 60.0])  # ||x0||_inf = 60.0 > 10

        simplex = SimplexInitializer.adaptive_simplex(3, lower, upper, 2, x0)

        # Step size should be capped at 10.0
        assert np.isclose(simplex[1, 0], x0[0] + 10.0)
        assert simplex[1, 1] == x0[1]


class TestInitializeDispatcher:
    """Tests for the initialize dispatcher method."""

    def test_initialize_uniform(self):
        """Initialize should dispatch to uniform method."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        rng = np.random.default_rng(42)

        solutions = SimplexInitializer.initialize(
            "uniform", 3, lower, upper, 2, rng=rng
        )

        assert solutions.shape == (3, 2)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

    def test_initialize_gaussian(self):
        """Initialize should dispatch to gaussian method."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        rng = np.random.default_rng(42)

        solutions = SimplexInitializer.initialize(
            "gaussian", 3, lower, upper, 2, rng=rng
        )

        assert solutions.shape == (3, 2)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

    def test_initialize_spendley(self):
        """Initialize should dispatch to spendleySimplex method."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        simplex = SimplexInitializer.initialize("spendleySimplex", 3, lower, upper, 2)

        assert simplex.shape == (3, 2)

    def test_initialize_pfeffer(self):
        """Initialize should dispatch to pfefferSimplex method."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        simplex = SimplexInitializer.initialize("pfefferSimplex", 3, lower, upper, 2)

        assert simplex.shape == (3, 2)

    def test_initialize_adaptive(self):
        """Initialize should dispatch to adaptiveSimplex method."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        simplex = SimplexInitializer.initialize("adaptiveSimplex", 3, lower, upper, 2)

        assert simplex.shape == (3, 2)

    def test_initialize_invalid_method(self):
        """Initialize should raise error for invalid method."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="is not a valid initialization method"):
            SimplexInitializer.initialize("invalid_method", 3, lower, upper, 2)

    def test_initialize_error_message_lists_valid_methods(self):
        """Error message should list all valid methods."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        with pytest.raises(ValueError, match="uniform.*gaussian.*spendleySimplex"):
            SimplexInitializer.initialize("bad_method", 3, lower, upper, 2)


class TestBoundEnforcement:
    """Tests for bound enforcement in initialization methods."""

    def test_all_methods_enforce_bounds(self):
        """All initialization methods should enforce bounds."""
        lower = np.array([0.0, 0.0])
        upper = np.array([0.5, 0.5])  # Tight bounds

        methods = [
            "uniform",
            "gaussian",
            "spendleySimplex",
            "pfefferSimplex",
            "adaptiveSimplex",
        ]

        for method in methods:
            if method in ["uniform", "gaussian"]:
                rng = np.random.default_rng(42)
                solutions = SimplexInitializer.initialize(
                    method, 3, lower, upper, 2, rng=rng
                )
            else:
                solutions = SimplexInitializer.initialize(method, 3, lower, upper, 2)

            assert np.all(solutions >= lower), f"{method} violated lower bounds"
            assert np.all(solutions <= upper), f"{method} violated upper bounds"

    def test_gaussian_with_out_of_bounds_center(self):
        """Gaussian initialization should clip out-of-bounds points."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        center = np.array([0.5, 0.5])
        rng = np.random.default_rng(42)

        # Use large sigma to generate out-of-bounds points
        solutions = SimplexInitializer.gaussian(
            10, lower, upper, 2, rng, center, sigma=5.0
        )

        # All solutions should be clipped to bounds
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

    def test_spendley_with_out_of_bounds_x0(self):
        """Spendley simplex should clip vertices that go out of bounds."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        x0 = np.array([0.9, 0.9])  # Near upper bound

        simplex = SimplexInitializer.spendley_simplex(3, lower, upper, 2, x0)

        # All vertices should be within bounds
        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_pfeffer_with_out_of_bounds_x0(self):
        """Pfeffer simplex should clip vertices that go out of bounds."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        x0 = np.array([0.98, 0.98])  # Very near upper bound

        simplex = SimplexInitializer.pfeffer_simplex(3, lower, upper, 2, x0)

        # All vertices should be within bounds
        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_adaptive_with_out_of_bounds_x0(self):
        """Adaptive simplex should clip vertices that go out of bounds."""
        lower = np.array([0.0, 0.0])
        upper = np.array([5.0, 5.0])
        x0 = np.array([4.5, 4.5])  # Near upper bound

        simplex = SimplexInitializer.adaptive_simplex(3, lower, upper, 2, x0)

        # All vertices should be within bounds
        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)


class TestEdgeCases:
    """Tests for edge cases in initialization."""

    def test_zero_width_bounds_single_dimension(self):
        """Initialization should reject zero-width bounds in any dimension."""
        lower = np.array([0.5, 0.0])
        upper = np.array([0.5, 1.0])  # First dimension has zero width

        # Uniform initialization - generates points at the boundary
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.uniform(3, lower, upper, 2, rng)
        assert np.all(solutions[:, 0] == 0.5)  # First dimension fixed
        assert np.all(solutions[:, 1] >= 0.0) and np.all(solutions[:, 1] <= 1.0)

        # Gaussian initialization - should raise error when enforcing bounds
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Lower bounds must be less than upper"):
            SimplexInitializer.gaussian(3, lower, upper, 2, rng)

        # Spendley simplex - should raise error when enforcing bounds
        with pytest.raises(ValueError, match="Lower bounds must be less than upper"):
            SimplexInitializer.spendley_simplex(3, lower, upper, 2)

        # Pfeffer simplex - should raise error when enforcing bounds
        with pytest.raises(ValueError, match="Lower bounds must be less than upper"):
            SimplexInitializer.pfeffer_simplex(3, lower, upper, 2)

        # Adaptive simplex - should raise error when enforcing bounds
        with pytest.raises(ValueError, match="Lower bounds must be less than upper"):
            SimplexInitializer.adaptive_simplex(3, lower, upper, 2)

    def test_single_point_bounds(self):
        """Initialization should reject single-point bounds (equal lower and upper)."""
        lower = np.array([0.5, 0.5])
        upper = np.array([0.5, 0.5])  # Single point

        # Uniform initialization - generates the single point
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.uniform(3, lower, upper, 2, rng)
        np.testing.assert_array_equal(solutions, np.array([[0.5, 0.5]] * 3))

        # Gaussian initialization - should raise error when enforcing bounds
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Lower bounds must be less than upper"):
            SimplexInitializer.gaussian(3, lower, upper, 2, rng)

        # Spendley simplex - should raise error when enforcing bounds
        with pytest.raises(ValueError, match="Lower bounds must be less than upper"):
            SimplexInitializer.spendley_simplex(3, lower, upper, 2)

        # Pfeffer simplex - should raise error when enforcing bounds
        with pytest.raises(ValueError, match="Lower bounds must be less than upper"):
            SimplexInitializer.pfeffer_simplex(3, lower, upper, 2)

        # Adaptive simplex - should raise error when enforcing bounds
        with pytest.raises(ValueError, match="Lower bounds must be less than upper"):
            SimplexInitializer.adaptive_simplex(3, lower, upper, 2)

    def test_very_small_bounds(self):
        """Initialization should handle very small bound ranges."""
        lower = np.array([0.0, 0.0])
        upper = np.array([1e-10, 1e-10])  # Very small bounds

        # Uniform initialization
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.uniform(3, lower, upper, 2, rng)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

        # Gaussian initialization
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.gaussian(3, lower, upper, 2, rng)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

    def test_very_large_bounds(self):
        """Initialization should handle very large bound ranges."""
        lower = np.array([-1e10, -1e10])
        upper = np.array([1e10, 1e10])  # Very large bounds

        # Uniform initialization
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.uniform(3, lower, upper, 2, rng)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

        # Gaussian initialization
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.gaussian(3, lower, upper, 2, rng)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

    def test_negative_bounds(self):
        """Initialization should handle negative bounds."""
        lower = np.array([-5.0, -10.0])
        upper = np.array([-1.0, -2.0])  # All negative

        # Uniform initialization
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.uniform(3, lower, upper, 2, rng)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

        # Spendley simplex
        simplex = SimplexInitializer.spendley_simplex(3, lower, upper, 2)
        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_asymmetric_bounds(self):
        """Initialization should handle asymmetric bounds."""
        lower = np.array([-100.0, 0.0])
        upper = np.array([1.0, 1000.0])  # Very different ranges

        # Uniform initialization
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.uniform(3, lower, upper, 2, rng)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

        # Spendley simplex
        simplex = SimplexInitializer.spendley_simplex(3, lower, upper, 2)
        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_high_dimensional_initialization(self):
        """Initialization should work for high-dimensional problems."""
        dim = 50
        lower = np.zeros(dim)
        upper = np.ones(dim)

        # Uniform initialization
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.uniform(dim + 1, lower, upper, dim, rng)
        assert solutions.shape == (dim + 1, dim)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

        # Spendley simplex
        simplex = SimplexInitializer.spendley_simplex(dim + 1, lower, upper, dim)
        assert simplex.shape == (dim + 1, dim)
        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)

    def test_one_dimensional_initialization(self):
        """Initialization should work for 1D problems."""
        lower = np.array([0.0])
        upper = np.array([1.0])

        # Uniform initialization
        rng = np.random.default_rng(42)
        solutions = SimplexInitializer.uniform(2, lower, upper, 1, rng)
        assert solutions.shape == (2, 1)
        assert np.all(solutions >= lower)
        assert np.all(solutions <= upper)

        # Spendley simplex
        simplex = SimplexInitializer.spendley_simplex(2, lower, upper, 1)
        assert simplex.shape == (2, 1)
        assert np.all(simplex >= lower)
        assert np.all(simplex <= upper)
