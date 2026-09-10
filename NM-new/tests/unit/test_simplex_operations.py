"""
Unit tests for simplex geometric operations.

These tests verify specific examples and edge cases for simplex transformations.
"""

import numpy as np
import pytest
from nelder_mead.core.simplex_operations import (
    compute_centroid,
    reflect,
    expand,
    contract_outside,
    contract_inside,
    shrink,
    enforce_bounds,
)


class TestComputeCentroid:
    """Tests for centroid computation."""

    def test_centroid_excludes_worst_by_default(self):
        """Test that centroid excludes last vertex by default."""
        simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        centroid = compute_centroid(simplex, exclude_worst=True)
        # Should be mean of first 2 vertices: (0+1)/2, (0+0)/2 = (0.5, 0)
        np.testing.assert_array_almost_equal(centroid, np.array([0.5, 0.0]))

    def test_centroid_includes_all_when_specified(self):
        """Test that centroid includes all vertices when exclude_worst=False."""
        simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        centroid = compute_centroid(simplex, exclude_worst=False)
        # Should be mean of all 3 vertices: (0+1+0)/3, (0+0+1)/3 = (1/3, 1/3)
        np.testing.assert_array_almost_equal(centroid, np.array([1 / 3, 1 / 3]))

    def test_centroid_higher_dimension(self):
        """Test centroid computation in 3D."""
        simplex = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        centroid = compute_centroid(simplex, exclude_worst=True)
        # Mean of first 3 vertices
        expected = np.array([1 / 3, 1 / 3, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected)


class TestReflect:
    """Tests for reflection operation."""

    def test_reflection_standard_coefficient(self):
        """Test reflection with standard coefficient (1.0)."""
        centroid = np.array([0.5, 0.0])
        worst = np.array([0.0, 1.0])
        reflected = reflect(centroid, worst, delta_r=1.0)
        # r = c + 1.0 * (c - w) = [0.5, 0] + [0.5, -1] = [1, -1]
        np.testing.assert_array_almost_equal(reflected, np.array([1.0, -1.0]))

    def test_reflection_opposite_side_of_centroid(self):
        """Test that reflection places point on opposite side of centroid."""
        centroid = np.array([1.0, 1.0])
        worst = np.array([0.0, 0.0])
        reflected = reflect(centroid, worst, delta_r=1.0)
        # r = [1, 1] + 1.0 * ([1, 1] - [0, 0]) = [2, 2]
        np.testing.assert_array_almost_equal(reflected, np.array([2.0, 2.0]))

        # Verify it's on opposite side: (r - c) and (w - c) should point in opposite directions
        r_direction = reflected - centroid
        w_direction = worst - centroid
        dot_product = np.dot(r_direction, w_direction)
        assert dot_product < 0, "Reflected point should be on opposite side of centroid"


class TestExpand:
    """Tests for expansion operation."""

    def test_expansion_standard_coefficient(self):
        """Test expansion with standard coefficient (2.0)."""
        centroid = np.array([0.5, 0.0])
        reflected = np.array([1.0, -1.0])
        expanded = expand(centroid, reflected, delta_e=2.0)
        # e = c + 2.0 * (r - c) = [0.5, 0] + 2.0 * [0.5, -1] = [1.5, -2]
        np.testing.assert_array_almost_equal(expanded, np.array([1.5, -2.0]))

    def test_expansion_extends_beyond_reflection(self):
        """Test that expansion is farther from centroid than reflection."""
        centroid = np.array([0.0, 0.0])
        reflected = np.array([1.0, 0.0])
        expanded = expand(centroid, reflected, delta_e=2.0)

        dist_reflected = np.linalg.norm(reflected - centroid)
        dist_expanded = np.linalg.norm(expanded - centroid)
        assert (
            dist_expanded > dist_reflected
        ), "Expanded point should be farther from centroid"


class TestContractOutside:
    """Tests for outside contraction operation."""

    def test_outside_contraction_standard_coefficient(self):
        """Test outside contraction with standard coefficient (0.5)."""
        centroid = np.array([0.5, 0.0])
        reflected = np.array([1.0, -1.0])
        contracted = contract_outside(centroid, reflected, delta_oc=0.5)
        # oc = c + 0.5 * (r - c) = [0.5, 0] + 0.5 * [0.5, -1] = [0.75, -0.5]
        np.testing.assert_array_almost_equal(contracted, np.array([0.75, -0.5]))

    def test_outside_contraction_closer_to_centroid(self):
        """Test that outside contraction is closer to centroid than reflection."""
        centroid = np.array([0.0, 0.0])
        reflected = np.array([2.0, 0.0])
        contracted = contract_outside(centroid, reflected, delta_oc=0.5)

        dist_reflected = np.linalg.norm(reflected - centroid)
        dist_contracted = np.linalg.norm(contracted - centroid)
        assert (
            dist_contracted < dist_reflected
        ), "Contracted point should be closer to centroid"


class TestContractInside:
    """Tests for inside contraction operation."""

    def test_inside_contraction_standard_coefficient(self):
        """Test inside contraction with standard coefficient (-0.5)."""
        centroid = np.array([0.5, 0.0])
        worst = np.array([0.0, 1.0])
        contracted = contract_inside(centroid, worst, delta_ic=-0.5)
        # ic = c + (-0.5) * (c - w) = [0.5, 0] + (-0.5) * ([0.5, 0] - [0, 1])
        #    = [0.5, 0] + (-0.5) * [0.5, -1] = [0.5, 0] + [-0.25, 0.5] = [0.25, 0.5]
        # which is the midpoint of the centroid and the worst vertex
        np.testing.assert_array_almost_equal(contracted, np.array([0.25, 0.5]))
        np.testing.assert_array_almost_equal(contracted, (centroid + worst) / 2)

    def test_inside_contraction_lies_between_centroid_and_worst(self):
        """The inside contraction point must be inside the simplex.

        Regression test: an earlier implementation used (w - c) instead of
        (c - w), placing the point on the reflection side of the centroid.
        """
        centroid = np.array([0.0, 0.0])
        worst = np.array([2.0, 0.0])
        contracted = contract_inside(centroid, worst, delta_ic=-0.5)

        # Must lie on the segment from centroid to worst
        t = np.dot(contracted - centroid, worst - centroid) / np.dot(
            worst - centroid, worst - centroid
        )
        assert 0.0 < t < 1.0, f"Inside contraction fell outside the segment (t={t})"

    def test_inside_contraction_closer_to_centroid(self):
        """Test that inside contraction is closer to centroid than worst point."""
        centroid = np.array([0.0, 0.0])
        worst = np.array([2.0, 0.0])
        contracted = contract_inside(centroid, worst, delta_ic=-0.5)

        dist_worst = np.linalg.norm(worst - centroid)
        dist_contracted = np.linalg.norm(contracted - centroid)
        assert (
            dist_contracted < dist_worst
        ), "Contracted point should be closer to centroid"


class TestShrink:
    """Tests for shrink operation."""

    def test_shrink_standard_coefficient(self):
        """Test shrink with standard coefficient (0.5)."""
        best = np.array([0.0, 0.0])
        simplex = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        shrunk = shrink(best, simplex, gamma_s=0.5)

        expected = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(shrunk, expected)

    def test_shrink_moves_toward_best(self):
        """Test that all vertices move closer to best point."""
        best = np.array([1.0, 1.0])
        simplex = np.array([[1.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
        shrunk = shrink(best, simplex, gamma_s=0.5)

        # Check that all vertices (except best) are closer to best
        for i in range(1, len(simplex)):
            dist_before = np.linalg.norm(simplex[i] - best)
            dist_after = np.linalg.norm(shrunk[i] - best)
            assert (
                dist_after < dist_before
            ), f"Vertex {i} should be closer to best after shrink"

    def test_shrink_preserves_best(self):
        """Test that best point remains unchanged after shrink."""
        best = np.array([1.0, 1.0])
        simplex = np.array([[1.0, 1.0], [3.0, 1.0], [1.0, 3.0]])
        shrunk = shrink(best, simplex, gamma_s=0.5)

        np.testing.assert_array_almost_equal(shrunk[0], best)


class TestEnforceBounds:
    """Tests for bound enforcement."""

    def test_enforce_bounds_clips_to_boundary(self):
        """Test that points outside bounds are clipped to nearest boundary."""
        point = np.array([1.5, -0.5, 0.5])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        enforced = enforce_bounds(point, lower, upper)
        expected = np.array([1.0, 0.0, 0.5])
        np.testing.assert_array_almost_equal(enforced, expected)

    def test_enforce_bounds_preserves_interior_points(self):
        """Test that points already within bounds are unchanged."""
        point = np.array([0.5, 0.5, 0.5])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        enforced = enforce_bounds(point, lower, upper)
        np.testing.assert_array_almost_equal(enforced, point)

    def test_enforce_bounds_handles_asymmetric_bounds(self):
        """Test bound enforcement with asymmetric bounds."""
        point = np.array([15.0, -15.0])
        lower = np.array([-10.0, -10.0])
        upper = np.array([10.0, 10.0])

        enforced = enforce_bounds(point, lower, upper)
        expected = np.array([10.0, -10.0])
        np.testing.assert_array_almost_equal(enforced, expected)

    def test_enforce_bounds_raises_on_invalid_bounds(self):
        """Test that invalid bounds raise ValueError."""
        point = np.array([0.5])
        lower = np.array([1.0])
        upper = np.array([0.0])  # Invalid: lower > upper

        with pytest.raises(
            ValueError, match="Lower bounds must be less than upper bounds"
        ):
            enforce_bounds(point, lower, upper)


class TestKnownSimplexConfigurations:
    """Tests with specific known simplex configurations."""

    def test_equilateral_triangle_simplex_2d(self):
        """Test operations on a regular equilateral triangle simplex in 2D."""
        # Equilateral triangle with side length 2
        simplex = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3)]])

        # Centroid should be at (1, sqrt(3)/3) when excluding worst
        centroid = compute_centroid(simplex, exclude_worst=True)
        expected_centroid = np.array([1.0, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected_centroid)

        # Test reflection of worst point
        worst = simplex[2]
        reflected = reflect(centroid, worst, delta_r=1.0)
        # Should place point on opposite side
        assert reflected[1] < 0, "Reflected point should be below x-axis"

    def test_unit_simplex_3d(self):
        """Test operations on a unit simplex in 3D."""
        # Standard 3-simplex with vertices at origin and unit vectors
        simplex = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        centroid = compute_centroid(simplex, exclude_worst=True)
        expected = np.array([1 / 3, 1 / 3, 0.0])
        np.testing.assert_array_almost_equal(centroid, expected)

        # Test shrink operation
        best = simplex[0]
        shrunk = shrink(best, simplex, gamma_s=0.5)
        # All points should move halfway toward origin
        expected_shrunk = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
        )
        np.testing.assert_array_almost_equal(shrunk, expected_shrunk)

    def test_right_triangle_simplex(self):
        """Test operations on a right triangle simplex."""
        simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Test all operations produce valid results
        centroid = compute_centroid(simplex, exclude_worst=True)
        worst = simplex[2]
        reflected = reflect(centroid, worst, delta_r=1.0)
        expanded = expand(centroid, reflected, delta_e=2.0)
        contracted_out = contract_outside(centroid, reflected, delta_oc=0.5)
        contracted_in = contract_inside(centroid, worst, delta_ic=-0.5)

        # All operations should produce finite values
        assert np.all(np.isfinite(reflected))
        assert np.all(np.isfinite(expanded))
        assert np.all(np.isfinite(contracted_out))
        assert np.all(np.isfinite(contracted_in))


class TestEdgeCases:
    """Tests for edge cases including degenerate simplices and boundary conditions."""

    def test_degenerate_simplex_collinear_points(self):
        """Test operations on a degenerate simplex with collinear points."""
        # All points on a line
        simplex = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        centroid = compute_centroid(simplex, exclude_worst=True)
        expected = np.array([0.5, 0.5])
        np.testing.assert_array_almost_equal(centroid, expected)

        # Operations should still work even with degenerate geometry
        worst = simplex[2]
        reflected = reflect(centroid, worst, delta_r=1.0)
        assert np.all(np.isfinite(reflected))

    def test_degenerate_simplex_duplicate_points(self):
        """Test operations when simplex has duplicate points."""
        # Two identical points
        simplex = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])

        centroid = compute_centroid(simplex, exclude_worst=True)
        expected = np.array([1.0, 1.0])
        np.testing.assert_array_almost_equal(centroid, expected)

        # Shrink should handle duplicates
        best = simplex[0]
        shrunk = shrink(best, simplex, gamma_s=0.5)
        assert np.all(np.isfinite(shrunk))

    def test_simplex_at_boundary(self):
        """Test operations when simplex is at problem boundary."""
        simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        # Reflection might go outside bounds
        centroid = compute_centroid(simplex, exclude_worst=True)
        worst = simplex[2]
        reflected = reflect(centroid, worst, delta_r=1.0)

        # Enforce bounds should bring it back
        enforced = enforce_bounds(reflected, lower, upper)
        assert np.all(enforced >= lower)
        assert np.all(enforced <= upper)

    def test_very_small_simplex(self):
        """Test operations on a very small simplex."""
        # Simplex with very small edge lengths
        simplex = np.array([[0.0, 0.0], [1e-10, 0.0], [0.0, 1e-10]])

        centroid = compute_centroid(simplex, exclude_worst=True)
        worst = simplex[2]

        # Operations should still produce valid results
        reflected = reflect(centroid, worst, delta_r=1.0)
        assert np.all(np.isfinite(reflected))
        assert not np.any(np.isnan(reflected))

    def test_single_dimension_simplex(self):
        """Test operations on a 1D simplex (2 points on a line)."""
        simplex = np.array([[0.0], [2.0]])

        centroid = compute_centroid(simplex, exclude_worst=True)
        expected = np.array([0.0])
        np.testing.assert_array_almost_equal(centroid, expected)

        worst = simplex[1]
        reflected = reflect(centroid, worst, delta_r=1.0)
        expected_reflected = np.array([-2.0])
        np.testing.assert_array_almost_equal(reflected, expected_reflected)

    def test_zero_coefficient_operations(self):
        """Test operations with zero coefficients."""
        centroid = np.array([1.0, 1.0])
        point = np.array([2.0, 2.0])

        # Zero expansion should return centroid
        expanded = expand(centroid, point, delta_e=0.0)
        np.testing.assert_array_almost_equal(expanded, centroid)

        # Zero contraction should return centroid
        contracted = contract_outside(centroid, point, delta_oc=0.0)
        np.testing.assert_array_almost_equal(contracted, centroid)


class TestNumericalStability:
    """Tests for numerical stability with extreme values."""

    def test_very_large_coordinates(self):
        """Test operations with very large coordinate values."""
        simplex = np.array([[1e8, 1e8], [1e8 + 1, 1e8], [1e8, 1e8 + 1]])

        centroid = compute_centroid(simplex, exclude_worst=True)
        worst = simplex[2]

        # Operations should not overflow
        reflected = reflect(centroid, worst, delta_r=1.0)
        assert np.all(np.isfinite(reflected))
        assert not np.any(np.isnan(reflected))

        expanded = expand(centroid, reflected, delta_e=2.0)
        assert np.all(np.isfinite(expanded))

    def test_very_small_coordinates(self):
        """Test operations with very small coordinate values."""
        simplex = np.array([[1e-8, 1e-8], [1e-8 + 1e-10, 1e-8], [1e-8, 1e-8 + 1e-10]])

        centroid = compute_centroid(simplex, exclude_worst=True)
        worst = simplex[2]

        # Operations should maintain precision
        reflected = reflect(centroid, worst, delta_r=1.0)
        assert np.all(np.isfinite(reflected))
        assert not np.any(np.isnan(reflected))

    def test_mixed_scale_coordinates(self):
        """Test operations with mixed scale coordinates (some large, some small)."""
        simplex = np.array([[1e-8, 1e8], [1e-7, 1e8 + 1], [1e-8, 1e8 + 2]])

        centroid = compute_centroid(simplex, exclude_worst=True)
        worst = simplex[2]

        # Operations should handle mixed scales
        reflected = reflect(centroid, worst, delta_r=1.0)
        assert np.all(np.isfinite(reflected))

        best = simplex[0]
        shrunk = shrink(best, simplex, gamma_s=0.5)
        assert np.all(np.isfinite(shrunk))

    def test_extreme_coefficients(self):
        """Test operations with extreme coefficient values."""
        centroid = np.array([1.0, 1.0])
        point = np.array([2.0, 2.0])

        # Very large coefficient
        reflected = reflect(centroid, point, delta_r=1000.0)
        assert np.all(np.isfinite(reflected))

        # Very small positive coefficient
        contracted = contract_outside(centroid, point, delta_oc=1e-10)
        assert np.all(np.isfinite(contracted))
        # Should be very close to centroid
        assert np.linalg.norm(contracted - centroid) < 1e-8

    def test_negative_coordinates(self):
        """Test operations with negative coordinates."""
        simplex = np.array([[-10.0, -10.0], [-9.0, -10.0], [-10.0, -9.0]])

        centroid = compute_centroid(simplex, exclude_worst=True)
        worst = simplex[2]

        # All operations should work with negative values
        reflected = reflect(centroid, worst, delta_r=1.0)
        expanded = expand(centroid, reflected, delta_e=2.0)
        contracted_out = contract_outside(centroid, reflected, delta_oc=0.5)
        contracted_in = contract_inside(centroid, worst, delta_ic=-0.5)

        assert np.all(np.isfinite(reflected))
        assert np.all(np.isfinite(expanded))
        assert np.all(np.isfinite(contracted_out))
        assert np.all(np.isfinite(contracted_in))

    def test_bounds_with_extreme_values(self):
        """Test bound enforcement with extreme values."""
        point = np.array([1e10, -1e10, 0.0])
        lower = np.array([-1e9, -1e9, -1e9])
        upper = np.array([1e9, 1e9, 1e9])

        enforced = enforce_bounds(point, lower, upper)
        expected = np.array([1e9, -1e9, 0.0])
        np.testing.assert_array_almost_equal(enforced, expected)

        # Verify bounds are respected
        assert np.all(enforced >= lower)
        assert np.all(enforced <= upper)

    def test_high_dimensional_simplex(self):
        """Test operations on high-dimensional simplex."""
        dim = 50
        # Create a simplex in 50D
        simplex = np.random.RandomState(42).randn(dim + 1, dim)

        centroid = compute_centroid(simplex, exclude_worst=True)
        assert centroid.shape == (dim,)
        assert np.all(np.isfinite(centroid))

        worst = simplex[-1]
        reflected = reflect(centroid, worst, delta_r=1.0)
        assert reflected.shape == (dim,)
        assert np.all(np.isfinite(reflected))

        best = simplex[0]
        shrunk = shrink(best, simplex, gamma_s=0.5)
        assert shrunk.shape == (dim + 1, dim)
        assert np.all(np.isfinite(shrunk))
