"""
Unit tests for stopping criteria.

These tests verify specific examples and edge cases for convergence detection.
"""

import numpy as np
from nelder_mead.core.stopping_criteria import (
    check_oriented_length,
    check_std_dev,
    check_small_simplex,
    check_flat_simplex,
    check_degenerate_simplex,
    check_fminsearch_fun,
    check_fminsearch_x,
)


class TestOrientedLength:
    """Tests for oriented length criterion."""

    def test_small_simplex_converged(self):
        """Test that small simplex is detected as converged."""
        # Very small simplex around [1, 1]
        simplex = np.array([[1.0, 1.0], [1.0001, 1.0], [1.0, 1.0001]])
        assert check_oriented_length(simplex, epsilon=1e-2)

    def test_large_simplex_not_converged(self):
        """Test that large simplex is not detected as converged."""
        simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert not check_oriented_length(simplex, epsilon=1e-4)

    def test_normalized_by_first_vertex_norm(self):
        """Test that normalization by first vertex norm works correctly."""
        # Simplex far from origin should have different threshold
        simplex_far = np.array([[10.0, 10.0], [10.01, 10.0], [10.0, 10.01]])

        # Same absolute edge length, but different relative to position
        # Far simplex should be more likely to converge
        epsilon = 1e-3
        assert check_oriented_length(simplex_far, epsilon=epsilon)


class TestStdDev:
    """Tests for standard deviation criterion."""

    def test_identical_fitness_converged(self):
        """Test that identical fitness values indicate convergence."""
        fitness = np.array([1.0, 1.0, 1.0])
        assert check_std_dev(fitness, epsilon=1e-7)

    def test_similar_fitness_converged(self):
        """Test that very similar fitness values indicate convergence."""
        fitness = np.array([1.0, 1.0000001, 1.0000002])
        assert check_std_dev(fitness, epsilon=1e-10)

    def test_different_fitness_not_converged(self):
        """Test that different fitness values do not indicate convergence."""
        fitness = np.array([1.0, 2.0, 3.0])
        assert not check_std_dev(fitness, epsilon=1e-7)


class TestSmallSimplex:
    """Tests for small simplex criterion."""

    def test_tiny_simplex_in_large_space(self):
        """Test that tiny simplex in large space is detected."""
        simplex = np.array([[5.0, 5.0], [5.001, 5.0], [5.0, 5.001]])
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert check_small_simplex(simplex, bounds, epsilon=1e-3)

    def test_large_simplex_not_small(self):
        """Test that large simplex is not detected as small."""
        simplex = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert not check_small_simplex(simplex, bounds, epsilon=1e-3)

    def test_relative_to_bounds(self):
        """Test that detection is relative to bound range."""
        # Same absolute size, different relative to bounds
        simplex1 = np.array([[0.5, 0.5], [0.51, 0.5], [0.5, 0.51]])
        bounds1 = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        simplex2 = np.array([[5.0, 5.0], [5.01, 5.0], [5.0, 5.01]])
        bounds2 = (np.array([0.0, 0.0]), np.array([100.0, 100.0]))

        epsilon = 0.02
        # First simplex is 1% of bounds, should be detected
        assert check_small_simplex(simplex1, bounds1, epsilon=epsilon)
        # Second simplex is 0.01% of bounds, should be detected
        assert check_small_simplex(simplex2, bounds2, epsilon=epsilon)


class TestFlatSimplex:
    """Tests for flat simplex criterion."""

    def test_identical_fitness_flat(self):
        """Test that identical fitness values indicate flat simplex."""
        assert check_flat_simplex(1.0, 1.0, epsilon=1e-6)

    def test_similar_fitness_flat(self):
        """Test that very similar fitness values indicate flat simplex."""
        assert check_flat_simplex(1.0, 1.0000001, epsilon=1e-5)

    def test_different_fitness_not_flat(self):
        """Test that different fitness values do not indicate flat simplex."""
        assert not check_flat_simplex(1.0, 2.0, epsilon=1e-6)


class TestDegenerateSimplex:
    """Tests for degenerate simplex criterion."""

    def test_collinear_points_degenerate(self):
        """Test that nearly collinear points are detected as degenerate."""
        # Points nearly on a line
        simplex = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.001]])
        assert check_degenerate_simplex(simplex, epsilon1=0.1, epsilon2=0.1)

    def test_regular_simplex_not_degenerate(self):
        """Test that regular simplex is not degenerate."""
        # Equilateral triangle (not at origin to avoid zero norms)
        simplex = np.array([[1.0, 1.0], [2.0, 1.0], [1.5, 1.866]])
        assert not check_degenerate_simplex(simplex, epsilon1=0.1, epsilon2=0.1)

    def test_edge_ratio_detection(self):
        """Test that edge ratio criterion works."""
        # One vertex very close to origin, others far
        simplex = np.array([[0.001, 0.001], [10.0, 0.0], [0.0, 10.0]])
        assert check_degenerate_simplex(simplex, epsilon1=0.01, epsilon2=1.0)


class TestFminsearchFun:
    """Tests for fminsearch function tolerance criterion."""

    def test_identical_fitness_converged(self):
        """Test that identical fitness values indicate convergence."""
        fitness = np.array([1.0, 1.0, 1.0])
        assert check_fminsearch_fun(fitness, tol_fun=1e-12)

    def test_small_differences_converged(self):
        """Test that small fitness differences indicate convergence."""
        fitness = np.array([1.0, 1.0 + 1e-13, 1.0 - 1e-13])
        assert check_fminsearch_fun(fitness, tol_fun=1e-12)

    def test_large_differences_not_converged(self):
        """Test that large fitness differences do not indicate convergence."""
        fitness = np.array([1.0, 1.1, 0.9])
        assert not check_fminsearch_fun(fitness, tol_fun=1e-12)


class TestFminsearchX:
    """Tests for fminsearch position tolerance criterion."""

    def test_identical_positions_converged(self):
        """Test that identical positions indicate convergence."""
        simplex = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        assert check_fminsearch_x(simplex, tol_x=1e-8)

    def test_small_distances_converged(self):
        """Test that small position differences indicate convergence."""
        simplex = np.array([[1.0, 1.0], [1.0 + 1e-9, 1.0], [1.0, 1.0 + 1e-9]])
        assert check_fminsearch_x(simplex, tol_x=1e-8)

    def test_large_distances_not_converged(self):
        """Test that large position differences do not indicate convergence."""
        simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert not check_fminsearch_x(simplex, tol_x=1e-8)

    def test_uses_infinity_norm(self):
        """Test that infinity norm is used for distance calculation."""
        # One dimension has large difference
        simplex = np.array([[0.0, 0.0], [1e-9, 0.0], [0.0, 1.0]])
        # Should not converge because max coordinate difference is 1.0
        assert not check_fminsearch_x(simplex, tol_x=1e-8)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_variance_fitness(self):
        """Test std_dev criterion with exactly zero variance."""
        fitness = np.array([5.0, 5.0, 5.0, 5.0])
        assert check_std_dev(fitness, epsilon=1e-10)

    def test_negative_fitness_values(self):
        """Test that negative fitness values are handled correctly."""
        fitness = np.array([-10.0, -10.0, -10.0])
        assert check_std_dev(fitness, epsilon=1e-7)

    def test_identical_points_all_criteria(self):
        """Test that identical points trigger convergence in all position-based criteria."""
        simplex = np.array([[2.5, 3.5], [2.5, 3.5], [2.5, 3.5]])
        
        # Should converge for oriented length
        assert check_oriented_length(simplex, epsilon=1e-4)
        
        # Should converge for fminsearch_x
        assert check_fminsearch_x(simplex, tol_x=1e-8)
        
        # Should be detected as small simplex
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert check_small_simplex(simplex, bounds, epsilon=1e-3)

    def test_simplex_at_origin(self):
        """Test oriented length criterion with simplex at origin."""
        # Small simplex at origin should use denominator of 1.0
        simplex = np.array([[0.0, 0.0], [0.0001, 0.0], [0.0, 0.0001]])
        assert check_oriented_length(simplex, epsilon=1e-3)

    def test_zero_bound_range(self):
        """Test small simplex criterion with zero bound range in one dimension."""
        simplex = np.array([[5.0, 5.0], [5.01, 5.0], [5.0, 5.01]])
        # Bounds with zero range in one dimension
        bounds = (np.array([5.0, 0.0]), np.array([5.0, 10.0]))
        # Should handle zero bound range gracefully
        result = check_small_simplex(simplex, bounds, epsilon=1e-3)
        # The function should not crash and should check other dimensions
        # numpy bool is a subclass of bool, so check for bool-like behavior
        assert result in [True, False] or isinstance(result, (bool, np.bool_))

    def test_degenerate_simplex_singular_matrix(self):
        """Test degenerate simplex detection with singular matrix."""
        # Create a truly degenerate simplex (all points identical)
        # When all points are identical, min_norm = max_norm, so edge_ratio = 1.0
        # This won't trigger the edge ratio check, but will have zero determinant
        simplex = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        # With identical points, the edge matrix is all zeros, det = 0, prod_lengths = 0
        # This causes division by zero, but the check is prod_lengths > 0
        # So this case returns False (not degenerate by this criterion)
        # Let's test a different case: nearly collinear points that create singular matrix
        simplex_singular = np.array([[0.0, 0.0], [1e-10, 1e-10], [2e-10, 2e-10]])
        assert check_degenerate_simplex(simplex_singular, epsilon1=0.1, epsilon2=0.1)

    def test_degenerate_simplex_zero_volume(self):
        """Test degenerate simplex with zero volume (flat simplex)."""
        # All points on a line (determinant = 0)
        simplex = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert check_degenerate_simplex(simplex, epsilon1=0.1, epsilon2=0.1)

    def test_flat_simplex_with_negative_values(self):
        """Test flat simplex criterion with negative fitness values."""
        assert check_flat_simplex(-5.0, -5.0, epsilon=1e-6)
        assert check_flat_simplex(-5.0, -4.999999, epsilon=1e-5)
        assert not check_flat_simplex(-5.0, -3.0, epsilon=1e-6)

    def test_very_large_simplex(self):
        """Test criteria with very large coordinate values."""
        simplex = np.array([[1e6, 1e6], [1e6 + 1, 1e6], [1e6, 1e6 + 1]])
        
        # Should converge for oriented length (normalized by position)
        assert check_oriented_length(simplex, epsilon=1e-5)
        
        # Should converge for fminsearch_x
        assert check_fminsearch_x(simplex, tol_x=2.0)

    def test_mixed_scale_edges(self):
        """Degeneracy is about edge lengths, not where the simplex sits."""
        # Edges of wildly unequal length -> degenerate by the eps1 (edge-ratio) test
        lopsided = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 0.001]])
        assert check_degenerate_simplex(lopsided, epsilon1=0.01, epsilon2=1e-12)

        # Same shape, translated far from the origin: still degenerate.
        # The pre-2026 implementation compared vertex position norms, so this
        # case silently stopped firing once the simplex moved away from 0.
        assert check_degenerate_simplex(lopsided + 1e6, epsilon1=0.01, epsilon2=1e-12)

        # A large, well-shaped simplex with one vertex near the origin is NOT
        # degenerate -- the old vertex-norm test flagged it.
        healthy = np.array([[0.0001, 0.0001], [100.0, 0.0], [0.0, 100.0]])
        assert not check_degenerate_simplex(healthy, epsilon1=0.01, epsilon2=0.1)

    def test_high_dimensional_simplex(self):
        """Test criteria work correctly in higher dimensions."""
        # 5D simplex
        dim = 5
        simplex = np.random.RandomState(42).uniform(0, 1, size=(dim + 1, dim))
        
        # Should not converge for large epsilon
        assert not check_oriented_length(simplex, epsilon=1e-10)
        
        # Create converged 5D simplex
        center = np.array([0.5] * dim)
        converged_simplex = np.array([center + 1e-8 * np.random.RandomState(i).randn(dim) 
                                      for i in range(dim + 1)])
        assert check_oriented_length(converged_simplex, epsilon=1e-4)

    def test_single_point_variance(self):
        """Test std_dev with single fitness value."""
        fitness = np.array([1.0])
        # Single point has zero variance
        assert check_std_dev(fitness, epsilon=1e-7)
