"""
Unit tests for restart strategies.

These tests verify specific examples and edge cases for restart point generation.
"""

import numpy as np
import pytest
from nelder_mead.core.restart_strategies import (
    UniformRestart,
    GaussianRestart,
    GaussianBestRestart,
    create_restart_strategy,
)


class TestUniformRestart:
    """Tests for uniform restart strategy."""

    def test_generates_points_within_bounds(self):
        """Test that generated points are within bounds."""
        strategy = UniformRestart(seed=42)
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        for _ in range(10):
            point = strategy.generate_point(2, bounds)
            assert np.all(point >= 0.0) and np.all(point <= 1.0)

    def test_generates_different_points(self):
        """Test that multiple calls generate different points."""
        strategy = UniformRestart(seed=42)
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        point1 = strategy.generate_point(2, bounds)
        point2 = strategy.generate_point(2, bounds)

        assert not np.allclose(point1, point2)

    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequence."""
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        strategy1 = UniformRestart(seed=42)
        point1 = strategy1.generate_point(2, bounds)

        strategy2 = UniformRestart(seed=42)
        point2 = strategy2.generate_point(2, bounds)

        np.testing.assert_array_almost_equal(point1, point2)

    def test_handles_asymmetric_bounds(self):
        """Test with asymmetric bounds."""
        strategy = UniformRestart(seed=42)
        bounds = (np.array([-10.0, -5.0]), np.array([10.0, 5.0]))

        for _ in range(10):
            point = strategy.generate_point(2, bounds)
            assert np.all(point >= np.array([-10.0, -5.0]))
            assert np.all(point <= np.array([10.0, 5.0]))

    def test_ignores_best_solution(self):
        """Test that best_solution parameter is ignored."""
        strategy = UniformRestart(seed=42)
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        best = np.array([0.5, 0.5])

        # Should work fine even with best_solution provided
        point = strategy.generate_point(2, bounds, best_solution=best)
        assert np.all(point >= 0.0) and np.all(point <= 1.0)


class TestGaussianRestart:
    """Tests for Gaussian restart strategy."""

    def test_generates_points_clipped_to_bounds(self):
        """Test that generated points are clipped to bounds."""
        strategy = GaussianRestart(seed=42)
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        for _ in range(10):
            point = strategy.generate_point(2, bounds)
            assert np.all(point >= 0.0) and np.all(point <= 1.0)

    def test_centered_at_origin(self):
        """Test that distribution is centered at origin."""
        strategy = GaussianRestart(seed=42)
        bounds = (np.array([-100.0, -100.0]), np.array([100.0, 100.0]))

        # Generate many points and check mean is near origin
        points = np.array([strategy.generate_point(2, bounds) for _ in range(1000)])
        mean = np.mean(points, axis=0)

        # Mean should be close to origin (within 0.1 for 1000 samples)
        assert np.allclose(mean, np.array([0.0, 0.0]), atol=0.1)

    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequence."""
        bounds = (np.array([-5.0, -5.0]), np.array([5.0, 5.0]))

        strategy1 = GaussianRestart(seed=42)
        point1 = strategy1.generate_point(2, bounds)

        strategy2 = GaussianRestart(seed=42)
        point2 = strategy2.generate_point(2, bounds)

        np.testing.assert_array_almost_equal(point1, point2)

    def test_ignores_best_solution(self):
        """Test that best_solution parameter is ignored."""
        strategy = GaussianRestart(seed=42)
        bounds = (np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
        best = np.array([2.0, 2.0])

        # Should work fine even with best_solution provided
        point = strategy.generate_point(2, bounds, best_solution=best)
        assert np.all(point >= -5.0) and np.all(point <= 5.0)


class TestGaussianBestRestart:
    """Tests for Gaussian-best restart strategy."""

    def test_generates_points_clipped_to_bounds(self):
        """Test that generated points are clipped to bounds."""
        strategy = GaussianBestRestart(seed=42)
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        best = np.array([5.0, 5.0])

        for _ in range(10):
            point = strategy.generate_point(2, bounds, best_solution=best)
            assert np.all(point >= 0.0) and np.all(point <= 10.0)

    def test_centered_at_best_solution(self):
        """Test that distribution is centered at best solution."""
        strategy = GaussianBestRestart(seed=42)
        bounds = (np.array([-100.0, -100.0]), np.array([100.0, 100.0]))
        best = np.array([10.0, -5.0])

        # Generate many points and check mean is near best
        points = np.array(
            [
                strategy.generate_point(2, bounds, best_solution=best)
                for _ in range(1000)
            ]
        )
        mean = np.mean(points, axis=0)

        # Mean should be close to best (within 0.2 for 1000 samples)
        assert np.allclose(mean, best, atol=0.2)

    def test_reproducible_with_seed(self):
        """Test that same seed produces same sequence."""
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        best = np.array([5.0, 5.0])

        strategy1 = GaussianBestRestart(seed=42)
        point1 = strategy1.generate_point(2, bounds, best_solution=best)

        strategy2 = GaussianBestRestart(seed=42)
        point2 = strategy2.generate_point(2, bounds, best_solution=best)

        np.testing.assert_array_almost_equal(point1, point2)

    def test_requires_best_solution(self):
        """Test that ValueError is raised if best_solution is None."""
        strategy = GaussianBestRestart(seed=42)
        bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))

        with pytest.raises(ValueError, match="requires best_solution"):
            strategy.generate_point(2, bounds, best_solution=None)

    def test_explores_around_best(self):
        """Test that points are generated around best solution."""
        strategy = GaussianBestRestart(seed=42)
        bounds = (np.array([0.0, 0.0]), np.array([100.0, 100.0]))
        best = np.array([50.0, 50.0])

        # Generate points and check they're distributed around best
        points = np.array(
            [strategy.generate_point(2, bounds, best_solution=best) for _ in range(100)]
        )

        # Most points should be within a few standard deviations of best
        distances = np.linalg.norm(points - best, axis=1)
        # With standard normal, ~95% should be within 2 std devs (distance ~2.8 in 2D)
        assert np.percentile(distances, 95) < 5.0


class TestCreateRestartStrategy:
    """Tests for restart strategy factory function."""

    def test_creates_uniform_strategy(self):
        """Test that 'uniform' creates UniformRestart."""
        strategy = create_restart_strategy("uniform", seed=42)
        assert isinstance(strategy, UniformRestart)

    def test_creates_gaussian_strategy(self):
        """Test that 'gaussian' creates GaussianRestart."""
        strategy = create_restart_strategy("gaussian", seed=42)
        assert isinstance(strategy, GaussianRestart)

    def test_creates_gaussian_best_strategy(self):
        """Test that 'gaussian_best' creates GaussianBestRestart."""
        strategy = create_restart_strategy("gaussian_best", seed=42)
        assert isinstance(strategy, GaussianBestRestart)

    def test_case_insensitive(self):
        """Test that strategy names are case-insensitive."""
        strategy1 = create_restart_strategy("UNIFORM", seed=42)
        strategy2 = create_restart_strategy("Uniform", seed=42)
        strategy3 = create_restart_strategy("uniform", seed=42)

        assert isinstance(strategy1, UniformRestart)
        assert isinstance(strategy2, UniformRestart)
        assert isinstance(strategy3, UniformRestart)

    def test_raises_on_invalid_strategy(self):
        """Test that ValueError is raised for invalid strategy name."""
        with pytest.raises(ValueError, match="not a valid restart strategy"):
            create_restart_strategy("invalid_strategy", seed=42)

    def test_error_message_lists_available_strategies(self):
        """Test that error message lists available strategies."""
        try:
            create_restart_strategy("invalid", seed=42)
        except ValueError as e:
            error_msg = str(e)
            assert "uniform" in error_msg
            assert "gaussian" in error_msg
            assert "gaussian_best" in error_msg


class TestRestartStrategyBase:
    """Tests for RestartStrategy base class."""

    def test_set_seed_updates_random_state(self):
        """Test that set_rng updates the random state."""
        strategy = UniformRestart(seed=42)
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        point1 = strategy.generate_point(2, bounds)

        # Reset RNG with same seed and generate again
        strategy.set_rng(np.random.default_rng(42))
        point2 = strategy.generate_point(2, bounds)

        # Should get the same point
        np.testing.assert_array_almost_equal(point1, point2)
