"""
Tests for Kelley's oriented-restart Nelder-Mead.

The reference is Kelley, *Iterative Methods for Optimization*, SIAM 1999,
section 8.1.4, and the companion paper SIAM J. Optim. 10(1):43-55. The
conditions these tests pin down are quoted where they matter, because the
first implementation of this class diverged from all three and nothing caught
it.
"""

import numpy as np
import pytest

from nelder_mead.algorithms.kelley_nm import KelleyNelderMead
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.benchmarks.unconstrained import Rosenbrock, Sphere


def _kw(problem, **over):
    kw = dict(
        objective_fn=problem.objective,
        lower_bounds=problem.bounds[0],
        upper_bounds=problem.bounds[1],
        max_fes=10000,
        seed=101,
    )
    kw.update(over)
    return kw


class TestSufficientDecreaseGuard:
    """
    Kelley p.139: "We propose performing an oriented restart when (8.4) fails
    *but* f^{k+1} - f^k < 0."

    Both halves are required. Nelder-Mead leaves the best vertex unchanged on a
    large fraction of iterations -- a reflection that beats the second-worst but
    not the best replaces only the worst vertex -- and on those iterations
    f^{k+1} - f^k is exactly 0, which satisfies (8.4)'s failure trivially. Without
    the strict-decrease half the test flags ordinary Nelder-Mead progress as
    stagnation.
    """

    def test_no_restart_when_best_vertex_unchanged(self):
        p = Sphere(dim=5)
        alg = KelleyNelderMead(**_kw(p))
        alg.run()
        # Sphere is the easy case: Kelley reports single-digit restart counts even
        # on McKinnon's pathological examples, so hundreds here means the detector
        # is firing on healthy iterations.
        assert alg.num_oriented_restarts <= 5, (
            f"{alg.num_oriented_restarts} oriented restarts on a 5-D sphere; "
            "the strict-decrease guard is missing"
        )

    def test_detector_uses_the_mean_not_the_best_vertex(self):
        """
        Kelley p.136: "while a Nelder-Mead iterate may not result in a reduction
        in the best function value, the average value ... will be reduced", and
        Assumption 8.1.1 then requires f-bar^{k+1} < f-bar^k. So (8.4) is stated
        on the mean.
        """
        p = Rosenbrock(dim=2)
        alg = KelleyNelderMead(**_kw(p, max_fes=200))
        alg.run()
        assert hasattr(alg, "_prev_mean")
        assert alg._prev_mean == pytest.approx(float(np.mean(alg.fitness_values)))

    def test_no_restart_when_mean_does_not_decrease(self):
        """A non-decreasing mean is not stagnation under Kelley's rule."""
        p = Rosenbrock(dim=2)
        alg = KelleyNelderMead(**_kw(p, max_fes=200))
        alg.run()
        alg._prev_mean = -np.inf          # improvement > 0, so the guard must hold
        before = alg.num_oriented_restarts
        alg._sort_simplex()
        assert alg.num_oriented_restarts == before

    def test_does_not_destroy_plain_nelder_mead_performance(self):
        """Kelley's method is a safeguard; it must not be far worse than plain NM."""
        for cls, dim in ((Sphere, 5), (Rosenbrock, 2), (Rosenbrock, 5)):
            p = cls(dim=dim)
            plain = NelderMead(**_kw(p))
            plain.run()
            kelley = KelleyNelderMead(**_kw(p))
            kelley.run()
            f_plain = float(np.min(plain.fitness_values))
            f_kelley = float(np.min(kelley.fitness_values))
            assert f_kelley <= max(1e-8, 1e4 * f_plain), (
                f"{p.name} d={dim}: Kelley {f_kelley:.3e} vs plain {f_plain:.3e}"
            )


class TestOrientedRestartGeometry:
    """
    Kelley eq. (8.7): y_j = y_1 - beta_{j-1} e_{j-1}, with
    beta_l = 0.5 * sigma_minus(S) * sign((Df)_l) when (Df)_l != 0, and
    beta_l = 0.5 * sigma_minus(S)               when (Df)_l  = 0,
    where sigma_minus is the *smallest* edge length from the best vertex.
    """

    def test_restart_uses_sigma_minus_and_shrinks_the_simplex(self):
        p = Rosenbrock(dim=3)
        alg = KelleyNelderMead(**_kw(p, max_fes=400))
        alg.run()

        from nelder_mead.core import simplex_operations as ops

        edges = np.linalg.norm(alg.simplex[1:] - alg.simplex[0], axis=1)
        sigma_minus = float(edges.min())
        alg._oriented_restart()
        new_edges = np.linalg.norm(alg.simplex[1:] - alg.simplex[0], axis=1)

        # "The diameter of the new simplex has not been increased ... Moreover all
        # edge lengths have been reduced. So after reordering sigma_+(S^{k+1}) <=
        # sigma_-(S^k)."
        assert new_edges.max() <= sigma_minus + 1e-9, (
            f"oriented restart grew the simplex: max edge {new_edges.max():.3e} "
            f"> sigma_minus {sigma_minus:.3e}"
        )

    def test_restart_keeps_the_best_vertex(self):
        p = Rosenbrock(dim=3)
        alg = KelleyNelderMead(**_kw(p, max_fes=400))
        alg.run()
        best_f = float(alg.fitness_values[0])
        alg._oriented_restart()
        assert float(alg.fitness_values[0]) <= best_f + 1e-12


class TestAlphaScaling:
    """
    Kelley p.138: "if the simplex diameter is much smaller than ||D_k f||, (8.4)
    could fail on the first iterate. We address this problem with the scaling
    alpha = alpha_0 * sigma_+(S^0) / ||D^0 f||."

    So the constructor argument is alpha_0, and the alpha actually used in the
    test is rescaled once from the initial simplex.
    """

    def test_alpha_is_rescaled_from_the_initial_simplex(self):
        p = Rosenbrock(dim=5)
        alg = KelleyNelderMead(**_kw(p, max_fes=300, kelley_alpha=1e-4))
        alg.run()
        assert alg.kelley_alpha_0 == pytest.approx(1e-4)
        assert alg.kelley_alpha != pytest.approx(1e-4), (
            "alpha was never rescaled by sigma_+(S^0)/||D^0 f||"
        )
        assert alg.kelley_alpha > 0.0

    def test_rejects_nonpositive_alpha(self):
        p = Sphere(dim=2)
        with pytest.raises(ValueError):
            KelleyNelderMead(**_kw(p, kelley_alpha=0.0))


def _mckinnon(tau, theta, phi):
    """McKinnon (1998) family; Nelder-Mead provably stalls at the origin, f = 0."""

    def f(v):
        x, y = float(v[0]), float(v[1])
        if x <= 0:
            return theta * phi * abs(x) ** tau + y + y * y
        return theta * x ** tau + y + y * y

    return f


_L1, _L2 = (1 + np.sqrt(33)) / 8, (1 - np.sqrt(33)) / 8
_MCKINNON_SIMPLEX = np.array([[0.0, 0.0], [1.0, 1.0], [_L1, _L2]])


def _pure_descent(cls, fn, max_fes=600):
    """The algorithm with the base random restart disabled, so only Kelley acts."""

    class _M(cls):
        def initialize_population(self, x0=None):
            return _MCKINNON_SIMPLEX.copy()

        def _restart_simplex(self):
            self._stop = True

        def should_terminate(self):
            return getattr(self, "_stop", False) or super().should_terminate()

    return _M(
        objective_fn=fn,
        lower_bounds=np.array([-10.0, -10.0]),
        upper_bounds=np.array([10.0, 10.0]),
        max_fes=max_fes,
        seed=101,
    )


class TestMcKinnonStagnation:
    """
    The reason Kelley's restart exists. McKinnon (1998), SIAM J. Optim. 9(1),
    builds a family on which Nelder-Mead converges to a non-stationary point by
    repeated inside contraction with the best vertex held fixed. Kelley reports
    that an oriented restart escapes it, taking a single restart on the smoothest
    case.

    This is the spec test: any implementation that cannot escape here is not
    Kelley's method, whatever its restart count elsewhere.
    """

    @pytest.mark.parametrize("tau,theta,phi", [(3.0, 6.0, 400.0), (2.0, 6.0, 60.0)])
    def test_plain_nelder_mead_stagnates(self, tau, theta, phi):
        alg = _pure_descent(NelderMead, _mckinnon(tau, theta, phi))
        alg.run()
        assert float(np.min(alg.fitness_values)) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("tau,theta,phi", [(3.0, 6.0, 400.0), (2.0, 6.0, 60.0)])
    def test_kelley_escapes(self, tau, theta, phi):
        alg = _pure_descent(KelleyNelderMead, _mckinnon(tau, theta, phi))
        alg.run()
        assert float(np.min(alg.fitness_values)) < -1e-3, "did not escape the origin"
        assert alg.num_oriented_restarts >= 1

    def test_smoothest_case_needs_a_single_restart(self):
        """Kelley p.140 reports exactly one oriented restart for (3, 6, 400)."""
        alg = _pure_descent(KelleyNelderMead, _mckinnon(3.0, 6.0, 400.0))
        alg.run()
        assert alg.num_oriented_restarts == 1
