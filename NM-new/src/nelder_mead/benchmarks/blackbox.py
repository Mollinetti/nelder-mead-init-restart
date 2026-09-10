#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation-based black-box design problems.

Every other benchmark in this package is a closed-form algebraic expression. That
is a real gap in the evidence for a derivative-free method: the introduction of the
paper motivates c-NM by pointing at problems "where the objective function was
nondifferentiable and/or noncontinuous", yet an analytic surrogate of an
engineering problem is smooth almost everywhere and cheap to evaluate, so it never
exercises that motivation.

:class:`CantileverBracketFEA` closes the gap. Each evaluation assembles and solves
a plane-stress finite element system, so:

- There is no closed form. The objective and the constraints are outputs of a
  linear solve on a mesh, exactly as an industrial sizing problem would be.
- The stress constraint is genuinely non-differentiable. It is the maximum von
  Mises stress over all elements, and the identity of the critical element changes
  as the design changes; at every such switch the constraint has a kink. This is
  the C^0 structure the paper's analysis assumes, not a smooth stand-in for it.
- Evaluations are expensive relative to the algebraic problems, on the order of
  milliseconds rather than microseconds, so a function-evaluation budget is a
  meaningful resource rather than a formality.

MOPTA08 (Jones, 2008), the standard expensive constrained benchmark, is also
available and is implemented in :mod:`nelder_mead.benchmarks.mopta08`. The two are
complementary rather than alternatives. MOPTA08 is larger (124 variables, 68
constraints) and is the recognized reference point, but it is an opaque
third-party executable: it must be downloaded per platform, it costs roughly 0.3 s
per evaluation, and its internals cannot be inspected. This FE problem is smaller
but fully open — the mesh, the physics and the constraints are all visible and
adjustable — which makes it the one to reach for when the question is *why* a
method behaves as it does rather than *how it ranks*. It also needs no external
binary, so it runs anywhere the package does.

References:
    Jones, D. R. (2008). Large-scale multi-disciplinary mass optimization in the
    auto industry. MOPTA 2008 Conference, Ontario, Canada.

    Gustafsson, T., & McBain, G. D. (2020). scikit-fem: A Python package for
    finite element assembly. Journal of Open Source Software, 5(52), 2369.
"""

import time
from typing import Optional, Tuple

import numpy as np

from .problem_suite import OptimizationProblem


class CantileverBracketFEA(OptimizationProblem):
    """
    Thickness sizing of a cantilever bracket, evaluated by finite element analysis.

    A rectangular plane-stress bracket is clamped along its left edge and loaded
    downward at its right edge. The plate is divided into contiguous vertical
    strips, and the design variable for each strip is its thickness. The goal is to
    minimize mass while keeping the peak von Mises stress and the tip deflection
    within their allowables, and keeping the thickness change between adjacent
    strips manufacturable.

    Mathematical formulation:
        Minimize: f(t) = sum_e area_e * t(e)                        [mass, m^3]

        Subject to:
            g1(t) = max_e vonMises_e(t) / sigma_allow - 1 <= 0
            g2(t) = |tip deflection(t)| / delta_max - 1 <= 0
            g_{2+i}(t) = |t_i - t_{i+1}| / max_taper - 1 <= 0,  i = 1..n-1

        where vonMises_e and the tip deflection are obtained by solving
        K(t) u = f on the mesh.

        Bounds: min_thickness <= t_i <= max_thickness

    Both g1 and the taper constraints are non-differentiable: the first is a
    maximum over elements, the rest are absolute values.

    Args:
        num_groups: Number of thickness strips, i.e. the problem dimension
        nx: Mesh divisions along the length
        ny: Mesh divisions through the depth
        length: Bracket length in metres
        height: Bracket height in metres
        youngs_modulus: Young's modulus in Pa
        poisson_ratio: Poisson's ratio
        tip_load: Downward tip load in N
        sigma_allow: Allowable von Mises stress in Pa
        delta_max: Allowable tip deflection in m
        max_taper: Allowable thickness change between adjacent strips in m
        min_thickness: Lower bound on strip thickness in m
        max_thickness: Upper bound on strip thickness in m

    Properties:
        - Dimension: num_groups (default 8)
        - Number of inequality constraints: num_groups + 1
        - Number of equality constraints: 0
        - No known optimum: the solution is not available in closed form, so
          `optimum_value` is deliberately left unset. For orientation only, SLSQP
          with finite-difference gradients from 15 random starts reaches a mass of
          0.026884 on the default configuration, at a tapered design running from
          0.0092 m at the root to the 0.005 m lower bound at the tip, with the
          stress constraint exactly active. That is a reference point, not a
          certified global optimum.
        - Non-trivially constrained: the minimum-mass design (every strip at its
          lower bound) is infeasible, exceeding the stress allowable by a factor of
          3.4 and the deflection allowable by 1.5, so the constraints genuinely
          shape the solution rather than being slack decoration.
        - Expensive: each evaluation is a sparse linear solve

    Attributes:
        solve_count (int): Number of finite element solves performed
        solve_time (float): Cumulative wall-clock seconds spent in the solver
        critical_element (int): Index of the element carrying the peak von Mises
            stress in the most recent solve. The stress constraint is a maximum
            over elements, so it is non-differentiable wherever this index changes.

    Example:
        >>> problem = CantileverBracketFEA(num_groups=4, nx=12, ny=4)
        >>> problem.dim
        4
        >>> problem.num_ineq_constraints
        5
        >>> mass = problem.objective(np.full(4, 0.02))
        >>> mass > 0
        True
    """

    def __init__(
        self,
        num_groups: int = 8,
        nx: int = 40,
        ny: int = 10,
        length: float = 4.0,
        height: float = 1.0,
        youngs_modulus: float = 210e9,
        poisson_ratio: float = 0.3,
        tip_load: float = 1e4,
        sigma_allow: float = 2.5e9,
        delta_max: float = 1.5e-3,
        max_taper: float = 0.01,
        min_thickness: float = 0.005,
        max_thickness: float = 0.05,
    ):
        if num_groups < 2:
            raise ValueError(f"num_groups must be at least 2, got {num_groups}")

        if nx < num_groups:
            raise ValueError(
                f"nx ({nx}) must be at least num_groups ({num_groups}) so that "
                "every thickness strip contains at least one element column"
            )

        if min_thickness >= max_thickness:
            raise ValueError("min_thickness must be less than max_thickness")

        self.num_groups = num_groups
        self.length = length
        self.height = height
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.tip_load = tip_load
        self.sigma_allow = sigma_allow
        self.delta_max = delta_max
        self.max_taper = max_taper

        # Instrumentation: the point of this problem is that evaluations are costly
        self.solve_count = 0
        self.solve_time = 0.0
        self.critical_element = -1

        # Building the mesh and the bases is the expensive part of setup and does
        # not depend on the design, so it happens once here rather than per solve.
        self._build_model(nx, ny)

        # Single-entry memo. BaseAlgorithm calls objective() and constraint_ineq()
        # separately for the same point, which would otherwise double the number of
        # finite element solves and make the reported cost per evaluation wrong.
        self._cache_key: Optional[bytes] = None
        self._cache_value: Optional[Tuple[float, float, float]] = None

        super().__init__(
            name=f"CantileverBracketFEA-{num_groups}D",
            dim=num_groups,
            bounds=(
                np.full(num_groups, min_thickness),
                np.full(num_groups, max_thickness),
            ),
            num_eq_constraints=0,
            num_ineq_constraints=num_groups + 1,
        )

    def _build_model(self, nx: int, ny: int) -> None:
        """Construct the mesh, bases, element areas and strip assignment."""
        try:
            from skfem import (
                Basis, ElementTriP0, ElementTriP1, ElementVector, MeshTri,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CantileverBracketFEA needs scikit-fem, which is not installed. "
                "Install it with 'pip install scikit-fem', or "
                "'pip install -e .[benchmarks]' for every benchmark dependency."
            ) from exc

        self._mesh = MeshTri().init_tensor(
            np.linspace(0.0, self.length, nx + 1),
            np.linspace(0.0, self.height, ny + 1),
        )
        self._basis = Basis(self._mesh, ElementVector(ElementTriP1()))
        self._p0_basis = Basis(self._mesh, ElementTriP0())

        # Assign each element to a thickness strip by its centroid's x coordinate
        centroid_x = self._mesh.p[0, self._mesh.t].mean(axis=0)
        self._group = np.clip(
            (centroid_x / self.length * self.num_groups).astype(int),
            0,
            self.num_groups - 1,
        )

        # Element areas, for the mass integral
        edge_1 = self._mesh.p[:, self._mesh.t[1]] - self._mesh.p[:, self._mesh.t[0]]
        edge_2 = self._mesh.p[:, self._mesh.t[2]] - self._mesh.p[:, self._mesh.t[0]]
        self._element_area = 0.5 * np.abs(
            edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        )

        # Lame parameters for plane strain
        nu, e_mod = self.poisson_ratio, self.youngs_modulus
        self._lam = e_mod * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self._mu = e_mod / (2.0 * (1.0 + nu))

        self._clamped = self._basis.get_dofs(lambda x: x[0] < 1e-9)
        self._tip = self._basis.get_dofs(lambda x: x[0] > self.length - 1e-9)

    def _simulate(self, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Run one finite element solve.

        Args:
            x: Strip thicknesses

        Returns:
            Tuple of (mass, peak von Mises stress, absolute tip deflection)
        """
        key = np.ascontiguousarray(x, dtype=float).tobytes()
        if key == self._cache_key:
            return self._cache_value

        from skfem import BilinearForm, asm, condense, solve
        from skfem.helpers import eye, sym_grad, trace, ddot

        started = time.perf_counter()

        thickness = np.asarray(x, dtype=float)[self._group]
        lam, mu = self._lam, self._mu

        @BilinearForm
        def stiffness(u, v, w):
            return w["thickness"] * (
                2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v))
            )

        thickness_field = self._p0_basis.zeros()
        thickness_field[:] = thickness
        stiffness_matrix = asm(
            stiffness,
            self._basis,
            thickness=self._p0_basis.interpolate(thickness_field),
        )

        forces = self._basis.zeros()
        tip_dofs = self._tip.nodal["u^2"]
        forces[tip_dofs] = -self.tip_load / len(tip_dofs)

        displacement = solve(
            *condense(stiffness_matrix, forces, D=self._clamped)
        )

        # Von Mises stress per element, from the strain at the quadrature points
        strain = sym_grad(self._basis.interpolate(displacement))
        stress = 2.0 * mu * strain + lam * trace(strain) * eye(trace(strain), 2)
        s_xx, s_yy, s_xy = stress[0, 0], stress[1, 1], stress[0, 1]
        von_mises = np.sqrt(
            s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2
        )
        # Membrane stress scales inversely with thickness for a plane-stress plate
        von_mises_per_element = von_mises.mean(axis=1) / np.maximum(thickness, 1e-12)

        mass = float(np.sum(self._element_area * thickness))
        peak_stress = float(np.max(von_mises_per_element))
        # Index of the element carrying the peak stress. The stress constraint is a
        # maximum over elements, so it has a kink wherever this index changes.
        self.critical_element = int(np.argmax(von_mises_per_element))
        tip_deflection = float(np.max(np.abs(displacement[tip_dofs])))

        self.solve_count += 1
        self.solve_time += time.perf_counter() - started

        self._cache_key = key
        self._cache_value = (mass, peak_stress, tip_deflection)
        return self._cache_value

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the bracket mass from the finite element model."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        mass, _, _ = self._simulate(x)
        return mass

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the stress, deflection and taper limits (g(x) <= 0).

        The first constraint is a maximum over elements and the taper constraints
        are absolute values, so this function is non-differentiable wherever the
        critical element changes or a taper reverses sign.
        """
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        x = np.asarray(x, dtype=float)
        _, peak_stress, tip_deflection = self._simulate(x)

        g = np.zeros(self.num_ineq_constraints)
        g[0] = peak_stress / self.sigma_allow - 1.0
        g[1] = tip_deflection / self.delta_max - 1.0
        # Manufacturability: adjacent strips may not differ by more than max_taper
        g[2:] = np.abs(np.diff(x)) / self.max_taper - 1.0

        return g

    def reset_instrumentation(self) -> None:
        """Zero the solve counter and timer, e.g. between benchmark runs."""
        self.solve_count = 0
        self.solve_time = 0.0

    @property
    def mean_solve_time(self) -> float:
        """Mean wall-clock seconds per finite element solve, 0.0 if none were run."""
        if self.solve_count == 0:
            return 0.0
        return self.solve_time / self.solve_count


def get_blackbox_problems(include_mopta08: bool = True):
    """
    Return the simulation-based black-box problems.

    MOPTA08 is included only when its executable is already present, so that the
    suite still runs on a machine that has not set it up. It is never downloaded
    implicitly here: fetching 30-90 MB as a side effect of listing the problems
    would be a surprise, and on a compute node it may not even be possible.
    Run ``python benchmarks/scripts/setup_mopta08.py`` to install it.

    Args:
        include_mopta08: If True, include MOPTA08 when its executable is available

    Returns:
        Dict mapping problem name to a freshly instantiated problem

    Example:
        >>> problems = get_blackbox_problems()
        >>> "CantileverBracketFEA-8D" in problems
        True
    """
    problem = CantileverBracketFEA()
    problems = {problem.name: problem}

    if include_mopta08:
        from .mopta08 import MOPTA08, is_available

        if is_available():
            mopta = MOPTA08(auto_download=False)
            problems[mopta.name] = mopta

    return problems
