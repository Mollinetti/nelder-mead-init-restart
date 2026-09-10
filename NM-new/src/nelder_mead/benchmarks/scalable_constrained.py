#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dimension-parameterized constrained problems for the scalability study.

The engineering design problems and the CEC2020 real-world suite are all fixed in
dimension, so they cannot answer the question "how does performance degrade as the
problem grows?" — comparing results across two different fixed problems confounds
dimension with every other property of those problems. The families here vary the
dimension while holding the structure of the objective and the constraints fixed,
so that dimension is the only thing that changes along a sweep.

The families follow the design of the CEC2017/2020 scalable constrained suite (Wu,
Mallipeddi & Suganthan, 2017), where each problem is defined for any n and is
evaluated at n in {10, 30, 50, 100}. They are not transcriptions of specific CEC
functions: the CEC suite's shift and rotation data are distributed as binary files
tied to particular n, whereas the sweep here needs arbitrary n. Each family below
therefore reimplements a documented constraint archetype from that suite in a form
that is well defined for any dimension, with the rotation generated reproducibly
from a seed rather than loaded from a data file.

The families are chosen to cover the structural axes that matter for a
simplex-based method:

- :class:`ScalableSphereInequality` - separable objective, separable constraints.
  The easy baseline; degradation here is attributable to dimension alone.
- :class:`ScalableRotatedInequality` - the same problem after a rotation, so the
  objective and constraints are no longer separable. Isolates the cost of
  coordinate coupling.
- :class:`ScalableEqualityManifold` - equality constraints confining the search to
  an (n-m)-dimensional manifold. The hardest structure for a direct search.
- :class:`ScalableActiveSetInequality` - constraint count scales with dimension and
  most constraints are active at the optimum, which is the regime the paper's
  SRD-11 discussion identifies as the failure mode.

References:
    Wu, G., Mallipeddi, R., & Suganthan, P. N. (2017). Problem definitions and
    evaluation criteria for the CEC 2017 competition on constrained single
    objective real-parameter optimization. Technical Report, Nanyang Technological
    University, Singapore.

    Kumar, A., Wu, G., Ali, M. Z., Mallipeddi, R., Suganthan, P. N., & Das, S.
    (2020). A test-suite of non-convex constrained optimization problems from the
    real-world and some baseline results. Swarm and Evolutionary Computation, 56,
    100693.
"""

from typing import Optional

import numpy as np

from .problem_suite import OptimizationProblem


def _rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """
    Build a reproducible random rotation matrix.

    Uses the QR decomposition of a Gaussian matrix, with the sign correction that
    makes the result uniform over the orthogonal group. Without the correction the
    distribution is biased by LAPACK's choice of signs on R's diagonal.

    Args:
        dim: Matrix dimension
        seed: Seed controlling the draw, so a sweep is reproducible

    Returns:
        An orthogonal (dim, dim) array

    Example:
        >>> rotation = _rotation_matrix(4, seed=1)
        >>> bool(np.allclose(rotation @ rotation.T, np.eye(4)))
        True
    """
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


class ScalableSphereInequality(OptimizationProblem):
    """
    Separable quadratic objective with two separable inequality constraints.

    The baseline of the sweep. Both the objective and the constraints decompose
    across coordinates, so nothing but the dimension itself gets harder as n grows.
    Any degradation observed here is attributable to dimension alone, which makes
    it the control against which the other three families are read.

    Mathematical formulation:
        Minimize: f(x) = sum_i (x_i - 1)^2

        Subject to:
            g1(x) = sum_i x_i - n/2 <= 0
            g2(x) = 1 - sum_i x_i^2 <= 0

        Bounds: -10 <= x_i <= 10

    The unconstrained minimum at x = 1 violates g1 for n >= 3, so the constraint is
    active at the solution and the problem does not degenerate into an
    unconstrained one as n grows.

    Known optimum: for n >= 4 the solution is x_i = 1/2 for all i, giving
    f(x*) = n/4. The symmetric point satisfies g2 only when 1/sqrt(n) <= 1/2, so
    for n < 4 the exclusion constraint binds instead, the closed form does not
    hold and no optimum value is recorded.

    Args:
        dim: Number of decision variables

    Properties:
        - Dimension: configurable
        - Number of inequality constraints: 2 (fixed, independent of dimension)
        - Number of equality constraints: 0

    Example:
        >>> problem = ScalableSphereInequality(dim=10)
        >>> problem.dim
        10
        >>> problem.num_ineq_constraints
        2
    """

    def __init__(self, dim: int = 10):
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")

        # Verified numerically against SLSQP from 40 random starts for
        # n in {4, 5, 10, 20, 30}; see tests/unit/test_scalable_constrained.py
        optimum_value = dim / 4.0 if dim >= 4 else None
        optimum_location = np.full(dim, 0.5) if dim >= 4 else None

        super().__init__(
            name=f"ScalableSphere-{dim}D",
            dim=dim,
            bounds=(np.full(dim, -10.0), np.full(dim, 10.0)),
            optimum_value=optimum_value,
            optimum_location=optimum_location,
            num_eq_constraints=0,
            num_ineq_constraints=2,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the shifted sphere objective."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return float(np.sum((x - 1.0) ** 2))

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the budget and exclusion constraints (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(2)
        g[0] = float(np.sum(x)) - self.dim / 2.0
        g[1] = 1.0 - float(np.sum(x**2))

        return g


class ScalableRotatedInequality(OptimizationProblem):
    """
    The same problem as :class:`ScalableSphereInequality` after a rotation.

    Applying an orthogonal transformation leaves the optimum value unchanged but
    destroys separability, so the objective and constraints couple every coordinate
    to every other. Comparing this family against the unrotated one separates the
    cost of coordinate coupling from the cost of dimension, which matters for a
    simplex method: the simplex adapts to the local landscape and should in
    principle be rotation-invariant, whereas a coordinate-wise method is not.

    Mathematical formulation:
        Let z = R(x - 1) for a fixed orthogonal R.

        Minimize: f(x) = sum_i z_i^2 + sum_i (z_i * z_{i+1})

        Subject to:
            g1(x) = sum_i z_i - n/2 <= 0
            g2(x) = 1 - sum_i z_i^2 <= 0

        Bounds: -10 <= x_i <= 10

    The cross term makes the objective non-separable even before the rotation, so
    the family stays hard if the rotation is ever removed.

    Args:
        dim: Number of decision variables
        seed: Seed for the rotation matrix, so a sweep is reproducible

    Properties:
        - Dimension: configurable
        - Number of inequality constraints: 2 (fixed, independent of dimension)
        - Number of equality constraints: 0

    Example:
        >>> problem = ScalableRotatedInequality(dim=10)
        >>> problem.dim
        10
    """

    def __init__(self, dim: int = 10, seed: int = 2020):
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")

        self.rotation = _rotation_matrix(dim, seed)
        self.seed = seed

        super().__init__(
            name=f"ScalableRotated-{dim}D",
            dim=dim,
            bounds=(np.full(dim, -10.0), np.full(dim, 10.0)),
            num_eq_constraints=0,
            num_ineq_constraints=2,
        )

    def _rotate(self, x: np.ndarray) -> np.ndarray:
        """Map the decision vector into the rotated frame."""
        return self.rotation @ (x - 1.0)

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the rotated, non-separable objective."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        z = self._rotate(x)
        return float(np.sum(z**2) + np.sum(z[:-1] * z[1:]))

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the rotated budget and exclusion constraints (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        z = self._rotate(x)
        g = np.zeros(2)
        g[0] = float(np.sum(z)) - self.dim / 2.0
        g[1] = 1.0 - float(np.sum(z**2))

        return g


class ScalableEqualityManifold(OptimizationProblem):
    """
    Quadratic objective confined to a nonlinear equality manifold.

    Equality constraints reduce the feasible set to a measure-zero manifold of
    dimension n - m. A simplex whose vertices are all infeasible has no gradient
    information pointing back onto the manifold, which is precisely the situation
    the paper's augmented Lagrangian formulation is designed to handle. This family
    lets the manifold's codimension be varied alongside the dimension.

    Mathematical formulation:
        Minimize: f(x) = sum_i (x_i - 1)^2

        Subject to, for j = 1..m:
            h_j(x) = sum_{i in block j} x_i^2 - 1 = 0

        where the coordinates are partitioned into m contiguous blocks.

        Bounds: -5 <= x_i <= 5

    Each constraint confines one block to the surface of a unit sphere, so the
    feasible set is a product of spheres: nonlinear, connected, and of dimension
    n - m.

    Known optimum: within a block of size k the objective is minimized by the
    projection of the all-ones vector onto the unit sphere, x_i = 1/sqrt(k),
    contributing k*(1/sqrt(k) - 1)^2 = (sqrt(k) - 1)^2. Summing over blocks gives
    f(x*) = sum_j (sqrt(k_j) - 1)^2.

    Args:
        dim: Number of decision variables
        num_eq: Number of equality constraints (blocks). Defaults to max(1, dim//5)

    Properties:
        - Dimension: configurable
        - Number of equality constraints: configurable
        - Number of inequality constraints: 0

    Example:
        >>> problem = ScalableEqualityManifold(dim=10, num_eq=2)
        >>> problem.num_eq_constraints
        2
        >>> len(problem.constraint_eq(np.ones(10)))
        2
    """

    def __init__(self, dim: int = 10, num_eq: Optional[int] = None):
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")

        if num_eq is None:
            num_eq = max(1, dim // 5)

        if not 1 <= num_eq <= dim:
            raise ValueError(
                f"num_eq must be between 1 and dim ({dim}), got {num_eq}"
            )

        # Contiguous, near-equal blocks; np.array_split handles the remainder
        self.blocks = [
            np.asarray(block) for block in np.array_split(np.arange(dim), num_eq)
        ]

        # Verified numerically against SLSQP from 40 random starts
        optimum_location = np.concatenate(
            [np.full(len(block), 1.0 / np.sqrt(len(block))) for block in self.blocks]
        )
        optimum_value = float(
            sum((np.sqrt(len(block)) - 1.0) ** 2 for block in self.blocks)
        )

        super().__init__(
            name=f"ScalableEquality-{dim}D-{num_eq}h",
            dim=dim,
            bounds=(np.full(dim, -5.0), np.full(dim, 5.0)),
            optimum_value=optimum_value,
            optimum_location=optimum_location,
            num_eq_constraints=num_eq,
            num_ineq_constraints=0,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the shifted sphere objective."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return float(np.sum((x - 1.0) ** 2))

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the per-block unit-sphere constraints (h(x) = 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return np.array(
            [float(np.sum(x[block] ** 2)) - 1.0 for block in self.blocks]
        )


class ScalableActiveSetInequality(OptimizationProblem):
    """
    Inequality count scaling with dimension, most of them active at the optimum.

    The paper attributes c-NM's degradation on SRD-11 to two causes: more decision
    variables, and more constraints confining the search to a narrow region. The
    other families here vary the first while holding the second fixed. This one
    varies both together, so that the constraint-to-variable ratio stays constant
    as n grows, and it places the constraints so that most of them are active at
    the solution rather than slack.

    Mathematical formulation:
        Minimize: f(x) = sum_i (x_i - 2)^2

        Subject to, for i = 1..n:
            g_i(x) = x_i + x_{(i mod n) + 1} - 1 <= 0

        Bounds: -10 <= x_i <= 10

    The unconstrained minimum at x = 2 violates every constraint, so the solution
    lies on the boundary with all n constraints active. The cyclic coupling means
    no constraint can be satisfied by adjusting a single coordinate.

    Known optimum (for the default constraints_per_dim = 1): the objective is
    convex and both it and the feasible set are invariant under cyclic shift, so
    averaging any solution over the n shifts gives a symmetric solution that is no
    worse. Setting x_i = t reduces the constraints to 2t <= 1, so t = 1/2 and
    f(x*) = n*(1/2 - 2)^2 = 2.25*n.

    Args:
        dim: Number of decision variables
        constraints_per_dim: Constraints per decision variable (default 1)

    Properties:
        - Dimension: configurable
        - Number of inequality constraints: dim * constraints_per_dim
        - Number of equality constraints: 0

    Example:
        >>> problem = ScalableActiveSetInequality(dim=10)
        >>> problem.num_ineq_constraints
        10
        >>> problem = ScalableActiveSetInequality(dim=10, constraints_per_dim=2)
        >>> problem.num_ineq_constraints
        20
    """

    def __init__(self, dim: int = 10, constraints_per_dim: int = 1):
        if dim < 2:
            raise ValueError(f"dim must be at least 2, got {dim}")

        if constraints_per_dim < 1:
            raise ValueError(
                f"constraints_per_dim must be at least 1, got {constraints_per_dim}"
            )

        self.constraints_per_dim = constraints_per_dim

        # Only the default coupling has the symmetric closed form; verified
        # numerically against SLSQP from 40 random starts
        optimum_value = 2.25 * dim if constraints_per_dim == 1 else None
        optimum_location = np.full(dim, 0.5) if constraints_per_dim == 1 else None

        super().__init__(
            name=f"ScalableActiveSet-{dim}D-{dim * constraints_per_dim}g",
            dim=dim,
            bounds=(np.full(dim, -10.0), np.full(dim, 10.0)),
            optimum_value=optimum_value,
            optimum_location=optimum_location,
            num_eq_constraints=0,
            num_ineq_constraints=dim * constraints_per_dim,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the shifted sphere objective."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return float(np.sum((x - 2.0) ** 2))

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the cyclically coupled pairwise budgets (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        constraints = []
        for offset in range(1, self.constraints_per_dim + 1):
            partner = np.roll(x, -offset)
            constraints.append(x + partner - 1.0)

        return np.concatenate(constraints)


#: The families swept in the scalability study, keyed by a short label.
SCALABLE_FAMILIES = {
    "sphere": ScalableSphereInequality,
    "rotated": ScalableRotatedInequality,
    "equality": ScalableEqualityManifold,
    "active_set": ScalableActiveSetInequality,
}

#: Default dimension ladder. It brackets the paper's classic suite (n <= 7) on both
#: sides so the sweep reproduces the known regime before extrapolating past it.
DEFAULT_DIMENSIONS = (2, 5, 10, 20, 30, 50)


def get_scalability_sweep(dimensions=DEFAULT_DIMENSIONS, families=None):
    """
    Build the problem grid for the scalability study.

    Args:
        dimensions: Iterable of dimensions to instantiate
        families: Iterable of family labels from :data:`SCALABLE_FAMILIES`.
                  Defaults to all of them.

    Returns:
        Dict mapping family label to a list of (dimension, problem) tuples,
        ordered by increasing dimension

    Raises:
        ValueError: If a family label is not recognised

    Example:
        >>> sweep = get_scalability_sweep(dimensions=(2, 5), families=["sphere"])
        >>> sorted(sweep)
        ['sphere']
        >>> [dim for dim, _ in sweep["sphere"]]
        [2, 5]
    """
    if families is None:
        families = list(SCALABLE_FAMILIES)

    unknown = set(families) - set(SCALABLE_FAMILIES)
    if unknown:
        raise ValueError(
            f"Unknown scalable families: {sorted(unknown)}. "
            f"Available: {sorted(SCALABLE_FAMILIES)}"
        )

    return {
        family: [
            (dim, SCALABLE_FAMILIES[family](dim=dim)) for dim in sorted(dimensions)
        ]
        for family in families
    }
