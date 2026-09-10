#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A subset of the CEC2020 real-world constrained optimization test-suite.

The suite of Kumar et al. (2020) collects 57 constrained problems drawn from
industrial chemical processes, process design and synthesis, mechanical design,
power systems, power electronics and livestock feed ration optimization. It is the
field's post-2020 answer to the criticism that constrained-optimization papers
re-use the same handful of small analytic design problems.

The twelve problems implemented here were selected to span a range of dimension
and constraint count well beyond the classic engineering suite: dimension runs from
2 to 38, inequality counts from 0 to 15 and equality counts from 0 to 32. Four of
the classic engineering problems reappear here in the suite's own formulation
(speed reducer, tension/compression spring, welded beam), which gives a direct
bridge between the two sets of results.

All formulations are transcribed from the official competition implementation
``cec20_func.m`` and the bounds from ``Cal_par.m``, distributed by the competition
organizers. They are not copied from secondary sources, because the coefficients of
these problems have drifted measurably across the citation chain.

Two discrepancies in the official sources are worth recording, since anyone
comparing against published baselines will hit them:

1. RC17 (tension/compression spring). ``Cal_par.m`` declares 3 inequality
   constraints, while ``cec20_func.m`` computes 4. The fourth,
   ``(x1 + x2)/1.5 - 1 <= 0``, is the standard outer-diameter limit and is present
   in every literature formulation of this problem, so all 4 are implemented here
   and :attr:`num_ineq_constraints` is 4.
2. Problems with integer-valued variables (RC08, RC14) are rounded inside the
   objective and constraint functions exactly as the official code does, rather
   than being handled by the optimizer. A continuous optimizer therefore sees a
   piecewise-constant landscape in those variables, which is the intended
   difficulty.

References:
    Kumar, A., Wu, G., Ali, M. Z., Mallipeddi, R., Suganthan, P. N., & Das, S.
    (2020). A test-suite of non-convex constrained optimization problems from the
    real-world and some baseline results. Swarm and Evolutionary Computation, 56,
    100693. https://doi.org/10.1016/j.swevo.2020.100693

    Official implementation: cec20_func.m / Cal_par.m, CEC2020 competition on
    real-world single objective constrained optimization.
    https://github.com/P-N-Suganthan/2020-RW-Constrained-Optimisation
"""

import numpy as np

from .problem_suite import OptimizationProblem


# The official code guards its logarithms with this epsilon; reproduced verbatim
# because it perturbs the equality residuals at the fourth significant figure.
_LOG_EPS = 1e-8


class RC01(OptimizationProblem):
    """
    Heat Exchanger Network Design (case 1).

    Industrial chemical process problem. Sizes a two-stage heat exchanger network
    to minimize installed area cost subject to energy and temperature balances.

    Mathematical formulation:
        Minimize: f(x) = 35*x1^0.6 + 35*x2^0.6

        Subject to (equalities):
            h1 = 200*x1*x4 - x3
            h2 = 200*x2*x6 - x5
            h3 = x3 - 10000*(x7 - 100)
            h4 = x5 - 10000*(300 - x7)
            h5 = x3 - 10000*(600 - x8)
            h6 = x5 - 10000*(900 - x9)
            h7 = x4*ln(|x8 - 100| + eps) - x4*ln(600 - x7 + eps) - x8 + x7 + 500
            h8 = x6*ln(|x9 - x7| + eps) - x6*ln(600) - x9 + x7 + 600

    Properties:
        - Dimension: 9
        - Number of equality constraints: 8
        - Number of inequality constraints: 0
        - Equality-dominated: the feasible set is a thin manifold, which is the
          hardest structure in the suite for a direct-search method

    Reference:
        Kumar et al. (2020), problem RC01.

    Example:
        >>> problem = RC01()
        >>> problem.dim
        9
        >>> problem.num_eq_constraints
        8
    """

    def __init__(self):
        super().__init__(
            name="RC01",
            dim=9,
            bounds=(
                np.array([0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 100.0, 100.0, 100.0]),
                np.array(
                    [10.0, 200.0, 100.0, 200.0, 2000000.0, 600.0, 600.0, 600.0, 900.0]
                ),
            ),
            num_eq_constraints=8,
            num_ineq_constraints=0,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the heat exchanger network cost."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return 35.0 * x[0] ** 0.6 + 35.0 * x[1] ** 0.6

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the energy and temperature balance equalities (h(x) = 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        h = np.zeros(8)
        h[0] = 200.0 * x[0] * x[3] - x[2]
        h[1] = 200.0 * x[1] * x[5] - x[4]
        h[2] = x[2] - 10000.0 * (x[6] - 100.0)
        h[3] = x[4] - 10000.0 * (300.0 - x[6])
        h[4] = x[2] - 10000.0 * (600.0 - x[7])
        h[5] = x[4] - 10000.0 * (900.0 - x[8])
        h[6] = (
            x[3] * np.log(abs(x[7] - 100.0) + _LOG_EPS)
            - x[3] * np.log((600.0 - x[6]) + _LOG_EPS)
            - x[7]
            + x[6]
            + 500.0
        )
        h[7] = (
            x[5] * np.log(abs(x[8] - x[6]) + _LOG_EPS)
            - x[5] * np.log(600.0)
            - x[8]
            + x[6]
            + 600.0
        )

        return h


class RC02(OptimizationProblem):
    """
    Heat Exchanger Network Design (case 2).

    The three-stage variant of :class:`RC01`, with a fractional-power objective in
    the exchanger areas rather than the flow rates.

    Mathematical formulation:
        Minimize: f(x) = (x1/(120*x4))^0.6 + (x2/(80*x5))^0.6 + (x3/(40*x6))^0.6

        Subject to 9 equality constraints (energy and temperature balances).

    Properties:
        - Dimension: 11
        - Number of equality constraints: 9
        - Number of inequality constraints: 0
        - Badly scaled: variable ranges span 1e4 to 2.05e6 alongside ranges of
          width 0.05, so any method sensitive to conditioning will struggle

    Reference:
        Kumar et al. (2020), problem RC02.

    Example:
        >>> problem = RC02()
        >>> problem.dim
        11
    """

    def __init__(self):
        super().__init__(
            name="RC02",
            dim=11,
            bounds=(
                np.array(
                    [1e4, 1e4, 1e4, 0.0, 0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0]
                ),
                np.array(
                    [
                        0.819e6,
                        1.131e6,
                        2.05e6,
                        0.05074,
                        0.05074,
                        0.05074,
                        200.0,
                        300.0,
                        300.0,
                        300.0,
                        400.0,
                    ]
                ),
            ),
            num_eq_constraints=9,
            num_ineq_constraints=0,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the three-stage heat exchanger network cost."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (
            (x[0] / (120.0 * x[3])) ** 0.6
            + (x[1] / (80.0 * x[4])) ** 0.6
            + (x[2] / (40.0 * x[5])) ** 0.6
        )

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the energy and temperature balance equalities (h(x) = 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        h = np.zeros(9)
        h[0] = x[0] - 1e4 * (x[6] - 100.0)
        h[1] = x[1] - 1e4 * (x[7] - x[6])
        h[2] = x[2] - 1e4 * (500.0 - x[7])
        h[3] = x[0] - 1e4 * (300.0 - x[8])
        h[4] = x[1] - 1e4 * (400.0 - x[9])
        h[5] = x[2] - 1e4 * (600.0 - x[10])
        h[6] = (
            x[3] * np.log(abs(x[8] - 100.0) + _LOG_EPS)
            - x[3] * np.log(300.0 - x[6] + _LOG_EPS)
            - x[8]
            - x[6]
            + 400.0
        )
        h[7] = (
            x[4] * np.log(abs(x[9] - x[6]) + _LOG_EPS)
            - x[4] * np.log(abs(400.0 - x[7]) + _LOG_EPS)
            - x[9]
            + x[6]
            - x[7]
            + 400.0
        )
        h[8] = (
            x[5] * np.log(abs(x[10] - x[7]) + _LOG_EPS)
            - x[5] * np.log(100.0)
            - x[10]
            + x[7]
            + 100.0
        )

        return h


class RC03(OptimizationProblem):
    """
    Optimal Operation of Alkylation Unit.

    Industrial chemical process problem. Maximizes the profit of an alkylation
    process, expressed here as a minimization of its negative.

    Mathematical formulation:
        Minimize: f(x) = -1.715*x1 - 0.035*x1*x6 - 4.0565*x3 - 10*x2 + 0.063*x3*x5

        Subject to 14 inequality constraints (g(x) <= 0) describing the reactor
        and fractionator operating envelope.

    Properties:
        - Dimension: 7
        - Number of inequality constraints: 14
        - Number of equality constraints: 0
        - Inequality-dominated with twice the constraints of decision variables,
          the structure the paper's SRD-11 discussion identifies as difficult

    Reference:
        Kumar et al. (2020), problem RC03.

    Example:
        >>> problem = RC03()
        >>> problem.num_ineq_constraints
        14
    """

    def __init__(self):
        super().__init__(
            name="RC03",
            dim=7,
            bounds=(
                np.array([1000.0, 0.0, 2000.0, 0.0, 0.0, 0.0, 0.0]),
                np.array([2000.0, 100.0, 4000.0, 100.0, 100.0, 20.0, 200.0]),
            ),
            num_eq_constraints=0,
            num_ineq_constraints=14,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the negated alkylation unit profit."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (
            -1.715 * x[0]
            - 0.035 * x[0] * x[5]
            - 4.0565 * x[2]
            - 10.0 * x[1]
            + 0.063 * x[2] * x[4]
        )

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the operating envelope inequalities (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(14)
        g[0] = (
            0.0059553571 * x[5] ** 2 * x[0]
            + 0.88392857 * x[2]
            - 0.1175625 * x[5] * x[0]
            - x[0]
        )
        g[1] = (
            1.1088 * x[0]
            + 0.1303533 * x[0] * x[5]
            - 0.0066033 * x[0] * x[5] ** 2
            - x[2]
        )
        g[2] = (
            6.66173269 * x[5] ** 2
            + 172.39878 * x[4]
            - 56.596669 * x[3]
            - 191.20592 * x[5]
            - 10000.0
        )
        g[3] = (
            1.08702 * x[5] + 0.32175 * x[3] - 0.03762 * x[5] ** 2 - x[4] + 56.85075
        )
        g[4] = (
            0.006198 * x[6] * x[3] * x[2]
            + 2462.3121 * x[1]
            - 25.125634 * x[1] * x[3]
            - x[2] * x[3]
        )
        g[5] = (
            161.18996 * x[2] * x[3]
            + 5000.0 * x[1] * x[3]
            - 489510.0 * x[1]
            - x[2] * x[3] * x[6]
        )
        g[6] = 0.33 * x[6] - x[4] + 44.333333
        g[7] = 0.022556 * x[4] - 0.007595 * x[6] - 1.0
        g[8] = 0.00061 * x[2] - 0.0005 * x[0] - 1.0
        g[9] = 0.819672 * x[0] - x[2] + 0.819672
        g[10] = 24500.0 * x[1] - 250.0 * x[1] * x[3] - x[2] * x[3]
        g[11] = (
            1020.4082 * x[3] * x[1] + 1.2244898 * x[2] * x[3] - 100000.0 * x[1]
        )
        g[12] = 6.25 * x[0] * x[5] + 6.25 * x[0] - 7.625 * x[2] - 100000.0
        g[13] = 1.22 * x[2] - x[5] * x[0] - x[0] + 1.0

        return g


class RC04(OptimizationProblem):
    """
    Reactor Network Design.

    Industrial chemical process problem. Maximizes the concentration of the desired
    product leaving a two-reactor network, expressed as a minimization of -x4.

    Mathematical formulation:
        Minimize: f(x) = -x4

        Subject to (equalities), with k1 = 0.09755988, k2 = 0.99*k1,
        k3 = 0.0391908, k4 = 0.9*k3:
            h1 = x1 + k1*x2*x5 - 1
            h2 = x2 - x1 + k2*x2*x6
            h3 = x3 + x1 + k3*x3*x5 - 1
            h4 = x4 - x3 + x2 - x1 + k4*x4*x6

        Subject to (inequality):
            g1 = sqrt(x5) + sqrt(x6) - 4

    Properties:
        - Dimension: 6
        - Number of equality constraints: 4
        - Number of inequality constraints: 1
        - Mixed equality and inequality structure

    Reference:
        Kumar et al. (2020), problem RC04.

    Example:
        >>> problem = RC04()
        >>> problem.num_eq_constraints, problem.num_ineq_constraints
        (4, 1)
    """

    # Reaction rate constants, from the official implementation
    K1 = 0.09755988
    K2 = 0.99 * 0.09755988
    K3 = 0.0391908
    K4 = 0.9 * 0.0391908

    def __init__(self):
        super().__init__(
            name="RC04",
            dim=6,
            bounds=(
                np.array([0.0, 0.0, 0.0, 0.0, 1e-5, 1e-5]),
                np.array([1.0, 1.0, 1.0, 1.0, 16.0, 16.0]),
            ),
            num_eq_constraints=4,
            num_ineq_constraints=1,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the negated outlet concentration of the desired product."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return -x[3]

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the species mass balances (h(x) = 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        h = np.zeros(4)
        h[0] = x[0] + self.K1 * x[1] * x[4] - 1.0
        h[1] = x[1] - x[0] + self.K2 * x[1] * x[5]
        h[2] = x[2] + x[0] + self.K3 * x[2] * x[4] - 1.0
        h[3] = x[3] - x[2] + x[1] - x[0] + self.K4 * x[3] * x[5]

        return h

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the total residence time limit (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return np.array([x[4] ** 0.5 + x[5] ** 0.5 - 4.0])


class RC05(OptimizationProblem):
    """
    Haverly's Pooling Problem.

    Industrial chemical process problem, and the canonical example of a bilinear
    pooling formulation: the products of flow and quality variables make the
    feasible region non-convex and create local optima that trap local methods.

    Mathematical formulation:
        Minimize: f(x) = -(9*x1 + 15*x2 - 6*x3 - 16*x4 - 10*(x5 + x6))

        Subject to (inequalities):
            g1 = x9*x7 + 2*x5 - 2.5*x1
            g2 = x9*x8 + 2*x6 - 1.5*x2

        Subject to (equalities):
            h1 = x7 + x8 - x3 - x4
            h2 = x1 - x7 - x5
            h3 = x2 - x8 - x6
            h4 = x9*x7 + x9*x8 - 3*x3 - x4

    Properties:
        - Dimension: 9
        - Number of inequality constraints: 2
        - Number of equality constraints: 4
        - Bilinear and non-convex

    Reference:
        Kumar et al. (2020), problem RC05.

    Example:
        >>> problem = RC05()
        >>> problem.dim
        9
    """

    def __init__(self):
        super().__init__(
            name="RC05",
            dim=9,
            bounds=(
                np.zeros(9),
                np.array(
                    [100.0, 200.0, 100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 200.0]
                ),
            ),
            num_eq_constraints=4,
            num_ineq_constraints=2,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the negated pooling network profit."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return -(
            9.0 * x[0]
            + 15.0 * x[1]
            - 6.0 * x[2]
            - 16.0 * x[3]
            - 10.0 * (x[4] + x[5])
        )

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the flow and quality balances (h(x) = 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        h = np.zeros(4)
        h[0] = x[6] + x[7] - x[2] - x[3]
        h[1] = x[0] - x[6] - x[4]
        h[2] = x[1] - x[7] - x[5]
        h[3] = x[8] * x[6] + x[8] * x[7] - 3.0 * x[2] - x[3]

        return h

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the product quality limits (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(2)
        g[0] = x[8] * x[6] + 2.0 * x[4] - 2.5 * x[0]
        g[1] = x[8] * x[7] + 2.0 * x[5] - 1.5 * x[1]

        return g


class RC06(OptimizationProblem):
    """
    Blending-Pooling-Separation Problem.

    The highest-dimensional problem in this subset, and the reason it is here: 38
    decision variables tied together by 32 equality constraints. The feasible set is
    a low-dimensional manifold in a 38-dimensional box, which is precisely the
    regime where a simplex-based direct search is expected to degrade.

    Mathematical formulation:
        Minimize: f(x) = 0.9979 + 0.00432*x5 + 0.01517*x13

        Subject to 32 equality constraints describing stream splits, component
        balances across three separation trains, and mass-fraction normalization.

    Properties:
        - Dimension: 38
        - Number of equality constraints: 32
        - Number of inequality constraints: 0
        - Only 2 of the 38 variables enter the objective; the rest are determined
          almost entirely by feasibility

    Reference:
        Kumar et al. (2020), problem RC06.

    Example:
        >>> problem = RC06()
        >>> problem.dim, problem.num_eq_constraints
        (38, 32)
    """

    def __init__(self):
        upper = np.array(
            [
                90.0, 150.0, 90.0, 150.0, 90.0, 90.0, 150.0, 90.0, 90.0, 90.0,
                150.0, 150.0, 90.0, 90.0, 150.0, 90.0, 150.0, 90.0, 150.0, 90.0,
                1.0, 1.2, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 0.5,
                0.5, 1.2, 0.5, 1.2, 1.2, 0.5, 1.2, 1.2,
            ]
        )
        super().__init__(
            name="RC06",
            dim=38,
            bounds=(np.zeros(38), upper),
            num_eq_constraints=32,
            num_ineq_constraints=0,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the blending-pooling-separation cost."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return 0.9979 + 0.00432 * x[4] + 0.01517 * x[12]

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the stream, component and normalization balances (h(x) = 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        h = np.zeros(32)
        # Overall and node stream balances
        h[0] = x[0] + x[1] + x[2] + x[3] - 300.0
        h[1] = x[5] - x[6] - x[7]
        h[2] = x[8] - x[9] - x[10] - x[11]
        h[3] = x[13] - x[14] - x[15] - x[16]
        h[4] = x[17] - x[18] - x[19]
        # Component balances around the first separation train
        h[5] = x[4] * x[20] - x[5] * x[21] - x[8] * x[22]
        h[6] = x[4] * x[23] - x[5] * x[24] - x[8] * x[25]
        h[7] = x[4] * x[26] - x[5] * x[27] - x[8] * x[28]
        # Component balances around the second separation train
        h[8] = x[12] * x[29] - x[13] * x[30] - x[17] * x[31]
        h[9] = x[12] * x[32] - x[13] * x[33] - x[17] * x[34]
        h[10] = x[12] * x[35] - x[13] * x[36] - x[17] * x[37]
        # Recycle balances
        h[11] = (1.0 / 3.0) * x[0] + x[14] * x[30] - x[4] * x[20]
        h[12] = (1.0 / 3.0) * x[0] + x[14] * x[33] - x[4] * x[23]
        h[13] = (1.0 / 3.0) * x[0] + x[14] * x[36] - x[4] * x[26]
        h[14] = (1.0 / 3.0) * x[1] + x[9] * x[22] - x[12] * x[29]
        h[15] = (1.0 / 3.0) * x[1] + x[9] * x[25] - x[12] * x[32]
        h[16] = (1.0 / 3.0) * x[1] + x[9] * x[28] - x[12] * x[35]
        # Product specifications
        h[17] = (
            (1.0 / 3.0) * x[2]
            + x[6] * x[21]
            + x[10] * x[22]
            + x[15] * x[30]
            + x[18] * x[31]
            - 30.0
        )
        h[18] = (
            (1.0 / 3.0) * x[2]
            + x[6] * x[24]
            + x[10] * x[25]
            + x[15] * x[33]
            + x[18] * x[34]
            - 50.0
        )
        h[19] = (
            (1.0 / 3.0) * x[2]
            + x[6] * x[27]
            + x[10] * x[28]
            + x[15] * x[36]
            + x[18] * x[37]
            - 30.0
        )
        # Mass-fraction normalization
        h[20] = x[20] + x[23] + x[26] - 1.0
        h[21] = x[21] + x[24] + x[27] - 1.0
        h[22] = x[22] + x[25] + x[28] - 1.0
        h[23] = x[29] + x[32] + x[35] - 1.0
        h[24] = x[30] + x[33] + x[36] - 1.0
        h[25] = x[31] + x[34] + x[37] - 1.0
        # Purity constraints forcing individual fractions to zero
        h[26] = x[24]
        h[27] = x[27]
        h[28] = x[22]
        h[29] = x[36]
        h[30] = x[31]
        h[31] = x[34]

        return h


class RC08(OptimizationProblem):
    """
    Process Synthesis Problem.

    The smallest problem in the suite: two variables, one of them integer-valued.
    It is included as the low-dimension end of the sweep, and because the rounding
    of x2 makes the landscape piecewise constant in one coordinate, which is a
    genuinely non-smooth feature rather than a smooth surrogate for one.

    Mathematical formulation:
        Minimize: f(x) = 2*x1 + x2,  where x2 is rounded to the nearest integer

        Subject to:
            g1 = 1.25 - x1^2 - x2 <= 0
            g2 = x1 + x2 - 1.6 <= 0

    Properties:
        - Dimension: 2
        - Number of inequality constraints: 2
        - Number of equality constraints: 0
        - Mixed-integer: x2 is rounded inside the objective and constraints

    Reference:
        Kumar et al. (2020), problem RC08.

    Example:
        >>> problem = RC08()
        >>> problem.dim
        2
    """

    def __init__(self):
        super().__init__(
            name="RC08",
            dim=2,
            bounds=(np.array([0.0, -0.51]), np.array([1.6, 1.49])),
            num_eq_constraints=0,
            num_ineq_constraints=2,
        )

    @staticmethod
    def _discretize(x: np.ndarray) -> np.ndarray:
        """Round the integer-valued second variable, as the official code does."""
        x = np.asarray(x, dtype=float).copy()
        x[1] = np.round(x[1])
        return x

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the process synthesis cost."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        x = self._discretize(x)
        return 2.0 * x[0] + x[1]

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the synthesis feasibility constraints (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        x = self._discretize(x)
        g = np.zeros(2)
        g[0] = 1.25 - x[0] ** 2 - x[1]
        g[1] = x[0] + x[1] - 1.6

        return g


class RC14(OptimizationProblem):
    """
    Multi-Product Batch Plant.

    Process design problem sizing three batch units for two products. Three of the
    ten variables are integer-valued unit counts.

    Mathematical formulation:
        Minimize: f(x) = alpha * (N1*V1^beta + N2*V2^beta + N3*V3^beta)

        with alpha = 250, beta = 0.6, horizon H = 6000, demands Q1 = 40000 and
        Q2 = 20000, size factors S and processing times t as tabulated in the
        official implementation.

        Subject to 10 inequality constraints: the horizon limit, three volume
        constraints and six cycle-time constraints.

    Properties:
        - Dimension: 10
        - Number of inequality constraints: 10
        - Number of equality constraints: 0
        - Mixed-integer: N1, N2 and N3 are rounded

    Reference:
        Kumar et al. (2020), problem RC14.

    Example:
        >>> problem = RC14()
        >>> problem.dim, problem.num_ineq_constraints
        (10, 10)
    """

    # Size factors S[product, stage] and processing times t[product, stage]
    S = np.array([[2.0, 3.0, 4.0], [4.0, 6.0, 3.0]])
    T = np.array([[8.0, 20.0, 8.0], [16.0, 4.0, 4.0]])
    H = 6000.0
    ALPHA = 250.0
    BETA = 0.6
    Q1 = 40000.0
    Q2 = 20000.0

    def __init__(self):
        super().__init__(
            name="RC14",
            dim=10,
            bounds=(
                np.array([0.51, 0.51, 0.51, 250.0, 250.0, 250.0, 6.0, 4.0, 40.0, 10.0]),
                np.array(
                    [3.49, 3.49, 3.49, 2500.0, 2500.0, 2500.0, 20.0, 16.0, 700.0, 450.0]
                ),
            ),
            num_eq_constraints=0,
            num_ineq_constraints=10,
        )

    @staticmethod
    def _unpack(x: np.ndarray):
        """Split into unit counts (rounded), volumes, cycle times and batch sizes."""
        n1, n2, n3 = (float(np.round(v)) for v in x[0:3])
        v1, v2, v3 = x[3], x[4], x[5]
        tl1, tl2 = x[6], x[7]
        b1, b2 = x[8], x[9]
        return n1, n2, n3, v1, v2, v3, tl1, tl2, b1, b2

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the total installed cost of the batch units."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        n1, n2, n3, v1, v2, v3, _, _, _, _ = self._unpack(x)
        return self.ALPHA * (
            n1 * v1**self.BETA + n2 * v2**self.BETA + n3 * v3**self.BETA
        )

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the horizon, volume and cycle-time limits (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        n1, n2, n3, v1, v2, v3, tl1, tl2, b1, b2 = self._unpack(x)

        g = np.zeros(10)
        g[0] = self.Q1 * tl1 / b1 + self.Q2 * tl2 / b2 - self.H
        g[1] = self.S[0, 0] * b1 + self.S[1, 0] * b2 - v1
        g[2] = self.S[0, 1] * b1 + self.S[1, 1] * b2 - v2
        g[3] = self.S[0, 2] * b1 + self.S[1, 2] * b2 - v3
        g[4] = self.T[0, 0] - n1 * tl1
        g[5] = self.T[0, 1] - n2 * tl1
        g[6] = self.T[0, 2] - n3 * tl1
        g[7] = self.T[1, 0] - n1 * tl2
        g[8] = self.T[1, 1] - n2 * tl2
        g[9] = self.T[1, 2] - n3 * tl2

        return g


class RC15(OptimizationProblem):
    """
    Weight Minimization of a Speed Reducer.

    The suite's formulation of the speed reducer, the same problem the paper
    analyses as SRD-11 and identifies as the case where c-NM degrades. The bounds
    and the objective match the paper's equation (37); the constraints are written
    here in the suite's own algebraic arrangement, which is equivalent but not
    term-for-term identical to the paper's ratio form.

    Mathematical formulation:
        Minimize: f(x) = 0.7854*x1*x2^2*(3.3333*x3^2 + 14.9334*x3 - 43.0934)
                         - 1.508*x1*(x6^2 + x7^2) + 7.477*(x6^3 + x7^3)
                         + 0.7854*(x4*x6^2 + x5*x7^2)

        Subject to 11 inequality constraints: bending and surface stress on the
        gear teeth, shaft deflection, transverse deflection, and dimensional limits.

    Properties:
        - Dimension: 7
        - Number of inequality constraints: 11
        - Number of equality constraints: 0
        - Narrow variable ranges, which the paper notes hinders construction of an
          initial simplex with a feasible vertex

    Reference:
        Kumar et al. (2020), problem RC15. Originally Golinski (1973).

    Example:
        >>> problem = RC15()
        >>> problem.dim, problem.num_ineq_constraints
        (7, 11)
    """

    def __init__(self):
        super().__init__(
            name="RC15",
            dim=7,
            bounds=(
                np.array([2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]),
                np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]),
            ),
            num_eq_constraints=0,
            num_ineq_constraints=11,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the total weight of the speed reducer."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (
            0.7854
            * x[0]
            * x[1] ** 2
            * (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934)
            - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2)
            + 7.477 * (x[5] ** 3 + x[6] ** 3)
            + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)
        )

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the stress, deflection and dimensional limits (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(11)
        g[0] = -x[0] * x[1] ** 2 * x[2] + 27.0
        g[1] = -x[0] * x[1] ** 2 * x[2] ** 2 + 397.5
        g[2] = -x[1] * x[5] ** 4 * x[2] * x[3] ** (-3) + 1.93
        g[3] = -x[1] * x[6] ** 4 * x[2] / x[4] ** 3 + 1.93
        g[4] = (
            10.0
            * x[5] ** (-3)
            * np.sqrt(16.91e6 + (745.0 * x[3] / (x[1] * x[2])) ** 2)
            - 1100.0
        )
        g[5] = (
            10.0
            * x[6] ** (-3)
            * np.sqrt(157.5e6 + (745.0 * x[4] / (x[1] * x[2])) ** 2)
            - 850.0
        )
        g[6] = x[1] * x[2] - 40.0
        g[7] = -x[0] / x[1] + 5.0
        g[8] = x[0] / x[1] - 12.0
        g[9] = 1.5 * x[5] - x[3] + 1.9
        g[10] = 1.1 * x[6] - x[4] + 1.9

        return g


class RC16(OptimizationProblem):
    """
    Optimal Design of Industrial Refrigeration System.

    Mechanical design problem with 14 variables and 15 inequality constraints, all
    of them written as normalized ratios that must not exceed one. The objective
    mixes integer and fractional powers (1.664, 1.8812, 0.3424, 2.893, 0.316),
    which makes it badly conditioned.

    Mathematical formulation:
        Minimize: a sum of eleven power-law cost terms in the 14 design variables;
        see the official implementation for the full expression.

        Subject to 15 inequality constraints of the form (ratio - 1) <= 0.

    Properties:
        - Dimension: 14
        - Number of inequality constraints: 15
        - Number of equality constraints: 0
        - More constraints than variables, with a highly nonlinear objective

    Reference:
        Kumar et al. (2020), problem RC16.

    Example:
        >>> problem = RC16()
        >>> problem.dim, problem.num_ineq_constraints
        (14, 15)
    """

    def __init__(self):
        super().__init__(
            name="RC16",
            dim=14,
            bounds=(np.full(14, 0.001), np.full(14, 5.0)),
            num_eq_constraints=0,
            num_ineq_constraints=15,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the total cost of the refrigeration system."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (
            63098.88 * x[1] * x[3] * x[11]
            + 5441.5 * x[1] ** 2 * x[11]
            + 115055.5 * x[1] ** 1.664 * x[5]
            + 6172.27 * x[1] ** 2 * x[5]
            + 63098.88 * x[0] * x[2] * x[10]
            + 5441.5 * x[0] ** 2 * x[10]
            + 115055.5 * x[0] ** 1.664 * x[4]
            + 6172.27 * x[0] ** 2 * x[4]
            + 140.53 * x[0] * x[10]
            + 281.29 * x[2] * x[10]
            + 70.26 * x[0] ** 2
            + 281.29 * x[0] * x[2]
            + 281.29 * x[2] ** 2
            + 14437.0
            * x[7] ** 1.8812
            * x[11] ** 0.3424
            * x[9]
            * x[13] ** (-1)
            * x[0] ** 2
            * x[6]
            * x[8] ** (-1)
            + 20470.2 * x[6] ** 2.893 * x[10] ** 0.316 * x[0] ** 2
        )

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the normalized design ratios (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(15)
        g[0] = 1.524 * x[6] ** (-1) - 1.0
        g[1] = 1.524 * x[7] ** (-1) - 1.0
        g[2] = 0.07789 * x[0] - 2.0 * x[6] ** (-1) * x[8] - 1.0
        g[3] = (
            7.05305
            * x[8] ** (-1)
            * x[0] ** 2
            * x[9]
            * x[7] ** (-1)
            * x[1] ** (-1)
            * x[13] ** (-1)
            - 1.0
        )
        g[4] = 0.0833 / x[12] * x[13] - 1.0
        g[5] = 0.04771 * x[9] * x[7] ** 1.8812 * x[11] ** 0.3424 - 1.0
        g[6] = 0.0488 * x[8] * x[6] ** 1.893 * x[10] ** 0.316 - 1.0
        g[7] = 0.0099 * x[0] / x[2] - 1.0
        g[8] = 0.0193 * x[1] / x[3] - 1.0
        g[9] = 0.0298 * x[0] / x[4] - 1.0
        g[10] = (
            47.136 * x[1] ** 0.333 / x[9] * x[11]
            - 1.333 * x[7] * x[12] ** 2.1195
            + 62.08 * x[12] ** 2.1195 * x[7] ** 0.2 / (x[11] * x[9])
            - 1.0
        )
        g[11] = 0.056 * x[1] / x[5] - 1.0
        g[12] = 2.0 / x[8] - 1.0
        g[13] = 2.0 / x[9] - 1.0
        g[14] = x[11] / x[10] - 1.0

        return g


class RC17(OptimizationProblem):
    """
    Tension/Compression Spring Design (case 1).

    The suite's formulation of the problem the paper calls MWTCS, with identical
    bounds.

    Note on the constraint count: the suite's parameter table ``Cal_par.m`` declares
    3 inequality constraints for this problem, but the official objective code
    ``cec20_func.m`` computes 4. The fourth is the outer-diameter limit
    ``(x1 + x2)/1.5 - 1 <= 0``, which appears in every literature formulation of
    this problem (and as g4 in the paper's own equation 38), so all 4 are
    implemented here.

    Mathematical formulation:
        Minimize: f(x) = x1^2 * x2 * (x3 + 2)

        Subject to:
            g1 = 1 - (x2^3 * x3)/(71785 * x1^4) <= 0
            g2 = (4*x2^2 - x1*x2)/(12566*(x2*x1^3 - x1^4)) + 1/(5108*x1^2) - 1 <= 0
            g3 = 1 - 140.45*x1/(x2^2 * x3) <= 0
            g4 = (x1 + x2)/1.5 - 1 <= 0

    Properties:
        - Dimension: 3
        - Number of inequality constraints: 4
        - Number of equality constraints: 0

    Reference:
        Kumar et al. (2020), problem RC17. Originally Arora (2004) and
        Belegundu & Arora (1985).

    Example:
        >>> problem = RC17()
        >>> problem.dim, problem.num_ineq_constraints
        (3, 4)
    """

    def __init__(self):
        super().__init__(
            name="RC17",
            dim=3,
            bounds=(
                np.array([0.05, 0.25, 2.0]),
                np.array([2.0, 1.3, 15.0]),
            ),
            num_eq_constraints=0,
            num_ineq_constraints=4,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the weight of the spring."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return x[0] ** 2 * x[1] * (x[2] + 2.0)

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate deflection, shear, surge and diameter limits (g(x) <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(4)
        g[0] = 1.0 - (x[1] ** 3 * x[2]) / (71785.0 * x[0] ** 4)
        g[1] = (
            (4.0 * x[1] ** 2 - x[0] * x[1])
            / (12566.0 * (x[1] * x[0] ** 3 - x[0] ** 4))
            + 1.0 / (5108.0 * x[0] ** 2)
            - 1.0
        )
        g[2] = 1.0 - 140.45 * x[0] / (x[1] ** 2 * x[2])
        g[3] = (x[0] + x[1]) / 1.5 - 1.0

        return g


class RC19(OptimizationProblem):
    """
    Welded Beam Design.

    The suite's formulation of the problem the paper calls WBD. It differs from the
    paper's WBD1 in the polar moment of inertia and the deflection expression, which
    is exactly the coefficient drift the revision plan warns about; both are
    implemented as their respective sources define them.

    Mathematical formulation:
        Minimize: f(x) = 1.10471*x1^2*x2 + 0.04811*x3*x4*(14 + x2)

        Subject to:
            g1 = tau(x) - tau_max <= 0
            g2 = sigma(x) - sigma_max <= 0
            g3 = x1 - x4 <= 0
            g4 = delta(x) - delta_max <= 0
            g5 = P - Pc(x) <= 0

        with P = 6000 lb, L = 14 in, E = 30e6 psi, G = 12e6 psi,
        tau_max = 13600 psi, sigma_max = 30000 psi, delta_max = 0.25 in.

    Properties:
        - Dimension: 4
        - Number of inequality constraints: 5
        - Number of equality constraints: 0

    Reference:
        Kumar et al. (2020), problem RC19.

    Example:
        >>> problem = RC19()
        >>> problem.dim, problem.num_ineq_constraints
        (4, 5)
    """

    P = 6000.0
    L = 14.0
    DELTA_MAX = 0.25
    E = 30e6
    G = 12e6
    TAU_MAX = 13600.0
    SIGMA_MAX = 30000.0

    def __init__(self):
        super().__init__(
            name="RC19",
            dim=4,
            bounds=(
                np.array([0.125, 0.1, 0.1, 0.1]),
                np.array([2.0, 10.0, 10.0, 2.0]),
            ),
            num_eq_constraints=0,
            num_ineq_constraints=5,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the fabrication cost of the welded beam."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return 1.10471 * x[0] ** 2 * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate stress, geometry, deflection and buckling limits (g <= 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        buckling_load = (
            4.013
            * self.E
            * np.sqrt(x[2] ** 2 * x[3] ** 6 / 30.0)
            / self.L**2
            * (1.0 - x[2] / (2.0 * self.L) * np.sqrt(self.E / (4.0 * self.G)))
        )
        normal_stress = 6.0 * self.P * self.L / (x[3] * x[2] ** 2)
        deflection = 6.0 * self.P * self.L**3 / (self.E * x[2] ** 2 * x[3])

        polar_moment = 2.0 * (
            np.sqrt(2.0) * x[0] * x[1] * (x[1] ** 2 / 4.0 + (x[0] + x[2]) ** 2 / 4.0)
        )
        radius = np.sqrt(x[1] ** 2 / 4.0 + (x[0] + x[2]) ** 2 / 4.0)
        moment = self.P * (self.L + x[1] / 2.0)

        secondary_shear = moment * radius / polar_moment
        primary_shear = self.P / (np.sqrt(2.0) * x[0] * x[1])
        shear_stress = np.sqrt(
            primary_shear**2
            + 2.0 * primary_shear * secondary_shear * x[1] / (2.0 * radius)
            + secondary_shear**2
        )

        g = np.zeros(5)
        g[0] = shear_stress - self.TAU_MAX
        g[1] = normal_stress - self.SIGMA_MAX
        g[2] = x[0] - x[3]
        g[3] = deflection - self.DELTA_MAX
        g[4] = self.P - buckling_load

        return g


def get_cec2020_rw_problems():
    """
    Return all implemented CEC2020 real-world problems.

    Returns:
        Dict mapping problem name to a freshly instantiated problem

    Example:
        >>> problems = get_cec2020_rw_problems()
        >>> sorted(problems.keys())[:3]
        ['RC01', 'RC02', 'RC03']
    """
    return {
        "RC01": RC01(),
        "RC02": RC02(),
        "RC03": RC03(),
        "RC04": RC04(),
        "RC05": RC05(),
        "RC06": RC06(),
        "RC08": RC08(),
        "RC14": RC14(),
        "RC15": RC15(),
        "RC16": RC16(),
        "RC17": RC17(),
        "RC19": RC19(),
    }


def get_cec2020_rw_by_dimension():
    """
    Return the implemented problems ordered by dimension.

    Useful for the scalability discussion, where results are read against problem
    size rather than against problem identity.

    Returns:
        List of (dimension, problem) tuples sorted by increasing dimension

    Example:
        >>> ordered = get_cec2020_rw_by_dimension()
        >>> ordered[0][0]
        2
        >>> ordered[-1][0]
        38
    """
    problems = get_cec2020_rw_problems().values()
    return sorted(((p.dim, p) for p in problems), key=lambda pair: pair[0])
