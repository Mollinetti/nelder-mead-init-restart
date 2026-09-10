"""
Constrained Benchmark Problems

This module implements standard constrained optimization test problems
from the CEC benchmark suite and engineering design problems.

References:
    - Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N.,
      Coello, C. C., & Deb, K. (2006). Problem definitions and evaluation criteria for
      the CEC 2006 special session on constrained real-parameter optimization.
    - Coello, C. A. C. (2000). Use of a self-adaptive penalty approach for engineering
      optimization problems. Computers in Industry, 41(2), 113-127.
"""

import numpy as np
from nelder_mead.benchmarks.problem_suite import OptimizationProblem


class G01(OptimizationProblem):
    """
    G01 problem from CEC 2006 benchmark suite.

    A quadratic function with 9 linear inequality constraints.
    This is a relatively simple constrained problem with a convex objective
    and linear constraints.

    Mathematical formulation:
        Minimize: f(x) = 5*Σ(x_i) for i=1 to 4 - 5*Σ(x_i^2) for i=1 to 4 - Σ(x_i) for i=5 to 13

        Subject to:
            g1(x) = 2*x1 + 2*x2 + x10 + x11 - 10 ≤ 0
            g2(x) = 2*x1 + 2*x3 + x10 + x12 - 10 ≤ 0
            g3(x) = 2*x2 + 2*x3 + x11 + x12 - 10 ≤ 0
            g4(x) = -8*x1 + x10 ≤ 0
            g5(x) = -8*x2 + x11 ≤ 0
            g6(x) = -8*x3 + x12 ≤ 0
            g7(x) = -2*x4 - x5 + x10 ≤ 0
            g8(x) = -2*x6 - x7 + x11 ≤ 0
            g9(x) = -2*x8 - x9 + x12 ≤ 0

        Bounds: 0 ≤ x_i ≤ 1 for i=1 to 9, 0 ≤ x_i ≤ 100 for i=10 to 13

    Properties:
        - Dimension: 13
        - Number of inequality constraints: 9
        - Number of equality constraints: 0
        - Known optimum: f(x*) = -15.0

    Example:
        >>> problem = G01()
        >>> # Known optimum location (approximate)
        >>> x_opt = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1])
        >>> abs(problem.objective(x_opt) - (-15.0)) < 0.1
        True
    """

    def __init__(self):
        dim = 13
        lower_bounds = np.zeros(dim)
        upper_bounds = np.ones(dim)
        upper_bounds[9:13] = 100.0  # x10, x11, x12, x13 have upper bound 100

        # Known optimum (approximate)
        x_opt = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1])

        super().__init__(
            name="G01",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=-15.0,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=9,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G01 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        # f(x) = 5*Σ(x_i) for i=1 to 4 - 5*Σ(x_i^2) for i=1 to 4 - Σ(x_i) for i=5 to 13
        term1 = 5.0 * np.sum(x[0:4])
        term2 = -5.0 * np.sum(x[0:4] ** 2)
        term3 = -np.sum(x[4:13])

        return term1 + term2 + term3

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G01 inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(9)
        g[0] = 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10
        g[1] = 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10
        g[2] = 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10
        g[3] = -8 * x[0] + x[9]
        g[4] = -8 * x[1] + x[10]
        g[5] = -8 * x[2] + x[11]
        g[6] = -2 * x[3] - x[4] + x[9]
        g[7] = -2 * x[5] - x[6] + x[10]
        g[8] = -2 * x[7] - x[8] + x[11]

        return g


class G04(OptimizationProblem):
    """
    G04 problem from CEC 2006 benchmark suite.

    A nonlinear problem with 6 inequality constraints. This problem has a
    relatively small feasible region and tests the algorithm's ability to
    find and stay within the feasible space.

    Mathematical formulation:
        Minimize: f(x) = 5.3578547*x3^2 + 0.8356891*x1*x5 + 37.293239*x1 - 40792.141

        Subject to:
            g1(x) = 85.334407 + 0.0056858*x2*x5 + 0.0006262*x1*x4 - 0.0022053*x3*x5 - 92 ≤ 0
            g2(x) = -85.334407 - 0.0056858*x2*x5 - 0.0006262*x1*x4 + 0.0022053*x3*x5 ≤ 0
            g3(x) = 80.51249 + 0.0071317*x2*x5 + 0.0029955*x1*x2 + 0.0021813*x3^2 - 110 ≤ 0
            g4(x) = -80.51249 - 0.0071317*x2*x5 - 0.0029955*x1*x2 - 0.0021813*x3^2 + 90 ≤ 0
            g5(x) = 9.300961 + 0.0047026*x3*x5 + 0.0012547*x1*x3 + 0.0019085*x3*x4 - 25 ≤ 0
            g6(x) = -9.300961 - 0.0047026*x3*x5 - 0.0012547*x1*x3 - 0.0019085*x3*x4 + 20 ≤ 0

        Bounds: 78 ≤ x1 ≤ 102, 33 ≤ x2 ≤ 45, 27 ≤ x_i ≤ 45 for i=3,4,5

    Properties:
        - Dimension: 5
        - Number of inequality constraints: 6
        - Number of equality constraints: 0
        - Known optimum: f(x*) ≈ -30665.539

    Example:
        >>> problem = G04()
        >>> # Test that objective function works
        >>> x = np.array([78, 33, 27, 27, 27])
        >>> isinstance(problem.objective(x), float)
        True
    """

    def __init__(self):
        dim = 5
        lower_bounds = np.array([78.0, 33.0, 27.0, 27.0, 27.0])
        upper_bounds = np.array([102.0, 45.0, 45.0, 45.0, 45.0])

        # Known optimum (approximate)
        x_opt = np.array([78.0, 33.0, 29.995256025682, 45.0, 36.775812905788])

        super().__init__(
            name="G04",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=-30665.539,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=6,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G04 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (
            5.3578547 * x[2] ** 2
            + 0.8356891 * x[0] * x[4]
            + 37.293239 * x[0]
            - 40792.141
        )

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G04 inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(6)
        g[0] = (
            85.334407
            + 0.0056858 * x[1] * x[4]
            + 0.0006262 * x[0] * x[3]
            - 0.0022053 * x[2] * x[4]
            - 92
        )
        g[1] = (
            -85.334407
            - 0.0056858 * x[1] * x[4]
            - 0.0006262 * x[0] * x[3]
            + 0.0022053 * x[2] * x[4]
        )
        g[2] = (
            80.51249
            + 0.0071317 * x[1] * x[4]
            + 0.0029955 * x[0] * x[1]
            + 0.0021813 * x[2] ** 2
            - 110
        )
        g[3] = (
            -80.51249
            - 0.0071317 * x[1] * x[4]
            - 0.0029955 * x[0] * x[1]
            - 0.0021813 * x[2] ** 2
            + 90
        )
        g[4] = (
            9.300961
            + 0.0047026 * x[2] * x[4]
            + 0.0012547 * x[0] * x[2]
            + 0.0019085 * x[2] * x[3]
            - 25
        )
        g[5] = (
            -9.300961
            - 0.0047026 * x[2] * x[4]
            - 0.0012547 * x[0] * x[2]
            - 0.0019085 * x[2] * x[3]
            + 20
        )

        return g


class G06(OptimizationProblem):
    """
    G06 problem from CEC 2006 benchmark suite.

    A cubic problem with 2 nonlinear inequality constraints. This problem
    has a very small feasible region (approximately 0.0066% of the search space).

    Mathematical formulation:
        Minimize: f(x) = (x1 - 10)^3 + (x2 - 20)^3

        Subject to:
            g1(x) = -(x1 - 5)^2 - (x2 - 5)^2 + 100 ≤ 0
            g2(x) = (x1 - 6)^2 + (x2 - 5)^2 - 82.81 ≤ 0

        Bounds: 13 ≤ x1 ≤ 100, 0 ≤ x2 ≤ 100

    Properties:
        - Dimension: 2
        - Number of inequality constraints: 2
        - Number of equality constraints: 0
        - Known optimum: f(x*) = -6961.81388
        - Very small feasible region

    Example:
        >>> problem = G06()
        >>> # Known optimum location
        >>> x_opt = np.array([14.095, 0.84296])
        >>> abs(problem.objective(x_opt) - (-6961.81388)) < 1.0
        True
    """

    def __init__(self):
        dim = 2
        lower_bounds = np.array([13.0, 0.0])
        upper_bounds = np.array([100.0, 100.0])

        # Known optimum
        x_opt = np.array([14.095, 0.84296])

        super().__init__(
            name="G06",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=-6961.81388,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=2,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G06 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (x[0] - 10) ** 3 + (x[1] - 20) ** 3

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G06 inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(2)
        g[0] = -((x[0] - 5) ** 2) - (x[1] - 5) ** 2 + 100
        g[1] = (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81

        return g


class G07(OptimizationProblem):
    """
    G07 problem from CEC 2006 benchmark suite.

    A quadratic problem with 8 inequality constraints. This problem has
    a moderate number of constraints and tests the algorithm's ability
    to navigate a constrained space.

    Mathematical formulation:
        Minimize: f(x) = x1^2 + x2^2 + x1*x2 - 14*x1 - 16*x2 + (x3-10)^2 +
                         4*(x4-5)^2 + (x5-3)^2 + 2*(x6-1)^2 + 5*x7^2 +
                         7*(x8-11)^2 + 2*(x9-10)^2 + (x10-7)^2 + 45

        Subject to:
            g1(x) = -105 + 4*x1 + 5*x2 - 3*x7 + 9*x8 ≤ 0
            g2(x) = 10*x1 - 8*x2 - 17*x7 + 2*x8 ≤ 0
            g3(x) = -8*x1 + 2*x2 + 5*x9 - 2*x10 - 12 ≤ 0
            g4(x) = 3*(x1-2)^2 + 4*(x2-3)^2 + 2*x3^2 - 7*x4 - 120 ≤ 0
            g5(x) = 5*x1^2 + 8*x2 + (x3-6)^2 - 2*x4 - 40 ≤ 0
            g6(x) = x1^2 + 2*(x2-2)^2 - 2*x1*x2 + 14*x5 - 6*x6 ≤ 0
            g7(x) = 0.5*(x1-8)^2 + 2*(x2-4)^2 + 3*x5^2 - x6 - 30 ≤ 0
            g8(x) = -3*x1 + 6*x2 + 12*(x9-8)^2 - 7*x10 ≤ 0

        Bounds: -10 ≤ x_i ≤ 10 for all i

    Properties:
        - Dimension: 10
        - Number of inequality constraints: 8
        - Number of equality constraints: 0
        - Known optimum: f(x*) = 24.3062091

    Example:
        >>> problem = G07()
        >>> x = np.zeros(10)
        >>> isinstance(problem.objective(x), float)
        True
    """

    def __init__(self):
        dim = 10
        lower_bounds = np.full(dim, -10.0)
        upper_bounds = np.full(dim, 10.0)

        # Known optimum (approximate)
        x_opt = np.array(
            [
                2.171996,
                2.363683,
                8.773926,
                5.095984,
                0.9906548,
                1.430574,
                1.321644,
                9.828726,
                8.280092,
                8.375927,
            ]
        )

        super().__init__(
            name="G07",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=24.3062091,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=8,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G07 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (
            x[0] ** 2
            + x[1] ** 2
            + x[0] * x[1]
            - 14 * x[0]
            - 16 * x[1]
            + (x[2] - 10) ** 2
            + 4 * (x[3] - 5) ** 2
            + (x[4] - 3) ** 2
            + 2 * (x[5] - 1) ** 2
            + 5 * x[6] ** 2
            + 7 * (x[7] - 11) ** 2
            + 2 * (x[8] - 10) ** 2
            + (x[9] - 7) ** 2
            + 45
        )

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G07 inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(8)
        g[0] = -105 + 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7]
        g[1] = 10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7]
        g[2] = -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12
        g[3] = (
            3 * (x[0] - 2) ** 2 + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3] - 120
        )
        g[4] = 5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40
        g[5] = x[0] ** 2 + 2 * (x[1] - 2) ** 2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5]
        g[6] = 0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * x[4] ** 2 - x[5] - 30
        g[7] = -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9]

        return g


class G09(OptimizationProblem):
    """
    G09 problem from CEC 2006 benchmark suite.

    A polynomial problem with 4 inequality constraints. This problem has
    a nonlinear objective and nonlinear constraints.

    Mathematical formulation:
        Minimize: f(x) = (x1-10)^2 + 5*(x2-12)^2 + x3^4 + 3*(x4-11)^2 +
                         10*x5^6 + 7*x6^2 + x7^4 - 4*x6*x7 - 10*x6 - 8*x7

        Subject to:
            g1(x) = -127 + 2*x1^2 + 3*x2^4 + x3 + 4*x4^2 + 5*x5 ≤ 0
            g2(x) = -282 + 7*x1 + 3*x2 + 10*x3^2 + x4 - x5 ≤ 0
            g3(x) = -196 + 23*x1 + x2^2 + 6*x6^2 - 8*x7 ≤ 0
            g4(x) = 4*x1^2 + x2^2 - 3*x1*x2 + 2*x3^2 + 5*x6 - 11*x7 ≤ 0

        Bounds: -10 ≤ x_i ≤ 10 for all i

    Properties:
        - Dimension: 7
        - Number of inequality constraints: 4
        - Number of equality constraints: 0
        - Known optimum: f(x*) = 680.6300573

    Example:
        >>> problem = G09()
        >>> x = np.zeros(7)
        >>> isinstance(problem.objective(x), float)
        True
    """

    def __init__(self):
        dim = 7
        lower_bounds = np.full(dim, -10.0)
        upper_bounds = np.full(dim, 10.0)

        # Known optimum (approximate)
        x_opt = np.array(
            [2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227]
        )

        super().__init__(
            name="G09",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=680.6300573,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=4,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G09 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (
            (x[0] - 10) ** 2
            + 5 * (x[1] - 12) ** 2
            + x[2] ** 4
            + 3 * (x[3] - 11) ** 2
            + 10 * x[4] ** 6
            + 7 * x[5] ** 2
            + x[6] ** 4
            - 4 * x[5] * x[6]
            - 10 * x[5]
            - 8 * x[6]
        )

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G09 inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(4)
        g[0] = -127 + 2 * x[0] ** 2 + 3 * x[1] ** 4 + x[2] + 4 * x[3] ** 2 + 5 * x[4]
        g[1] = -282 + 7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4]
        g[2] = -196 + 23 * x[0] + x[1] ** 2 + 6 * x[5] ** 2 - 8 * x[6]
        g[3] = (
            4 * x[0] ** 2
            + x[1] ** 2
            - 3 * x[0] * x[1]
            + 2 * x[2] ** 2
            + 5 * x[5]
            - 11 * x[6]
        )

        return g


class G10(OptimizationProblem):
    """
    G10 problem from CEC 2006 benchmark suite.

    A linear objective with 6 nonlinear inequality constraints. This problem
    has both linear and nonlinear constraints.

    Mathematical formulation:
        Minimize: f(x) = x1 + x2 + x3

        Subject to:
            g1(x) = -1 + 0.0025*(x4 + x6) ≤ 0
            g2(x) = -1 + 0.0025*(x5 + x7 - x4) ≤ 0
            g3(x) = -1 + 0.01*(x8 - x5) ≤ 0
            g4(x) = -x1*x6 + 833.33252*x4 + 100*x1 - 83333.333 ≤ 0
            g5(x) = -x2*x7 + 1250*x5 + x2*x4 - 1250*x4 ≤ 0
            g6(x) = -x3*x8 + 1250000 + x3*x5 - 2500*x5 ≤ 0

        Bounds: 100 ≤ x1 ≤ 10000, 1000 ≤ x_i ≤ 10000 for i=2,3,
                10 ≤ x_i ≤ 1000 for i=4,5,6,7,8

    Properties:
        - Dimension: 8
        - Number of inequality constraints: 6
        - Number of equality constraints: 0
        - Known optimum: f(x*) = 7049.248

    Example:
        >>> problem = G10()
        >>> x = np.array([100, 1000, 1000, 10, 10, 10, 10, 10])
        >>> isinstance(problem.objective(x), float)
        True
    """

    def __init__(self):
        dim = 8
        lower_bounds = np.array([100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        upper_bounds = np.array(
            [10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        )

        # Known optimum (approximate)
        x_opt = np.array(
            [
                579.3167,
                1359.943,
                5110.071,
                182.0174,
                295.5985,
                217.9799,
                286.4162,
                395.5979,
            ]
        )

        super().__init__(
            name="G10",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=7049.248,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=6,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G10 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return x[0] + x[1] + x[2]

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G10 inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(6)
        g[0] = -1 + 0.0025 * (x[3] + x[5])
        g[1] = -1 + 0.0025 * (x[4] + x[6] - x[3])
        g[2] = -1 + 0.01 * (x[7] - x[4])
        g[3] = -x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333
        g[4] = -x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3]
        g[5] = -x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4]

        return g


class PressureVessel(OptimizationProblem):
    """
    Pressure Vessel Design Problem.

    A classic engineering design problem that minimizes the total cost of
    a cylindrical pressure vessel with hemispherical heads. The design
    variables are the shell thickness, head thickness, inner radius, and
    length of the cylindrical section.

    Mathematical formulation:
        Minimize: f(x) = 0.6224*x1*x3*x4 + 1.7781*x2*x3^2 + 3.1661*x1^2*x4 + 19.84*x1^2*x3

        Subject to:
            g1(x) = -x1 + 0.0193*x3 ≤ 0
            g2(x) = -x2 + 0.00954*x3 ≤ 0
            g3(x) = -π*x3^2*x4 - (4/3)*π*x3^3 + 1296000 ≤ 0
            g4(x) = x4 - 240 ≤ 0

        Bounds: 0 ≤ x1, x2 ≤ 99, 10 ≤ x3, x4 ≤ 200

        Note: In the original formulation, x1 and x2 are discrete (multiples of 0.0625),
              but we treat them as continuous for this implementation.

    Properties:
        - Dimension: 4
        - Number of inequality constraints: 4
        - Number of equality constraints: 0
        - Known optimum: f(x*) ≈ 6059.714
        - Real-world engineering problem

    Reference:
        Coello, C. A. C. (2000). Use of a self-adaptive penalty approach for
        engineering optimization problems. Computers in Industry, 41(2), 113-127.

    Example:
        >>> problem = PressureVessel()
        >>> # Known optimum (approximate)
        >>> x_opt = np.array([0.8125, 0.4375, 42.0984, 176.6366])
        >>> abs(problem.objective(x_opt) - 6059.714) < 10.0
        True
    """

    def __init__(self):
        dim = 4
        lower_bounds = np.array([0.0, 0.0, 10.0, 10.0])
        upper_bounds = np.array([99.0, 99.0, 200.0, 200.0])

        # Known optimum (approximate)
        x_opt = np.array([0.8125, 0.4375, 42.0984, 176.6366])

        super().__init__(
            name="PressureVessel",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=6059.714,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=4,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the pressure vessel cost function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        # Cost = material cost + forming cost + welding cost
        return (
            0.6224 * x[0] * x[2] * x[3]
            + 1.7781 * x[1] * x[2] ** 2
            + 3.1661 * x[0] ** 2 * x[3]
            + 19.84 * x[0] ** 2 * x[2]
        )

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the pressure vessel inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(4)
        g[0] = -x[0] + 0.0193 * x[2]
        g[1] = -x[1] + 0.00954 * x[2]
        g[2] = -np.pi * x[2] ** 2 * x[3] - (4.0 / 3.0) * np.pi * x[2] ** 3 + 1296000
        g[3] = x[3] - 240

        return g


class WeldedBeam(OptimizationProblem):
    """
    Welded Beam Design Problem.

    A classic engineering design problem that minimizes the fabrication cost
    of a welded beam subject to constraints on shear stress, bending stress,
    buckling load, end deflection, and side constraints.

    Design variables:
        x1 = weld thickness (h)
        x2 = length of clamped bar (l)
        x3 = height of the bar (t)
        x4 = thickness of the bar (b)

    Mathematical formulation:
        Minimize: f(x) = 1.10471*x1^2*x2 + 0.04811*x3*x4*(14.0 + x2)

        Subject to:
            g1(x) = τ(x) - τ_max ≤ 0
            g2(x) = σ(x) - σ_max ≤ 0
            g3(x) = x1 - x4 ≤ 0
            g4(x) = 0.10471*x1^2 + 0.04811*x3*x4*(14.0 + x2) - 5.0 ≤ 0
            g5(x) = 0.125 - x1 ≤ 0
            g6(x) = δ(x) - δ_max ≤ 0
            g7(x) = P - P_c(x) ≤ 0

        Where:
            τ(x) = sqrt(τ'^2 + 2*τ'*τ''*x2/(2*R) + τ''^2)
            τ' = P/(sqrt(2)*x1*x2)
            τ'' = M*R/J
            M = P*(L + x2/2)
            R = sqrt(x2^2/4 + ((x1+x3)/2)^2)
            J = 2*(sqrt(2)*x1*x2*(x2^2/12 + ((x1+x3)/2)^2))
            σ(x) = 6*P*L/(x4*x3^2)
            δ(x) = 4*P*L^3/(E*x3^3*x4)
            P_c(x) = 4.013*E*sqrt(x3^2*x4^6/36)/L^2 * (1 - x3/(2*L)*sqrt(E/(4*G)))

        Constants:
            P = 6000 lb (applied load)
            L = 14 in (beam length)
            E = 30e6 psi (Young's modulus)
            G = 12e6 psi (shear modulus)
            τ_max = 13600 psi (maximum shear stress)
            σ_max = 30000 psi (maximum normal stress)
            δ_max = 0.25 in (maximum deflection)

        Bounds: 0.1 ≤ x1, x4 ≤ 2.0, 0.1 ≤ x2, x3 ≤ 10.0

    Properties:
        - Dimension: 4
        - Number of inequality constraints: 7
        - Number of equality constraints: 0
        - Known optimum: f(x*) ≈ 1.7249
        - Real-world engineering problem

    Reference:
        Coello, C. A. C. (2000). Use of a self-adaptive penalty approach for
        engineering optimization problems. Computers in Industry, 41(2), 113-127.

    Example:
        >>> problem = WeldedBeam()
        >>> # Known optimum (approximate)
        >>> x_opt = np.array([0.2057, 3.4705, 9.0366, 0.2057])
        >>> abs(problem.objective(x_opt) - 1.7249) < 0.01
        True
    """

    def __init__(self):
        dim = 4
        lower_bounds = np.array([0.1, 0.1, 0.1, 0.1])
        upper_bounds = np.array([2.0, 10.0, 10.0, 2.0])

        # Known optimum (approximate)
        x_opt = np.array([0.2057, 3.4705, 9.0366, 0.2057])

        super().__init__(
            name="WeldedBeam",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=1.7249,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=7,
        )

        # Problem constants
        self.P = 6000.0  # Applied load (lb)
        self.L = 14.0  # Beam length (in)
        self.E = 30e6  # Young's modulus (psi)
        self.G = 12e6  # Shear modulus (psi)
        self.tau_max = 13600.0  # Maximum shear stress (psi)
        self.sigma_max = 30000.0  # Maximum normal stress (psi)
        self.delta_max = 0.25  # Maximum deflection (in)

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the welded beam cost function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        # Cost = welding cost + material cost
        return 1.10471 * x[0] ** 2 * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the welded beam inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        # Compute intermediate values
        tau_prime = self.P / (np.sqrt(2) * x[0] * x[1])
        M = self.P * (self.L + x[1] / 2.0)
        R = np.sqrt(x[1] ** 2 / 4.0 + ((x[0] + x[2]) / 2.0) ** 2)
        J = 2 * (
            np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 12.0 + ((x[0] + x[2]) / 2.0) ** 2)
        )
        tau_double_prime = M * R / J
        tau = np.sqrt(
            tau_prime**2
            + 2 * tau_prime * tau_double_prime * x[1] / (2 * R)
            + tau_double_prime**2
        )

        sigma = 6 * self.P * self.L / (x[3] * x[2] ** 2)
        delta = 4 * self.P * self.L**3 / (self.E * x[2] ** 3 * x[3])

        # Buckling load
        P_c = (
            4.013
            * self.E
            * np.sqrt(x[2] ** 2 * x[3] ** 6 / 36.0)
            / self.L**2
            * (1 - x[2] / (2 * self.L) * np.sqrt(self.E / (4 * self.G)))
        )

        g = np.zeros(7)
        g[0] = tau - self.tau_max
        g[1] = sigma - self.sigma_max
        g[2] = x[0] - x[3]
        g[3] = 0.10471 * x[0] ** 2 + 0.04811 * x[2] * x[3] * (14.0 + x[1]) - 5.0
        g[4] = 0.125 - x[0]
        g[5] = delta - self.delta_max
        g[6] = self.P - P_c

        return g


class G05(OptimizationProblem):
    """
    G05 problem from the CEC 2006 benchmark suite.

    A cubic objective with two linear inequalities and three nonlinear equality
    constraints. Included for equality-constraint coverage: the g-series problems
    already in this module are inequality-only, and equality constraints confine
    the search to a measure-zero manifold, which is a materially different
    difficulty for a direct-search method.

    Mathematical formulation:
        Minimize: f(x) = 3*x1 + 1e-6*x1^3 + 2*x2 + (2e-6/3)*x2^3

        Subject to:
            g1(x) = x4 - x3 - 0.55 ≤ 0
            g2(x) = x3 - x4 - 0.55 ≤ 0
            h1(x) = 1000*sin(-x3 - 0.25) + 1000*sin(-x4 - 0.25) + 894.8 - x1 = 0
            h2(x) = 1000*sin(x3 - 0.25) + 1000*sin(x3 - x4 - 0.25) + 894.8 - x2 = 0
            h3(x) = 1000*sin(x4 - 0.25) + 1000*sin(x4 - x3 - 0.25) + 1294.8 = 0

        Bounds: 0 ≤ x1, x2 ≤ 1200, -0.55 ≤ x3, x4 ≤ 0.55

    Properties:
        - Dimension: 4
        - Number of inequality constraints: 2
        - Number of equality constraints: 3
        - Known optimum: f(x*) = 5126.4967140071
        - Feasible region is a measure-zero manifold

    Reference:
        Liang, J. J., et al. (2006). Problem definitions and evaluation criteria
        for the CEC 2006 special session on constrained real-parameter
        optimization. Technical Report, Nanyang Technological University.

    Example:
        >>> problem = G05()
        >>> x_opt = np.array([679.9453, 1026.067, 0.1188764, -0.3962336])
        >>> abs(problem.objective(x_opt) - 5126.4967140071) < 1e-2
        True
    """

    def __init__(self):
        dim = 4
        lower_bounds = np.array([0.0, 0.0, -0.55, -0.55])
        upper_bounds = np.array([1200.0, 1200.0, 0.55, 0.55])

        # Known optimum
        x_opt = np.array([679.9453, 1026.067, 0.1188764, -0.3962336])

        super().__init__(
            name="G05",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=5126.4967140071,
            optimum_location=x_opt,
            num_eq_constraints=3,
            num_ineq_constraints=2,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G05 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return (
            3.0 * x[0]
            + 1e-6 * x[0] ** 3
            + 2.0 * x[1]
            + (2e-6 / 3.0) * x[1] ** 3
        )

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G05 equality constraints (h(x) = 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        h = np.zeros(3)
        h[0] = (
            1000.0 * np.sin(-x[2] - 0.25)
            + 1000.0 * np.sin(-x[3] - 0.25)
            + 894.8
            - x[0]
        )
        h[1] = (
            1000.0 * np.sin(x[2] - 0.25)
            + 1000.0 * np.sin(x[2] - x[3] - 0.25)
            + 894.8
            - x[1]
        )
        h[2] = (
            1000.0 * np.sin(x[3] - 0.25)
            + 1000.0 * np.sin(x[3] - x[2] - 0.25)
            + 1294.8
        )

        return h

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G05 inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(2)
        g[0] = x[3] - x[2] - 0.55
        g[1] = x[2] - x[3] - 0.55

        return g


class G08(OptimizationProblem):
    """
    G08 problem from the CEC 2006 benchmark suite.

    A small, strongly multimodal problem. The objective is a ratio of products of
    sines, so the landscape is covered in local optima while the feasible region
    is large. It separates methods that explore from methods that merely descend,
    which is exactly the distinction the paper's restart strategy addresses.

    Mathematical formulation:
        Minimize: f(x) = -sin^3(2*pi*x1) * sin(2*pi*x2) / (x1^3 * (x1 + x2))

        Subject to:
            g1(x) = x1^2 - x2 + 1 ≤ 0
            g2(x) = 1 - x1 + (x2 - 4)^2 ≤ 0

        Bounds: 0 ≤ x1, x2 ≤ 10

    Properties:
        - Dimension: 2
        - Number of inequality constraints: 2
        - Number of equality constraints: 0
        - Known optimum: f(x*) = -0.0958250414
        - Highly multimodal with a large feasible region

    Reference:
        Liang, J. J., et al. (2006). Problem definitions and evaluation criteria
        for the CEC 2006 special session on constrained real-parameter
        optimization. Technical Report, Nanyang Technological University.

    Example:
        >>> problem = G08()
        >>> x_opt = np.array([1.2279713, 4.2453733])
        >>> abs(problem.objective(x_opt) - (-0.0958250414)) < 1e-6
        True
    """

    def __init__(self):
        dim = 2
        lower_bounds = np.array([0.0, 0.0])
        upper_bounds = np.array([10.0, 10.0])

        # Known optimum
        x_opt = np.array([1.2279713, 4.2453733])

        super().__init__(
            name="G08",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=-0.0958250414,
            optimum_location=x_opt,
            num_eq_constraints=0,
            num_ineq_constraints=2,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G08 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        numerator = np.sin(2.0 * np.pi * x[0]) ** 3 * np.sin(2.0 * np.pi * x[1])
        denominator = x[0] ** 3 * (x[0] + x[1])

        return -numerator / denominator

    def constraint_ineq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G08 inequality constraints (g(x) ≤ 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        g = np.zeros(2)
        g[0] = x[0] ** 2 - x[1] + 1.0
        g[1] = 1.0 - x[0] + (x[1] - 4.0) ** 2

        return g


class G11(OptimizationProblem):
    """
    G11 problem from the CEC 2006 benchmark suite.

    The smallest equality-constrained problem in the suite: a quadratic objective
    restricted to a parabola. Its value here is diagnostic rather than
    competitive. Because the feasible set is a one-dimensional curve in a
    two-dimensional box, any method that cannot reach and stay on an equality
    manifold fails visibly, and it does so in a problem small enough to inspect.

    Mathematical formulation:
        Minimize: f(x) = x1^2 + (x2 - 1)^2

        Subject to:
            h1(x) = x2 - x1^2 = 0

        Bounds: -1 ≤ x1, x2 ≤ 1

    Properties:
        - Dimension: 2
        - Number of inequality constraints: 0
        - Number of equality constraints: 1
        - Known optimum: f(x*) = 0.7499
        - Two symmetric global optima at x1 = ±1/sqrt(2)

    Reference:
        Liang, J. J., et al. (2006). Problem definitions and evaluation criteria
        for the CEC 2006 special session on constrained real-parameter
        optimization. Technical Report, Nanyang Technological University.

    Example:
        >>> problem = G11()
        >>> x_opt = np.array([-0.7071068, 0.5])
        >>> abs(problem.objective(x_opt) - 0.7499) < 1e-3
        True
    """

    def __init__(self):
        dim = 2
        lower_bounds = np.array([-1.0, -1.0])
        upper_bounds = np.array([1.0, 1.0])

        # Known optimum (one of the two symmetric solutions)
        x_opt = np.array([-0.7071068, 0.5])

        super().__init__(
            name="G11",
            dim=dim,
            bounds=(lower_bounds, upper_bounds),
            optimum_value=0.7499,
            optimum_location=x_opt,
            num_eq_constraints=1,
            num_ineq_constraints=0,
        )

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the G11 objective function."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return x[0] ** 2 + (x[1] - 1.0) ** 2

    def constraint_eq(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the G11 equality constraint (h(x) = 0)."""
        if len(x) != self.dim:
            raise ValueError(f"Expected {self.dim} dimensions, got {len(x)}")

        return np.array([x[1] - x[0] ** 2])


def get_all_constrained_problems():
    """
    Get a dictionary of all constrained benchmark problems.

    Returns:
        dict: Dictionary mapping problem names to problem instances

    Example:
        >>> problems = get_all_constrained_problems()
        >>> 'G01' in problems
        True
        >>> isinstance(problems['G01'], OptimizationProblem)
        True
    """
    return {
        "G01": G01(),
        "G04": G04(),
        "G05": G05(),
        "G06": G06(),
        "G07": G07(),
        "G08": G08(),
        "G09": G09(),
        "G10": G10(),
        "G11": G11(),
        "PressureVessel": PressureVessel(),
        "WeldedBeam": WeldedBeam(),
    }


def get_cec_problems():
    """
    Get a dictionary of CEC benchmark problems only.

    Returns:
        dict: Dictionary mapping CEC problem names to problem instances
    """
    return {
        "G01": G01(),
        "G04": G04(),
        "G05": G05(),
        "G06": G06(),
        "G07": G07(),
        "G08": G08(),
        "G09": G09(),
        "G10": G10(),
        "G11": G11(),
    }


def get_engineering_problems():
    """
    Get a dictionary of engineering design problems only.

    Returns:
        dict: Dictionary mapping engineering problem names to problem instances
    """
    return {"PressureVessel": PressureVessel(), "WeldedBeam": WeldedBeam()}
