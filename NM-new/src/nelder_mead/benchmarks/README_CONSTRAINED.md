# Constrained Benchmark Problems

This module provides a comprehensive suite of constrained optimization benchmark problems for testing and validating constrained optimization algorithms.

## Overview

The constrained benchmark suite includes:
- **6 CEC 2006 problems** (G01, G04, G06, G07, G09, G10)
- **2 Engineering design problems** (Pressure Vessel, Welded Beam)

All problems inherit from the `OptimizationProblem` base class and provide:
- Objective function evaluation
- Equality constraint evaluation (if applicable)
- Inequality constraint evaluation
- Feasibility checking
- Constraint violation computation
- Known optimal values and locations (where available)

## CEC 2006 Benchmark Suite

### G01 - Quadratic with Linear Constraints
- **Dimension**: 13
- **Constraints**: 9 linear inequality constraints
- **Optimum**: f(x*) = -15.0
- **Characteristics**: Relatively simple, convex objective, linear constraints

### G04 - Nonlinear with Bilinear Constraints
- **Dimension**: 5
- **Constraints**: 6 nonlinear inequality constraints
- **Optimum**: f(x*) ≈ -30665.539
- **Characteristics**: Small feasible region, bilinear constraint terms

### G06 - Cubic with Nonlinear Constraints
- **Dimension**: 2
- **Constraints**: 2 nonlinear inequality constraints
- **Optimum**: f(x*) = -6961.81388
- **Characteristics**: Very small feasible region (~0.0066% of search space)

### G07 - Quadratic with Mixed Constraints
- **Dimension**: 10
- **Constraints**: 8 inequality constraints (linear and nonlinear)
- **Optimum**: f(x*) = 24.3062091
- **Characteristics**: Moderate complexity, mixed constraint types

### G09 - Polynomial with Nonlinear Constraints
- **Dimension**: 7
- **Constraints**: 4 nonlinear inequality constraints
- **Optimum**: f(x*) = 680.6300573
- **Characteristics**: High-degree polynomial terms

### G10 - Linear Objective with Nonlinear Constraints
- **Dimension**: 8
- **Constraints**: 6 nonlinear inequality constraints
- **Optimum**: f(x*) = 7049.248
- **Characteristics**: Linear objective, complex nonlinear constraints

## Engineering Design Problems

### Pressure Vessel Design
- **Dimension**: 4 (shell thickness, head thickness, inner radius, length)
- **Constraints**: 4 inequality constraints
- **Optimum**: f(x*) ≈ 6059.714
- **Objective**: Minimize total fabrication cost
- **Real-world application**: Cylindrical pressure vessel with hemispherical heads

### Welded Beam Design
- **Dimension**: 4 (weld thickness, clamped bar length, bar height, bar thickness)
- **Constraints**: 7 inequality constraints
- **Optimum**: f(x*) ≈ 1.7249
- **Objective**: Minimize fabrication cost
- **Real-world application**: Welded beam subject to stress and deflection constraints

## Usage Examples

### Basic Usage

```python
from nelder_mead.benchmarks.constrained import G06

# Create problem instance
problem = G06()

# Evaluate at a point
x = np.array([14.0, 1.0])
f = problem.objective(x)
g = problem.constraint_ineq(x)  # g(x) ≤ 0
h = problem.constraint_eq(x)    # h(x) = 0

# Check feasibility
is_feasible = problem.is_feasible(x)
violation = problem.compute_violation(x)
```

### Using with Optimization Algorithms

```python
from nelder_mead.benchmarks.constrained import PressureVessel
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.constraints.deb_barrier import DebBarrier

# Create problem
problem = PressureVessel()

# Create barrier method for constraint handling
barrier = DebBarrier(
    bounds=problem.bounds,
    num_solutions=problem.dim + 1,
    penalty_coeff=1e6
)

# Create and run algorithm
algorithm = NelderMead(
    objective_fn=problem.objective,
    constraint_eq_fn=problem.constraint_eq,
    constraint_ineq_fn=problem.constraint_ineq,
    lower_bounds=problem.bounds[0],
    upper_bounds=problem.bounds[1],
    max_fes=1000,
    seed=42,
    barrier=barrier
)

algorithm.run()
best_x, best_f, eq_vio, ineq_vio, total_vio = algorithm.get_best_solution()
```

### Getting All Problems

```python
from nelder_mead.benchmarks.constrained import (
    get_all_constrained_problems,
    get_cec_problems,
    get_engineering_problems
)

# Get all constrained problems
all_problems = get_all_constrained_problems()

# Get only CEC problems
cec_problems = get_cec_problems()

# Get only engineering problems
eng_problems = get_engineering_problems()

# Iterate over problems
for name, problem in all_problems.items():
    print(f"{name}: dim={problem.dim}, "
          f"ineq={problem.num_ineq_constraints}, "
          f"eq={problem.num_eq_constraints}")
```

## Problem Properties

All problems provide the following attributes:
- `name`: Problem name
- `dim`: Problem dimensionality
- `bounds`: Tuple of (lower_bounds, upper_bounds)
- `optimum_value`: Known optimal objective value (if available)
- `optimum_location`: Known optimal solution (if available)
- `num_eq_constraints`: Number of equality constraints
- `num_ineq_constraints`: Number of inequality constraints

## Constraint Conventions

- **Inequality constraints**: g(x) ≤ 0 (feasible when g(x) ≤ 0)
- **Equality constraints**: h(x) = 0 (feasible when |h(x)| ≈ 0)
- **Bound constraints**: lower ≤ x ≤ upper

## Testing

Comprehensive unit tests are provided in `tests/unit/test_constrained.py`:
- Problem initialization and metadata
- Objective function evaluation
- Constraint function evaluation
- Known optima verification
- Feasibility checking
- Constraint violation computation
- Edge cases and error handling

Integration tests in `tests/integration/test_constrained_optimization.py` verify compatibility with optimization algorithms.

## References

1. Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N.,
   Coello, C. C., & Deb, K. (2006). Problem definitions and evaluation criteria for
   the CEC 2006 special session on constrained real-parameter optimization.

2. Coello, C. A. C. (2000). Use of a self-adaptive penalty approach for engineering
   optimization problems. Computers in Industry, 41(2), 113-127.

## Notes

- Known optimal locations are approximate and may not be exactly feasible due to numerical precision
- Some problems have very small feasible regions, making them challenging for optimization algorithms
- Engineering problems use real-world physical constants and constraints
- All problems are suitable for testing constraint handling mechanisms in optimization algorithms
