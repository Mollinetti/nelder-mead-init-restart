# Core Mathematical Operations

This module contains pure mathematical functions for Nelder-Mead optimization algorithms.

## Modules

### simplex_operations.py

Pure functions for geometric simplex transformations. All operations are stateless and deterministic.

**Functions:**
- `compute_centroid()` - Compute centroid of simplex vertices (excluding worst by default)
- `reflect()` - Compute reflected point across centroid from worst point
- `expand()` - Compute expanded point beyond reflected point
- `contract_outside()` - Compute outside contraction between centroid and reflected point
- `contract_inside()` - Compute inside contraction between centroid and worst point
- `shrink()` - Shrink all simplex vertices toward best point
- `enforce_bounds()` - Clip point to box constraints

**Key Properties:**
- All operations return new arrays (no mutation)
- Formulas follow Nelder & Mead (1965) and Gao & Han (2012)
- Comprehensive unit tests verify geometric correctness

### stopping_criteria.py

Functions for detecting convergence conditions. Each returns a boolean indicating whether the criterion is met.

**Functions:**
- `check_oriented_length()` - Maximum edge length normalized by first vertex position
- `check_std_dev()` - Variance of objective function values across simplex
- `check_small_simplex()` - Simplex size relative to search space bounds
- `check_flat_simplex()` - All objective values nearly identical
- `check_degenerate_simplex()` - Loss of geometric structure (edge ratio or volume)
- `check_fminsearch_fun()` - MATLAB fminsearch-style function tolerance
- `check_fminsearch_x()` - MATLAB fminsearch-style position tolerance

**References:**
- Lagarias et al. (1998) - Convergence properties of Nelder-Mead

### restart_strategies.py

Strategies for reinitializing the simplex when convergence is detected.

**Classes:**
- `RestartStrategy` - Abstract base class
- `UniformRestart` - Generate points uniformly within bounds
- `GaussianRestart` - Generate points from N(0, I), clipped to bounds
- `GaussianBestRestart` - Generate points from N(best, I), clipped to bounds

**Factory Function:**
- `create_restart_strategy(name, seed)` - Create strategy by name

**Key Features:**
- All strategies support reproducibility via random seed
- Points are automatically enforced to be within bounds
- Gaussian-best strategy enables local exploration around current best

## Testing

All modules have comprehensive unit tests in `tests/unit/`:
- `test_simplex_operations.py` - 18 tests
- `test_stopping_criteria.py` - 22 tests
- `test_restart_strategies.py` - 21 tests

Total: **61 passing unit tests**

Run tests with:
```bash
python -m pytest tests/unit/ -v
```

## Usage Examples

### Simplex Operations

```python
import numpy as np
from nelder_mead.core.simplex_operations import (
    compute_centroid, reflect, expand, enforce_bounds
)

# Create a simplex (3 vertices in 2D)
simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

# Compute centroid (excluding worst point)
centroid = compute_centroid(simplex, exclude_worst=True)

# Reflect worst point across centroid
worst = simplex[-1]
reflected = reflect(centroid, worst, delta_r=1.0)

# Expand beyond reflection
expanded = expand(centroid, reflected, delta_e=2.0)

# Enforce bounds
bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
clipped = enforce_bounds(expanded, bounds[0], bounds[1])
```

### Stopping Criteria

```python
from nelder_mead.core.stopping_criteria import (
    check_oriented_length, check_std_dev, check_fminsearch_fun
)

# Check if simplex has converged
if check_oriented_length(simplex, epsilon=1e-4):
    print("Simplex is very small - converged!")

# Check if fitness values are nearly identical
fitness = np.array([1.0, 1.0001, 1.0002])
if check_std_dev(fitness, epsilon=1e-7):
    print("Fitness values are nearly identical - converged!")

# MATLAB fminsearch-style criterion
if check_fminsearch_fun(fitness, tol_fun=1e-12):
    print("Function tolerance met - converged!")
```

### Restart Strategies

```python
from nelder_mead.core.restart_strategies import create_restart_strategy

# Create restart strategy
strategy = create_restart_strategy("gaussian_best", seed=42)

# Generate new point for restart
bounds = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
best_solution = np.array([5.0, 5.0])
new_point = strategy.generate_point(2, bounds, best_solution=best_solution)

# Generate multiple points for full simplex restart
dim = 2
new_simplex = np.array([
    strategy.generate_point(dim, bounds, best_solution=best_solution)
    for _ in range(dim + 1)
])
```

## Design Principles

1. **Separation of Concerns**: Pure mathematical operations are separate from algorithm logic
2. **Testability**: Every function can be tested in isolation
3. **Immutability**: Operations return new arrays without mutating inputs
4. **Reproducibility**: Random operations support seeding for deterministic behavior
5. **Clear Interfaces**: Well-defined contracts with comprehensive docstrings

## Requirements Validated

This module implements and validates the following requirements from the specification:

- **Requirements 2.1-2.6**: Nelder-Mead simplex operations (reflection, expansion, contraction, shrink, centroid)
- **Requirements 4.1-4.5**: Stopping criteria (oriented length, std dev, small simplex, flat simplex, degenerate simplex)
- **Requirements 5.1-5.3**: Restart mechanisms (uniform, Gaussian, Gaussian-best)
- **Requirements 11.1-11.4**: Geometric correctness validation
- **Requirements 16.1, 16.4**: Bound enforcement
