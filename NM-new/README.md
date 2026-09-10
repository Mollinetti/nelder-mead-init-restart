# Nelder-Mead Optimization Library

A comprehensive, modular implementation of Nelder-Mead optimization algorithms with constraint handling for academic research and practical applications.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This library provides a clean, well-tested implementation of the Nelder-Mead simplex algorithm and its adaptive variant, along with multiple constraint handling methods. It was developed as part of doctoral research on optimization algorithms and is designed for:

- **Academic Research**: Reproducible experiments with comprehensive benchmarking
- **Algorithm Development**: Modular architecture for easy extension
- **Practical Applications**: Robust constraint handling for real-world problems

### Key Features

- ✅ **Classic Nelder-Mead Algorithm** with configurable parameters
- ✅ **Adaptive Nelder-Mead** with dimension-dependent coefficients (Gao & Han, 2012)
- ✅ **Multiple Constraint Handling Methods**:
  - Hard Barrier (infinite penalty)
  - Deb's Constraint Handling
  - Augmented Lagrangian Method
  - Progressive Barrier Method
- ✅ **Comprehensive Benchmark Suite**:
  - Unconstrained problems (Sphere, Rosenbrock, Rastrigin, Ackley, etc.)
  - Constrained problems (CEC 2006 suite, engineering design problems)
- ✅ **Batch Testing Infrastructure** for algorithm comparison
- ✅ **Statistical Analysis Tools** for publication-ready results
- ✅ **Reproducible Results** with seed control
- ✅ **Extensive Documentation** and examples

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Benchmark Testing](#benchmark-testing)
- [Extending the Library](#extending-the-library)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Installation

### Requirements

- Python 3.8 or higher
- NumPy >= 1.20.0
- (Optional) Matplotlib >= 3.3.0 for plotting
- (Optional) Hypothesis >= 6.0.0 for property-based testing

### Install from Source

```bash
# Clone the repository
git clone https://github.com/mollinetti/nelder-mead-optimization.git
cd nelder-mead-optimization

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "from nelder_mead.algorithms import NelderMead; print('Installation successful!')"
```

## Quick Start

Here's a minimal example to get you started:

```python
import numpy as np
from nelder_mead.algorithms.nelder_mead import NelderMead

# Define your objective function
def sphere(x):
    return np.sum(x**2)

# Create the optimizer
optimizer = NelderMead(
    objective_fn=sphere,
    lower_bounds=np.array([-5.0, -5.0]),
    upper_bounds=np.array([5.0, 5.0]),
    max_fes=1000,
    seed=42
)

# Run optimization
optimizer.run()

# Get results
best_x, best_f, _, _, _ = optimizer.get_best_solution()
print(f"Best solution: {best_x}")
print(f"Best fitness: {best_f}")
```

## Basic Usage

### Unconstrained Optimization

```python
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.benchmarks.unconstrained import Rosenbrock

# Use a benchmark problem
problem = Rosenbrock(dim=5)

# Create optimizer
optimizer = NelderMead(
    objective_fn=problem.objective,
    lower_bounds=problem.bounds[0],
    upper_bounds=problem.bounds[1],
    max_fes=5000,
    seed=42
)

# Run and get results
optimizer.run()
best_x, best_f, _, _, _ = optimizer.get_best_solution()
```

### Constrained Optimization

```python
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.constraints import AugmentedLagrangian
import numpy as np

# Define problem
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def constraint_ineq(x):
    return np.array([x[0] + x[1] - 2])  # x + y <= 2

# Create barrier method
barrier = AugmentedLagrangian(
    bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
    num_solutions=3,
    m_eq=0,
    p_ineq=1
)

# Create optimizer with barrier
optimizer = NelderMead(
    objective_fn=objective,
    constraint_ineq_fn=constraint_ineq,
    lower_bounds=np.array([-5.0, -5.0]),
    upper_bounds=np.array([5.0, 5.0]),
    max_fes=2000,
    seed=42,
    barrier=barrier
)

optimizer.run()
best_x, best_f, _, _, total_vio = optimizer.get_best_solution()
print(f"Solution: {best_x}, Fitness: {best_f}, Violation: {total_vio}")
```

### Using Adaptive Nelder-Mead

```python
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead

# Adaptive NM automatically adjusts parameters based on dimension
optimizer = AdaptiveNelderMead(
    objective_fn=objective,
    lower_bounds=lower,
    upper_bounds=upper,
    max_fes=5000,
    seed=42
)

optimizer.run()
```

## Advanced Features

### Custom Algorithm Parameters

```python
optimizer = NelderMead(
    objective_fn=objective,
    lower_bounds=lower,
    upper_bounds=upper,
    max_fes=5000,
    # Custom Nelder-Mead coefficients
    delta_r=1.0,    # Reflection
    delta_e=2.0,    # Expansion
    delta_oc=0.5,   # Outside contraction
    delta_ic=-0.5,  # Inside contraction
    gamma_s=0.5,    # Shrink
    # Initialization method
    init_method="spendleySimplex",  # or "uniform", "gaussian", "pfeffer", "adaptive"
    # Restart strategy
    restart_strategy="gaussian_best",  # or "uniform", "gaussian"
    seed=42
)
```

### Available Constraint Handling Methods

```python
from nelder_mead.constraints import (
    HardBarrier,           # Infinite penalty for violations
    DebBarrier,            # Deb's constraint handling
    AugmentedLagrangian,   # Lagrangian multiplier method
    ProgressiveBarrier     # Adaptive threshold method
)

# Hard Barrier - simplest but can be unstable
barrier = HardBarrier(bounds=(lower, upper), num_solutions=dim+1)

# Deb Barrier - penalty based on worst fitness
barrier = DebBarrier(bounds=(lower, upper), num_solutions=dim+1)

# Augmented Lagrangian - adaptive multipliers (recommended)
barrier = AugmentedLagrangian(
    bounds=(lower, upper),
    num_solutions=dim+1,
    m_eq=num_equality_constraints,
    p_ineq=num_inequality_constraints,
    rho=2.0,      # Penalty parameter
    gamma=1.2,    # Penalty increase factor
    tau=0.95      # Violation tolerance
)

# Progressive Barrier - adaptive threshold
barrier = ProgressiveBarrier(
    bounds=(lower, upper),
    penalty_coeff=1e4
)
```

## Benchmark Testing

### Running Single Experiments

```python
from nelder_mead.testing.batch_runner import BatchRunner
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.benchmarks.unconstrained import Sphere

runner = BatchRunner(verbose=True)
problem = Sphere(dim=10)

# Run algorithm 10 times with different seeds
results = runner.run_experiment(
    algorithm_class=NelderMead,
    problem=problem,
    num_runs=10,
    max_fes=5000
)

# Analyze results
fitness_values = [r.best_fitness for r in results]
print(f"Mean fitness: {np.mean(fitness_values):.6e}")
print(f"Std deviation: {np.std(fitness_values):.6e}")
```

### Comparing Multiple Algorithms

```python
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
from nelder_mead.benchmarks.unconstrained import Rosenbrock, Rastrigin

# Configure algorithms
algorithm_configs = [
    {'class': NelderMead, 'name': 'Classic-NM', 'max_fes': 5000},
    {'class': AdaptiveNelderMead, 'name': 'Adaptive-NM', 'max_fes': 5000}
]

# Configure problems
problems = [
    Sphere(dim=10),
    Rosenbrock(dim=10),
    Rastrigin(dim=10)
]

# Run comparison
results = runner.compare_algorithms(
    algorithm_configs=algorithm_configs,
    problems=problems,
    num_runs=10,
    seeds=range(1, 11)
)
```

### Statistical Analysis and Visualization

```python
from nelder_mead.testing.result_analyzer import ResultAnalyzer

analyzer = ResultAnalyzer()

# Compute statistics
stats = analyzer.compute_statistics(results)
print(f"Mean: {stats['mean']:.6e}")
print(f"Best: {stats['best']:.6e}")
print(f"Success rate: {stats['success_rate']*100:.1f}%")

# Generate convergence plots
analyzer.generate_convergence_plots(
    results_dict={'NM': nm_results, 'ANM': anm_results},
    problem_name='Sphere (10D)',
    output_path='convergence.png',
    log_scale=True
)

# Generate comparison table (LaTeX format for papers)
latex_table = analyzer.generate_comparison_table(
    results_dict=results,
    problem_names=['Sphere', 'Rosenbrock', 'Rastrigin'],
    output_path='comparison.tex',
    format='latex',
    include_std=True,
    scientific_notation=True
)
```

## Extending the Library

### Adding a New Algorithm

Create a custom algorithm by extending `NelderMead` or `BaseAlgorithm`:

```python
from nelder_mead.algorithms.nelder_mead import NelderMead

class MyCustomNM(NelderMead):
    """Custom Nelder-Mead variant with modified coefficients."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override coefficients
        self.delta_e = 1.5  # Custom expansion
        self.gamma_s = 0.7  # Custom shrink
    
    # Optionally override run() for completely custom logic
    def run(self):
        # Your custom optimization loop
        pass
```

See `examples/custom_algorithm.py` for a complete example.

### Adding a New Benchmark Problem

Create a custom problem by extending `OptimizationProblem`:

```python
from nelder_mead.benchmarks.problem_suite import OptimizationProblem
import numpy as np

class MyProblem(OptimizationProblem):
    """Custom optimization problem."""
    
    def __init__(self, dim=5):
        bounds = (np.full(dim, -10.0), np.full(dim, 10.0))
        super().__init__(
            name="MyProblem",
            dim=dim,
            bounds=bounds,
            optimum_value=0.0,
            optimum_location=np.zeros(dim)
        )
    
    def objective(self, x):
        """Implement your objective function."""
        return np.sum(x**2)
    
    def constraint_ineq(self, x):
        """Optional: inequality constraints g(x) <= 0."""
        return np.array([])
    
    def constraint_eq(self, x):
        """Optional: equality constraints h(x) = 0."""
        return np.array([])
```

See `examples/custom_problem.py` for a complete example.

### Adding a New Constraint Handling Method

Create a custom barrier by extending `Barrier`:

```python
from nelder_mead.constraints.barrier_base import Barrier

class MyBarrier(Barrier):
    """Custom constraint handling method."""
    
    def __init__(self, bounds, num_solutions, **kwargs):
        super().__init__(bounds, num_solutions)
        # Initialize your parameters
    
    def penalize_constraint_violation(self, solution, eq_violations, ineq_violations):
        """
        Compute penalty for constraint violations.
        
        Args:
            solution: Current solution
            eq_violations: Equality constraint violations
            ineq_violations: Inequality constraint violations
        
        Returns:
            Penalty value (float)
        """
        # Your penalty computation
        return penalty
    
    def update(self, *args, **kwargs):
        """Optional: Update barrier parameters adaptively."""
        pass
```

## API Reference

### Core Modules

#### `nelder_mead.algorithms`

- **`NelderMead`**: Classic Nelder-Mead algorithm
  - `__init__(objective_fn, lower_bounds, upper_bounds, max_fes, ...)`
  - `run()`: Execute optimization
  - `get_best_solution()`: Returns `(best_x, best_f, eq_vio, ineq_vio, total_vio)`

- **`AdaptiveNelderMead`**: Adaptive variant with dimension-dependent parameters
  - Same interface as `NelderMead`

#### `nelder_mead.constraints`

- **`HardBarrier`**: Infinite penalty for violations
- **`DebBarrier`**: Deb's constraint handling
- **`AugmentedLagrangian`**: Lagrangian multiplier method
- **`ProgressiveBarrier`**: Adaptive threshold method

#### `nelder_mead.benchmarks`

- **`unconstrained`**: Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Schwefel
- **`constrained`**: G01-G10 (CEC 2006), PressureVessel, WeldedBeam

#### `nelder_mead.testing`

- **`BatchRunner`**: Run batch experiments
  - `run_experiment(algorithm_class, problem, num_runs, ...)`
  - `compare_algorithms(algorithm_configs, problems, ...)`

- **`ResultAnalyzer`**: Analyze and visualize results
  - `compute_statistics(results)`
  - `generate_convergence_plots(...)`
  - `generate_comparison_table(...)`

### Initialization Methods

- `"uniform"`: Random uniform within bounds
- `"gaussian"`: Gaussian distribution
- `"spendleySimplex"`: Regular simplex (Spendley et al.)
- `"pfeffer"`: Pfeffer's simplex (Lagarias 1995)
- `"adaptiveSimplex"`: Adaptive simplex (Gao & Han 2012)

### Restart Strategies

- `"uniform"`: Uniform random restart
- `"gaussian"`: Gaussian restart centered at origin
- `"gaussian_best"`: Gaussian restart centered at best solution (default)

## Examples

The `examples/` directory contains comprehensive demonstrations:

- **`basic_usage.py`**: Simple getting started example
- **`adaptive_nelder_mead_demo.py`**: Comparing classic vs adaptive NM
- **`constrained_optimization.py`**: Using different barrier methods
- **`custom_algorithm.py`**: Creating custom algorithm variants
- **`custom_problem.py`**: Adding new benchmark problems
- **`batch_runner_example.py`**: Batch testing and comparison
- **`result_analysis_demo.py`**: Statistical analysis and visualization
- **`benchmark_unconstrained.py`**: Testing on unconstrained problems
- **`benchmark_constrained.py`**: Testing on constrained problems

Run any example:

```bash
python examples/basic_usage.py
python examples/adaptive_nelder_mead_demo.py
```

## Testing

The library includes comprehensive unit tests, property-based tests, and integration tests.

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=nelder_mead --cov-report=html

# Run specific test modules
python -m pytest tests/unit/test_nelder_mead.py
python -m pytest tests/property/test_simplex_properties.py
python -m pytest tests/integration/test_batch_runner_integration.py
```

### Test Statistics

- **664 total tests**: 544 unit + 106 property-based + 14 integration
- **98.8% pass rate** (656 passed, 8 skipped matplotlib tests)
- **89% code coverage** (94-100% on core modules)
- **~18 seconds** execution time

### Test Structure

```
tests/
├── unit/                    # Unit tests for individual components (544 tests)
│   ├── test_simplex_operations.py
│   ├── test_stopping_criteria.py
│   ├── test_nelder_mead.py
│   └── ...
├── property/                # Property-based tests (106 tests)
│   ├── test_simplex_properties.py
│   ├── test_constraint_properties.py
│   ├── test_algorithm_properties.py
│   └── test_reproducibility.py
└── integration/             # Integration tests (14 tests)
    ├── test_batch_runner_integration.py
    └── test_result_analyzer_integration.py
```

### Property-Based Testing

The library includes 106 property-based tests that validate universal correctness properties using the Hypothesis framework. These tests verify that algorithms behave correctly across a wide range of randomly generated inputs, providing strong correctness guarantees.

Example properties tested:
- Simplex operations produce geometrically correct transformations
- All initialization methods respect problem bounds
- Stopping criteria use correct mathematical formulas
- Constraint handling methods apply correct penalty formulas
- Algorithms produce reproducible results with fixed seeds

## Project Structure

```
nelder-mead-optimization/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── setup.py                       # Package installation
├── pyproject.toml                 # Modern Python packaging
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Development dependencies
│
├── src/                           # Source code
│   └── nelder_mead/              # Main package
│       ├── algorithms/           # Algorithm implementations
│       ├── constraints/          # Constraint handling methods
│       ├── core/                 # Core mathematical operations
│       ├── initialization/       # Simplex initialization
│       ├── benchmarks/           # Benchmark problems
│       └── testing/              # Testing infrastructure
│
├── tests/                         # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
│
├── examples/                      # Usage examples
│   ├── README.md                 # Examples guide
│   ├── basic_usage.py
│   ├── constrained_optimization.py
│   └── ...
│
├── docs/                          # Documentation
│   ├── README.md                 # Documentation index
│   ├── installation.md           # Installation guide
│   ├── quickstart.md             # Quick start tutorial
│   ├── api_reference.md          # API documentation
│   ├── tutorials/                # Detailed tutorials
│   ├── validation/               # Validation reports
│   └── paper_references.md       # Academic citations
│
├── benchmarks/                    # Benchmark scripts & results
│   ├── README.md                 # Benchmarking guide
│   ├── scripts/                  # Benchmark execution scripts
│   └── results/                  # Benchmark results & reports
│
├── thesis/                        # Original thesis code (archived)
│   ├── README.md                 # Archive explanation
│   ├── swarm_algorithms/         # Original implementations
│   └── experiments/              # Original experiments
│
└── .kiro/                         # Kiro specification documents
    └── specs/
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Follow the code style** (PEP 8 for Python)
4. **Document your changes** with docstrings and comments
5. **Submit a pull request** with a clear description

### Development Setup

```bash
# Clone your fork
git clone https://github.com/mollinetti/nelder-mead-optimization.git
cd nelder-mead-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -e ".[dev,plotting]"

# Run tests
python -m pytest tests/
```

## Citation

If you use this library in your research, please cite:

```bibtex
@software{nelder_mead_library,
  title = {Nelder-Mead Optimization Library},
  author = {Marco Mollinetti},
  year = {2024},
  url = {https://github.com/mollinetti/nelder-mead-optimization}
}
```

### References

- Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308-313.
- Gao, F., & Han, L. (2012). Implementing the Nelder-Mead simplex algorithm with adaptive parameters. *Computational Optimization and Applications*, 51(1), 259-277.
- Deb, K. (2000). An efficient constraint handling method for genetic algorithms. *Computer Methods in Applied Mechanics and Engineering*, 186(2-4), 311-338.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This library was developed as part of doctoral research on optimization algorithms
- Thanks to the optimization research community for benchmark problems and test suites
- Special thanks to contributors and users who provided feedback

## Support

For questions, issues, or feature requests:

- **Issues**: [GitHub Issues](https://github.com/mollinetti/nelder-mead-optimization/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mollinetti/nelder-mead-optimization/discussions)
- **Email**: marco.mollinetti@gmail.com

---

**Happy Optimizing! 🚀**
