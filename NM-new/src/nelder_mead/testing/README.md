# Testing Infrastructure

This module provides infrastructure for running batch experiments with optimization algorithms on benchmark problems.

## Components

### BatchRunner

The `BatchRunner` class provides methods for:

1. **Running single experiments**: Execute an algorithm on a problem multiple times with different random seeds
2. **Comparing algorithms**: Run multiple algorithms on multiple problems and organize results for comparison
3. **Recording results**: Store all results in structured format with detailed statistics
4. **Exporting data**: Export results to dictionary format suitable for JSON/CSV export

### ExperimentResult

The `ExperimentResult` dataclass stores the results from a single algorithm run:

- `algorithm_name`: Name of the algorithm
- `problem_name`: Name of the problem
- `seed`: Random seed used
- `best_solution`: Best solution found
- `best_fitness`: Best objective value
- `best_violation`: Total constraint violation
- `fes_used`: Number of function evaluations used
- `convergence_history`: List of (fes, fitness) tuples tracking progress
- `wall_time`: Execution time in seconds
- `success`: Whether the optimum was reached within tolerance
- `final_simplex`: Final simplex state (for Nelder-Mead algorithms)

## Usage Examples

### Example 1: Single Experiment

Run an algorithm on a problem multiple times:

```python
from nelder_mead.testing import BatchRunner
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.benchmarks.unconstrained import Sphere

runner = BatchRunner(verbose=True)
problem = Sphere(dim=10)

results = runner.run_experiment(
    algorithm_class=NelderMead,
    problem=problem,
    num_runs=30,
    max_fes=5000,
    init_method="spendleySimplex"
)

# Analyze results
fitness_values = [r.best_fitness for r in results]
print(f"Mean fitness: {np.mean(fitness_values):.6e}")
print(f"Success rate: {sum(r.success for r in results)}/{len(results)}")
```

### Example 2: Algorithm Comparison

Compare multiple algorithms on multiple problems:

```python
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
from nelder_mead.benchmarks.unconstrained import Rosenbrock

runner = BatchRunner(verbose=True)

algorithm_configs = [
    {'class': NelderMead, 'name': 'Classic-NM', 'max_fes': 5000},
    {'class': AdaptiveNelderMead, 'name': 'Adaptive-NM', 'max_fes': 5000}
]

problems = [
    Sphere(dim=10),
    Rosenbrock(dim=10)
]

results = runner.compare_algorithms(
    algorithm_configs=algorithm_configs,
    problems=problems,
    num_runs=30,
    seeds=range(1, 31)
)

# Access results: results['Classic-NM']['Sphere'] gives list of 30 ExperimentResult objects
```

### Example 3: Export Results

Export results to structured format:

```python
# Run experiment
results = runner.run_experiment(
    algorithm_class=NelderMead,
    problem=Sphere(dim=5),
    num_runs=10,
    max_fes=2000
)

# Export to dictionary
export_dict = runner.export_results_to_dict(results)

# Access statistics
print(export_dict['statistics']['mean'])
print(export_dict['statistics']['success_rate'])

# Access individual runs
for run in export_dict['runs']:
    print(f"Seed {run['seed']}: fitness = {run['best_fitness']}")
```

### Example 4: Constrained Problems

Run on constrained problems with barrier methods:

```python
from nelder_mead.benchmarks.constrained import G01
from nelder_mead.constraints.deb_barrier import DebBarrier

problem = G01()
barrier = DebBarrier(
    bounds=(problem.bounds[0], problem.bounds[1]),
    num_solutions=problem.dim + 1
)

results = runner.run_experiment(
    algorithm_class=NelderMead,
    problem=problem,
    num_runs=30,
    max_fes=10000,
    barrier=barrier
)
```

## Features

### Reproducibility

All experiments are fully reproducible when using the same random seeds:

```python
# Run 1
results1 = runner.run_experiment(
    algorithm_class=NelderMead,
    problem=Sphere(dim=5),
    num_runs=3,
    seeds=[1, 2, 3],
    max_fes=1000
)

# Run 2 (identical configuration)
results2 = runner.run_experiment(
    algorithm_class=NelderMead,
    problem=Sphere(dim=5),
    num_runs=3,
    seeds=[1, 2, 3],
    max_fes=1000
)

# Results are identical
assert results1[0].best_fitness == results2[0].best_fitness
```

### Convergence Tracking

Convergence history is automatically recorded at regular intervals:

```python
result = results[0]
for fes, fitness in result.convergence_history:
    print(f"FES {fes}: fitness = {fitness:.6e}")
```

### Success Detection

Success is automatically detected when the known optimum is reached within tolerance:

```python
results = runner.run_experiment(
    algorithm_class=NelderMead,
    problem=Sphere(dim=5),  # Known optimum = 0.0
    num_runs=30,
    success_tolerance=1e-2  # Success if |fitness - optimum| < 1e-2
)

success_rate = sum(r.success for r in results) / len(results)
print(f"Success rate: {success_rate * 100:.1f}%")
```

### Statistical Analysis

The `export_results_to_dict` method computes comprehensive statistics:

- Mean, median, standard deviation
- Best and worst fitness values
- Success rate
- Average function evaluations
- Average wall time

```python
export_dict = runner.export_results_to_dict(results)
stats = export_dict['statistics']

print(f"Mean: {stats['mean']:.6e}")
print(f"Std: {stats['std']:.6e}")
print(f"Success rate: {stats['success_rate']:.2%}")
```

## Requirements Validated

The BatchRunner implementation validates the following requirements:

- **Requirement 15.1**: Run each algorithm on each problem multiple times with different seeds
- **Requirement 15.2**: Record results in structured format
- **Requirement 15.4**: Use consistent problem configurations when comparing algorithms

## Testing

The BatchRunner is thoroughly tested with:

- **18 unit tests** in `tests/unit/test_batch_runner.py`
- **9 integration tests** in `tests/integration/test_batch_runner_integration.py`

All tests verify:
- Correct execution with multiple seeds
- Reproducibility with fixed seeds
- Proper result recording and statistics
- Algorithm comparison functionality
- Export functionality
- Error handling for invalid inputs

Run tests with:
```bash
pytest tests/unit/test_batch_runner.py -v
pytest tests/integration/test_batch_runner_integration.py -v
```

## See Also

- `nelder_mead.algorithms`: Algorithm implementations
- `nelder_mead.benchmarks`: Benchmark problem suites
- `nelder_mead.constraints`: Constraint handling methods
- Example script: `examples/batch_runner_example.py`
