"""
Refactored Implementation Benchmark Script

This script runs the refactored Nelder-Mead implementation on the benchmark suite
using the SAME configuration as the baseline (Task 12.1) for fair comparison.

Task 12.2: Run refactored implementation on same benchmark suite
- Execute on same problems with same configurations
- Use same seeds (101-105) for reproducibility
- Use same max FES (10,000)
- Use same initialization (Spendley Simplex)
- Use same restart strategy (Gaussian Best)
- Record refactored results for comparison
"""

import numpy as np
import json
from datetime import datetime
from nelder_mead.algorithms.nelder_mead import NelderMead
from nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
from nelder_mead.benchmarks.unconstrained import Sphere, Rosenbrock, Rastrigin
from nelder_mead.benchmarks.constrained import G01, G04, G06


def run_single_experiment(algorithm_class, problem, seed, max_fes=10000):
    """
    Run a single experiment.
    
    Args:
        algorithm_class: NelderMead or AdaptiveNelderMead class
        problem: Benchmark problem instance
        seed: Random seed
        max_fes: Maximum function evaluations
    
    Returns:
        dict: Results including best fitness, best solution, FES used, etc.
    """
    lower, upper = problem.bounds
    
    # Create algorithm instance with SAME configuration as baseline
    alg = algorithm_class(
        objective_fn=problem.objective,
        constraint_eq_fn=problem.constraint_eq if hasattr(problem, 'constraint_eq') else None,
        constraint_ineq_fn=problem.constraint_ineq if hasattr(problem, 'constraint_ineq') else None,
        lower_bounds=lower,
        upper_bounds=upper,
        max_fes=max_fes,
        init_method="spendleySimplex",
        seed=seed,
        restart_strategy="gaussian_best"
    )
    
    # Run the algorithm
    alg.run()
    
    # Extract results
    best_solution, best_fitness, eq_vio, ineq_vio, total_violation = alg.get_best_solution()
    
    # Check if solution is feasible and successful
    is_feasible = total_violation < 1e-6
    
    # For unconstrained problems, check if within 1% of optimum
    # For constrained problems, check if feasible
    if hasattr(problem, 'optimum_value'):
        if problem.optimum_value == 0:
            # Avoid division by zero
            is_successful = abs(best_fitness - problem.optimum_value) < 1e-2
        else:
            relative_error = abs(best_fitness - problem.optimum_value) / abs(problem.optimum_value)
            is_successful = relative_error < 0.01 and is_feasible
    else:
        is_successful = is_feasible
    
    return {
        'seed': seed,
        'best_fitness': float(best_fitness),
        'best_solution': best_solution.tolist(),
        'best_violation': float(total_violation),
        'fes_used': int(alg.fes),
        'is_feasible': bool(is_feasible),
        'is_successful': bool(is_successful)
    }


def run_benchmark_suite(algorithm_class, algorithm_name, num_runs=5, max_fes=10000):
    """
    Run the algorithm on the full benchmark suite.
    
    Args:
        algorithm_class: NelderMead or AdaptiveNelderMead class
        algorithm_name: Name of the algorithm for reporting
        num_runs: Number of independent runs per problem
        max_fes: Maximum function evaluations per run
    
    Returns:
        dict: Complete results for all problems
    """
    results = {
        'algorithm': algorithm_name,
        'timestamp': datetime.now().isoformat(),
        'num_runs': num_runs,
        'max_fes': max_fes,
        'problems': {}
    }
    
    # Define benchmark problems (SAME as baseline)
    unconstrained_problems = {
        'Sphere': Sphere(dim=10),
        'Rosenbrock': Rosenbrock(dim=10),
        'Rastrigin': Rastrigin(dim=10)
    }
    
    constrained_problems = {
        'G01': G01(),
        'G04': G04(),
        'G06': G06()
    }
    
    all_problems = {**unconstrained_problems, **constrained_problems}
    
    # Run experiments
    for problem_name, problem in all_problems.items():
        print(f"\n{'='*60}")
        print(f"Running {algorithm_name} on {problem_name} (dim={problem.dim})")
        print(f"{'='*60}")
        
        problem_results = []
        seeds = range(101, 101 + num_runs)  # SAME seeds as baseline (101-105)
        
        for i, seed in enumerate(seeds, 1):
            print(f"  Run {i}/{num_runs} (seed={seed})...", end=' ')
            try:
                result = run_single_experiment(
                    algorithm_class=algorithm_class,
                    problem=problem,
                    seed=seed,
                    max_fes=max_fes
                )
                problem_results.append(result)
                print(f"✓ f={result['best_fitness']:.6e}, vio={result['best_violation']:.6e}")
            except Exception as e:
                print(f"✗ Error: {e}")
                problem_results.append({
                    'seed': seed,
                    'error': str(e),
                    'best_fitness': None,
                    'best_violation': None,
                    'fes_used': None,
                    'is_feasible': False,
                    'is_successful': False
                })
        
        # Compute statistics
        valid_results = [r for r in problem_results if r.get('best_fitness') is not None]
        
        if valid_results:
            fitness_values = [r['best_fitness'] for r in valid_results]
            violation_values = [r['best_violation'] for r in valid_results]
            fes_values = [r['fes_used'] for r in valid_results]
            success_count = sum(1 for r in valid_results if r['is_successful'])
            feasible_count = sum(1 for r in valid_results if r['is_feasible'])
            
            statistics = {
                'mean_fitness': float(np.mean(fitness_values)),
                'median_fitness': float(np.median(fitness_values)),
                'std_fitness': float(np.std(fitness_values)),
                'best_fitness': float(np.min(fitness_values)),
                'worst_fitness': float(np.max(fitness_values)),
                'mean_violation': float(np.mean(violation_values)),
                'mean_fes': float(np.mean(fes_values)),
                'success_rate': float(success_count / len(valid_results)),
                'feasibility_rate': float(feasible_count / len(valid_results)),
                'num_valid_runs': len(valid_results)
            }
        else:
            statistics = {
                'mean_fitness': None,
                'median_fitness': None,
                'std_fitness': None,
                'best_fitness': None,
                'worst_fitness': None,
                'mean_violation': None,
                'mean_fes': None,
                'success_rate': 0.0,
                'feasibility_rate': 0.0,
                'num_valid_runs': 0
            }
        
        results['problems'][problem_name] = {
            'dimension': problem.dim,
            'optimum_value': problem.optimum_value if hasattr(problem, 'optimum_value') else None,
            'runs': problem_results,
            'statistics': statistics
        }
        
        # Print summary
        print(f"\n  Summary for {problem_name}:")
        print(f"    Best fitness:     {statistics['best_fitness']:.6e}" if statistics['best_fitness'] else "    Best fitness:     N/A")
        print(f"    Mean fitness:     {statistics['mean_fitness']:.6e}" if statistics['mean_fitness'] else "    Mean fitness:     N/A")
        print(f"    Std fitness:      {statistics['std_fitness']:.6e}" if statistics['std_fitness'] else "    Std fitness:      N/A")
        print(f"    Success rate:     {statistics['success_rate']*100:.1f}%")
        print(f"    Feasibility rate: {statistics['feasibility_rate']*100:.1f}%")
    
    return results


def main():
    """Main function to run refactored implementation experiments."""
    print("="*60)
    print("REFACTORED BENCHMARK: Nelder-Mead Implementation")
    print("="*60)
    print("\nTask 12.2: Run refactored implementation on benchmark suite")
    print("- Unconstrained: Sphere, Rosenbrock, Rastrigin")
    print("- Constrained: G01, G04, G06")
    print("\nConfiguration (SAME as baseline):")
    print("  - Runs per problem: 5")
    print("  - Seeds: 101-105")
    print("  - Max FES: 10000")
    print("  - Initialization: Spendley Simplex")
    print("  - Restart: Gaussian Best")
    
    # Run NM (Classic Nelder-Mead)
    print("\n" + "="*60)
    print("Running Classic Nelder-Mead (NM)")
    print("="*60)
    nm_results = run_benchmark_suite(
        algorithm_class=NelderMead,
        algorithm_name="NM",
        num_runs=5,
        max_fes=10000
    )
    
    # Save results
    output_file = 'refactored_results_nm.json'
    with open(output_file, 'w') as f:
        json.dump(nm_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Run ANM (Adaptive Nelder-Mead)
    print("\n" + "="*60)
    print("Running Adaptive Nelder-Mead (ANM)")
    print("="*60)
    anm_results = run_benchmark_suite(
        algorithm_class=AdaptiveNelderMead,
        algorithm_name="ANM",
        num_runs=5,
        max_fes=10000
    )
    
    # Save results
    output_file = 'refactored_results_anm.json'
    with open(output_file, 'w') as f:
        json.dump(anm_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print final summary
    print("\n" + "="*60)
    print("REFACTORED BENCHMARK COMPLETE")
    print("="*60)
    print("\nResults Summary:")
    print("\nClassic Nelder-Mead (NM):")
    for problem_name, problem_data in nm_results['problems'].items():
        stats = problem_data['statistics']
        print(f"  {problem_name:12s}: Best={stats['best_fitness']:.6e}, Mean={stats['mean_fitness']:.6e}, Success={stats['success_rate']*100:.1f}%" 
              if stats['best_fitness'] else f"  {problem_name:12s}: Failed")
    
    print("\nAdaptive Nelder-Mead (ANM):")
    for problem_name, problem_data in anm_results['problems'].items():
        stats = problem_data['statistics']
        print(f"  {problem_name:12s}: Best={stats['best_fitness']:.6e}, Mean={stats['mean_fitness']:.6e}, Success={stats['success_rate']*100:.1f}%"
              if stats['best_fitness'] else f"  {problem_name:12s}: Failed")
    
    print("\n✓ Refactored results recorded for Task 12.2")
    print("\nNext step: Run Task 12.3 to compare baseline vs refactored results")


if __name__ == '__main__':
    main()
