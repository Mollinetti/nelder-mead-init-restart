"""
Baseline Benchmark Script for Original Implementation

This script runs the original Nelder-Mead implementation from swarm_algorithms/
on the benchmark suite to establish baseline results for comparison with the
refactored implementation.

Task 12.1: Run original implementation on benchmark suite
- Execute on unconstrained problems (Sphere, Rosenbrock, Rastrigin)
- Execute on constrained problems (G01, G04, G06)
- Record baseline results (best fitness, mean fitness, success rate)
"""

import numpy as np
import json
from datetime import datetime
from swarm_algorithms.base import BaseAlgorithm
from nelder_mead.benchmarks.unconstrained import Sphere, Rosenbrock, Rastrigin
from nelder_mead.benchmarks.constrained import G01, G04, G06


# Create a wrapper for NM that properly handles the violation function
class OriginalNM(BaseAlgorithm):
    """
    Wrapper for original Nelder-Mead implementation that properly handles
    the violation function interface.
    """
    
    def __init__(self, objective_function, violation_function_eq, violation_function_ineq, 
                 lower_bounds, upper_bounds, max_fes=500, num_solutions=50, 
                 init_method="spendleySimplex", seed=101,
                 delta_r=1., delta_e=2., delta_oc=0.5, delta_ic=-0.5, gamma_s=0.5, 
                 restart="gaussian_best", d_nm=25):
        
        # Initialize base with d+1 solutions for d-dimensional problem
        super(OriginalNM, self).__init__(
            objective_function, violation_function_eq, violation_function_ineq,
            lower_bounds, upper_bounds, max_fes, d_nm + 1, init_method, seed
        )
        
        self.delta_r = delta_r
        self.delta_e = delta_e
        self.delta_oc = delta_oc
        self.delta_ic = delta_ic
        self.gamma_s = gamma_s
        self.num_solutions = d_nm + 1
        self.algorithm_initials = "NM"
        
        # Restart strategy
        self.__available_strategies = ["uniform", "gaussian", "gaussian_best"]
        self.restart_strategy = self.__validate_restart(restart)
        
        self.thresh = 10e-7
        self.no_improv_break = 10 * self.dim
        self.no_improv_counter = 0
    
    def __validate_restart(self, restart_strategy):
        restart_strategy = restart_strategy.lower()
        if restart_strategy in self.__available_strategies:
            return getattr(self, "restart_" + restart_strategy)
        else:
            raise ValueError(f"{restart_strategy} is not a valid mutation strategy. Available: {self.__available_strategies}")
    
    def restart_gaussian(self):
        return self.enforce_solution(np.random.multivariate_normal(np.zeros(self.dim), np.identity(self.dim)))
    
    def restart_uniform(self):
        return self.enforce_solution(np.random.uniform(self.lower_bounds, self.upper_bounds, (self.dim)))
    
    def restart_gaussian_best(self):
        return self.enforce_solution(self.global_best_solution + np.random.multivariate_normal(np.zeros(self.dim), np.identity(self.dim)))
    
    def get_centroid(self):
        return np.sum(self.solutions[:-1], axis=0) / (self.num_solutions - 1)
    
    def shrink(self, y_0, y_i):
        return self.enforce_solution(y_0 + (self.gamma_s * (y_i - y_0)))
    
    def reflection(self, centroid, sol):
        return self.enforce_solution(centroid + (self.delta_r * (centroid - sol)))
    
    def inside_contraction(self, centroid, refl):
        return self.enforce_solution(centroid + (self.delta_ic * (refl - centroid)))
    
    def outside_contraction(self, centroid, refl):
        return self.enforce_solution(centroid + (self.delta_oc * (refl - centroid)))
    
    def expansion(self, centroid, refl):
        return self.enforce_solution(centroid + (self.delta_e * (refl - centroid)))
    
    def stop_fmin1_simplex(self, tolFun=10e-12):
        fit_list = np.asarray([np.abs(self.solutions_fitness[i] - self.solutions_fitness[0]) for i in range(self.num_solutions)])
        return np.max(fit_list) <= tolFun
    
    def stop_fmin2_simplex(self, tolX=10e-8):
        fit_list = np.asarray([np.linalg.norm(self.solutions[i] - self.solutions[0], ord=np.inf) for i in range(self.num_solutions)])
        return np.max(fit_list) <= tolX
    
    def stop_std_dev(self, epsilon=10e-7):
        f_bar = (1 / (self.num_solutions + 1)) * np.sum(self.solutions_fitness)
        return np.sum([(self.solutions_fitness[i] - f_bar) / self.num_solutions for i in range(self.num_solutions)]) < epsilon
    
    def run(self):
        while not self.should_end():
            # Check for stop conditions and restart
            if self.stop_fmin1_simplex() or self.stop_fmin2_simplex() or self.stop_std_dev():
                for s in range(self.num_solutions):
                    self.solutions[s] = self.enforce_solution(self.restart_strategy())
                    self.solutions_fitness[s], self.solutions_violations_eq[s], self.solutions_violations_ineq[s] = self.objective_function(self.solutions[s])
                    if self.should_end():
                        return True
            
            # Sort solutions by fitness
            sorted_sols = np.asarray(sorted(list(np.c_[self.solutions, self.solutions_fitness, self.solutions_violations_eq, self.solutions_violations_ineq]), key=lambda x: x[-3]))
            
            self.solutions = np.copy(sorted_sols[:, :self.dim])
            self.solutions_fitness = np.copy(sorted_sols[:, -3])
            self.solutions_violations_eq = np.copy(sorted_sols[:, -2])
            self.solutions_violations_ineq = np.copy(sorted_sols[:, -1])
            
            # Calculate centroid
            centroid = self.get_centroid()
            
            # Perform REFLECTION
            y_r = self.reflection(centroid, self.solutions[-1])
            f_y_r, vio_eq_y_r, vio_ineq_y_r = self.objective_function(y_r)
            
            if self.should_end():
                return True
            
            # If f^0 <= f^y_r < f^n-1, replace y^n by the reflected point y^r
            if (self.solutions_fitness[0] <= f_y_r) and (self.solutions_fitness[-2] > f_y_r):
                self.solutions[-1], self.solutions_fitness[-1], self.solutions_violations_eq[-1], self.solutions_violations_ineq[-1] = np.copy(y_r), np.copy(f_y_r), np.copy(vio_eq_y_r), np.copy(vio_ineq_y_r)
                continue
            
            # If f^y_r < f^0 then attempt expansion
            if f_y_r < self.solutions_fitness[0]:
                y_e = self.expansion(centroid, y_r)
                f_y_e, vio_eq_y_e, vio_ineq_y_e = self.objective_function(y_e)
                
                if f_y_e < f_y_r:
                    self.solutions[-1], self.solutions_fitness[-1], self.solutions_violations_eq[-1], self.solutions_violations_ineq[-1] = np.copy(y_e), np.copy(f_y_e), np.copy(vio_eq_y_e), np.copy(vio_ineq_y_e)
                    if self.should_end():
                        return True
                    continue
                else:
                    self.solutions[-1], self.solutions_fitness[-1], self.solutions_violations_eq[-1], self.solutions_violations_ineq[-1] = np.copy(y_r), np.copy(f_y_r), np.copy(vio_eq_y_r), np.copy(vio_ineq_y_r)
                    if self.should_end():
                        return True
                    continue
            
            # If f^r < f^n do an outside contraction
            if (f_y_r < self.solutions_fitness[-1]) and (f_y_r >= self.solutions_fitness[-2]):
                y_oc = self.outside_contraction(centroid, y_r)
                f_y_oc, vio_eq_y_oc, vio_ineq_y_oc = self.objective_function(y_oc)
                
                if f_y_oc <= f_y_r:
                    self.solutions[-1], self.solutions_fitness[-1], self.solutions_violations_eq[-1], self.solutions_violations_ineq[-1] = np.copy(y_oc), np.copy(f_y_oc), np.copy(vio_eq_y_oc), np.copy(vio_ineq_y_oc)
                    if self.should_end():
                        return True
                    continue
            
            # Otherwise do an inside contraction
            if f_y_r >= self.solutions_fitness[-1]:
                y_ic = self.inside_contraction(centroid, y_r)
                f_y_ic, vio_eq_y_ic, vio_ineq_y_ic = self.objective_function(y_ic)
                
                if f_y_ic <= self.solutions_fitness[-1]:
                    self.solutions[-1], self.solutions_fitness[-1], self.solutions_violations_eq[-1], self.solutions_violations_ineq[-1] = np.copy(y_ic), np.copy(f_y_ic), np.copy(vio_eq_y_ic), np.copy(vio_ineq_y_ic)
                    if self.should_end():
                        return True
                    continue
            
            # If everything else fails perform a shrink
            for s in range(self.num_solutions):
                self.solutions[s] = self.shrink(self.solutions[0], self.solutions[s])
                self.solutions_fitness[s], self.solutions_violations_eq[s], self.solutions_violations_ineq[s] = self.objective_function(self.solutions[s])
                if self.should_end():
                    return True


class OriginalANM(OriginalNM):
    """Adaptive Nelder-Mead with dimension-dependent parameters."""
    
    def __init__(self, *args, **kwargs):
        super(OriginalANM, self).__init__(*args, **kwargs)
        self.algorithm_initials = "ANM"
        
        # Change parameters to reflect the ones defined at Gao et al.
        self.delta_e = 1 + (2 / self.dim)
        self.delta_oc = 0.75 - (1 / (2 * self.dim))
        self.delta_ic = -(0.75 - (1 / (2 * self.dim)))
        self.gamma_s = 1 - (1 / self.dim)


def create_violation_functions(problem):
    """
    Create violation functions compatible with the BaseAlgorithm interface.
    
    BaseAlgorithm expects separate violation functions for equality and
    inequality constraints that return arrays of violations.
    """
    def eq_violation_fn(x):
        # Equality constraint violations
        if hasattr(problem, 'constraint_eq'):
            eq_violations = problem.constraint_eq(x)
            if len(eq_violations) > 0:
                return np.abs(eq_violations)
        return np.array([0.0])
    
    def ineq_violation_fn(x):
        # Inequality constraint violations (g(x) <= 0)
        # Also include bound violations
        lower, upper = problem.bounds
        bound_vio_lower = np.maximum(0, lower - x)
        bound_vio_upper = np.maximum(0, x - upper)
        
        ineq_vios = []
        if hasattr(problem, 'constraint_ineq'):
            ineq_violations = problem.constraint_ineq(x)
            if len(ineq_violations) > 0:
                ineq_vios = np.maximum(0, ineq_violations)
        
        # Combine bound violations and inequality violations
        all_vios = np.concatenate([bound_vio_lower, bound_vio_upper])
        if len(ineq_vios) > 0:
            all_vios = np.concatenate([all_vios, ineq_vios])
        
        return all_vios if len(all_vios) > 0 else np.array([0.0])
    
    return eq_violation_fn, ineq_violation_fn


def compute_total_violation(problem, x):
    """Compute total violation for a solution."""
    eq_fn, ineq_fn = create_violation_functions(problem)
    eq_vio = eq_fn(x)
    ineq_vio = ineq_fn(x)
    return np.sum(eq_vio) + np.sum(ineq_vio)


def run_single_experiment(algorithm_class, problem, seed, max_fes=10000, barrier_type="deb"):
    """
    Run a single experiment with the original implementation.
    
    Args:
        algorithm_class: OriginalNM or OriginalANM class
        problem: Benchmark problem instance
        seed: Random seed
        max_fes: Maximum function evaluations
        barrier_type: Type of barrier method ("hard", "deb", "augmented", "progressive")
    
    Returns:
        dict: Results including best fitness, best solution, FES used, etc.
    """
    lower, upper = problem.bounds
    
    # Create violation functions
    eq_violation_fn, ineq_violation_fn = create_violation_functions(problem)
    
    # For NM, we need d+1 solutions where d is the dimension
    d_nm = problem.dim
    
    # Create algorithm instance
    alg = algorithm_class(
        objective_function=problem.objective,
        violation_function_eq=eq_violation_fn,
        violation_function_ineq=ineq_violation_fn,
        lower_bounds=lower,
        upper_bounds=upper,
        max_fes=max_fes,
        num_solutions=d_nm,  # This will be overridden to d+1 in OriginalNM.__init__
        init_method="spendleySimplex",
        seed=seed,
        restart="gaussian_best",
        d_nm=d_nm
    )
    
    # Run the algorithm
    alg.run()
    
    # Extract results
    best_solution = alg.global_best_solution
    best_fitness = alg.global_best_solution_fitness
    best_violation = compute_total_violation(problem, best_solution)
    fes_used = alg.fes
    
    # Check if solution is feasible and successful
    is_feasible = best_violation < 1e-6
    
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
        'best_violation': float(best_violation),
        'fes_used': int(fes_used),
        'is_feasible': bool(is_feasible),
        'is_successful': bool(is_successful)
    }


def run_benchmark_suite(algorithm_class, algorithm_name, num_runs=30, max_fes=10000):
    """
    Run the algorithm on the full benchmark suite.
    
    Args:
        algorithm_class: NM or ANM class
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
    
    # Define benchmark problems
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
        seeds = range(101, 101 + num_runs)
        
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
        print(f"    Best fitness:    {statistics['best_fitness']:.6e}" if statistics['best_fitness'] else "    Best fitness:    N/A")
        print(f"    Mean fitness:    {statistics['mean_fitness']:.6e}" if statistics['mean_fitness'] else "    Mean fitness:    N/A")
        print(f"    Std fitness:     {statistics['std_fitness']:.6e}" if statistics['std_fitness'] else "    Std fitness:     N/A")
        print(f"    Success rate:    {statistics['success_rate']*100:.1f}%")
        print(f"    Feasibility rate: {statistics['feasibility_rate']*100:.1f}%")
    
    return results


def main():
    """Main function to run baseline experiments."""
    print("="*60)
    print("BASELINE BENCHMARK: Original Nelder-Mead Implementation")
    print("="*60)
    print("\nTask 12.1: Run original implementation on benchmark suite")
    print("- Unconstrained: Sphere, Rosenbrock, Rastrigin")
    print("- Constrained: G01, G04, G06")
    print("\nConfiguration:")
    print("  - Runs per problem: 30")
    print("  - Max FES: 10000")
    print("  - Initialization: Spendley Simplex")
    print("  - Restart: Gaussian Best")
    
    # Run NM (Classic Nelder-Mead)
    print("\n" + "="*60)
    print("Running Classic Nelder-Mead (NM)")
    print("="*60)
    nm_results = run_benchmark_suite(
        algorithm_class=OriginalNM,
        algorithm_name="NM",
        num_runs=30,
        max_fes=10000
    )
    
    # Save results
    output_file = 'baseline_results_original_nm.json'
    with open(output_file, 'w') as f:
        json.dump(nm_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Run ANM (Adaptive Nelder-Mead)
    print("\n" + "="*60)
    print("Running Adaptive Nelder-Mead (ANM)")
    print("="*60)
    anm_results = run_benchmark_suite(
        algorithm_class=OriginalANM,
        algorithm_name="ANM",
        num_runs=30,
        max_fes=10000
    )
    
    # Save results
    output_file = 'baseline_results_original_anm.json'
    with open(output_file, 'w') as f:
        json.dump(anm_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print final summary
    print("\n" + "="*60)
    print("BASELINE BENCHMARK COMPLETE")
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
    
    print("\n✓ Baseline results recorded for Task 12.1")


if __name__ == '__main__':
    main()
