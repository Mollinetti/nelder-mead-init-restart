"""
Property-based tests for algorithm components.

This module tests universal correctness properties of algorithm-related operations
using property-based testing with the Hypothesis library. Each property is validated
across 100+ randomized test cases to ensure correctness holds for all valid inputs.

Feature: nelder-mead-thesis-refactor
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.nelder_mead.core.restart_strategies import (
    UniformRestart,
    GaussianRestart,
    GaussianBestRestart,
    create_restart_strategy,
)


# Custom strategies for generating test data
@st.composite
def bounds_strategy(draw, min_dim=2, max_dim=10):
    """Generate valid lower and upper bounds."""
    dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    lower = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=0.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    upper = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    return dim, np.array(lower), np.array(upper)


@st.composite
def best_solution_strategy(draw, dim):
    """Generate a valid best solution within bounds."""
    solution = draw(
        st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    return np.array(solution)


# Feature: nelder-mead-thesis-refactor, Property 16: Restart strategies generate valid points
@settings(max_examples=100)
@given(
    bounds_data=bounds_strategy(),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_16_uniform_restart_generates_valid_points(bounds_data, seed):
    """
    Property 16: Restart strategies generate valid points (UniformRestart).
    
    For any restart strategy (uniform, Gaussian, Gaussian-best) and problem bounds,
    all generated restart points must be within bounds after enforcement.
    
    This test validates UniformRestart strategy.
    
    Validates: Requirements 5.1, 5.2, 5.3
    """
    dim, lower_bounds, upper_bounds = bounds_data
    bounds = (lower_bounds, upper_bounds)
    
    # Create UniformRestart strategy
    strategy = UniformRestart(seed=seed)
    
    # Generate multiple points to test consistency
    num_points = 10
    for _ in range(num_points):
        point = strategy.generate_point(dim, bounds)
        
        # Verify point is within bounds
        assert point.shape == (dim,), f"Point should have shape ({dim},), got {point.shape}"
        assert np.all(point >= lower_bounds), \
            f"Point {point} violates lower bounds {lower_bounds}"
        assert np.all(point <= upper_bounds), \
            f"Point {point} violates upper bounds {upper_bounds}"
        
        # Verify point is a valid numpy array with finite values
        assert np.all(np.isfinite(point)), f"Point {point} contains non-finite values"


@settings(max_examples=100)
@given(
    bounds_data=bounds_strategy(),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_16_gaussian_restart_generates_valid_points(bounds_data, seed):
    """
    Property 16: Restart strategies generate valid points (GaussianRestart).
    
    For any restart strategy (uniform, Gaussian, Gaussian-best) and problem bounds,
    all generated restart points must be within bounds after enforcement.
    
    This test validates GaussianRestart strategy.
    
    Validates: Requirements 5.1, 5.2, 5.3
    """
    dim, lower_bounds, upper_bounds = bounds_data
    bounds = (lower_bounds, upper_bounds)
    
    # Create GaussianRestart strategy
    strategy = GaussianRestart(seed=seed)
    
    # Generate multiple points to test consistency
    num_points = 10
    for _ in range(num_points):
        point = strategy.generate_point(dim, bounds)
        
        # Verify point is within bounds (clipped by the strategy)
        assert point.shape == (dim,), f"Point should have shape ({dim},), got {point.shape}"
        assert np.all(point >= lower_bounds), \
            f"Point {point} violates lower bounds {lower_bounds}"
        assert np.all(point <= upper_bounds), \
            f"Point {point} violates upper bounds {upper_bounds}"
        
        # Verify point is a valid numpy array with finite values
        assert np.all(np.isfinite(point)), f"Point {point} contains non-finite values"


@settings(max_examples=100)
@given(
    bounds_data=bounds_strategy(),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_16_gaussian_best_restart_generates_valid_points(bounds_data, seed):
    """
    Property 16: Restart strategies generate valid points (GaussianBestRestart).
    
    For any restart strategy (uniform, Gaussian, Gaussian-best) and problem bounds,
    all generated restart points must be within bounds after enforcement.
    
    This test validates GaussianBestRestart strategy.
    
    Validates: Requirements 5.1, 5.2, 5.3
    """
    dim, lower_bounds, upper_bounds = bounds_data
    bounds = (lower_bounds, upper_bounds)
    
    # Generate a best solution within bounds
    best_solution = lower_bounds + np.random.RandomState(seed).random(dim) * (upper_bounds - lower_bounds)
    
    # Create GaussianBestRestart strategy
    strategy = GaussianBestRestart(seed=seed)
    
    # Generate multiple points to test consistency
    num_points = 10
    for _ in range(num_points):
        point = strategy.generate_point(dim, bounds, best_solution=best_solution)
        
        # Verify point is within bounds (clipped by the strategy)
        assert point.shape == (dim,), f"Point should have shape ({dim},), got {point.shape}"
        assert np.all(point >= lower_bounds), \
            f"Point {point} violates lower bounds {lower_bounds}"
        assert np.all(point <= upper_bounds), \
            f"Point {point} violates upper bounds {upper_bounds}"
        
        # Verify point is a valid numpy array with finite values
        assert np.all(np.isfinite(point)), f"Point {point} contains non-finite values"


@settings(max_examples=100)
@given(
    bounds_data=bounds_strategy(),
    strategy_name=st.sampled_from(["uniform", "gaussian", "gaussian_best"]),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_16_factory_creates_valid_strategies(bounds_data, strategy_name, seed):
    """
    Property 16: Restart strategies generate valid points (Factory method).
    
    For any restart strategy created via factory method and problem bounds,
    all generated restart points must be within bounds after enforcement.
    
    This test validates the create_restart_strategy factory method.
    
    Validates: Requirements 5.1, 5.2, 5.3
    """
    dim, lower_bounds, upper_bounds = bounds_data
    bounds = (lower_bounds, upper_bounds)
    
    # Create strategy using factory
    strategy = create_restart_strategy(strategy_name, seed=seed)
    
    # Generate a best solution for gaussian_best strategy
    best_solution = lower_bounds + np.random.RandomState(seed).random(dim) * (upper_bounds - lower_bounds)
    
    # Generate multiple points to test consistency
    num_points = 10
    for _ in range(num_points):
        if strategy_name == "gaussian_best":
            point = strategy.generate_point(dim, bounds, best_solution=best_solution)
        else:
            point = strategy.generate_point(dim, bounds)
        
        # Verify point is within bounds
        assert point.shape == (dim,), f"Point should have shape ({dim},), got {point.shape}"
        assert np.all(point >= lower_bounds), \
            f"Point {point} violates lower bounds {lower_bounds}"
        assert np.all(point <= upper_bounds), \
            f"Point {point} violates upper bounds {upper_bounds}"
        
        # Verify point is a valid numpy array with finite values
        assert np.all(np.isfinite(point)), f"Point {point} contains non-finite values"


@settings(max_examples=100)
@given(
    bounds_data=bounds_strategy(),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_16_restart_strategies_respect_bounds_edge_cases(bounds_data, seed):
    """
    Property 16: Restart strategies generate valid points (Edge cases).
    
    Test edge cases where generated points might fall outside bounds before clipping.
    This ensures the clipping mechanism works correctly for all strategies.
    
    Validates: Requirements 5.1, 5.2, 5.3
    """
    dim, lower_bounds, upper_bounds = bounds_data
    bounds = (lower_bounds, upper_bounds)
    
    # Test with very tight bounds to increase likelihood of clipping
    tight_lower = np.full(dim, -0.1)
    tight_upper = np.full(dim, 0.1)
    tight_bounds = (tight_lower, tight_upper)
    
    # Test all strategies with tight bounds
    strategies = [
        UniformRestart(seed=seed),
        GaussianRestart(seed=seed),
    ]
    
    for strategy in strategies:
        for _ in range(5):
            point = strategy.generate_point(dim, tight_bounds)
            
            # Verify point is within tight bounds
            assert np.all(point >= tight_lower), \
                f"Point {point} violates tight lower bounds {tight_lower}"
            assert np.all(point <= tight_upper), \
                f"Point {point} violates tight upper bounds {tight_upper}"
    
    # Test GaussianBestRestart with best solution at boundary
    best_at_boundary = tight_upper.copy()
    strategy_best = GaussianBestRestart(seed=seed)
    
    for _ in range(5):
        point = strategy_best.generate_point(dim, tight_bounds, best_solution=best_at_boundary)
        
        # Verify point is within tight bounds
        assert np.all(point >= tight_lower), \
            f"Point {point} violates tight lower bounds {tight_lower}"
        assert np.all(point <= tight_upper), \
            f"Point {point} violates tight upper bounds {tight_upper}"


# ============================================================================
# Base Algorithm Property Tests (Properties 27-32)
# ============================================================================


# Helper: Create a simple test algorithm for property testing
class MockAlgorithm:
    """Mock algorithm for testing base algorithm properties."""
    
    def __init__(self, objective_fn, lower_bounds, upper_bounds, max_fes, 
                 num_solutions, seed, tracking_interval=None):
        from src.nelder_mead.algorithms.base_algorithm import BaseAlgorithm
        
        # Create a concrete implementation for testing
        class ConcreteAlgorithm(BaseAlgorithm):
            def run(self):
                # Simple implementation: evaluate random solutions
                self.population = self.initialize_population()
                self.fitness_values, _, _, _ = self.evaluate_population(self.population)
                
                # Continue evaluating until budget exhausted
                while not self.should_terminate():
                    # Generate and evaluate a random solution
                    solution = self.lower_bounds + self.rng.random(self.dim) * (
                        self.upper_bounds - self.lower_bounds
                    )
                    self.evaluate(solution)
        
        self.alg = ConcreteAlgorithm(
            objective_fn=objective_fn,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            max_fes=max_fes,
            num_solutions=num_solutions,
            seed=seed,
        )
        
        # Override tracking interval if specified
        if tracking_interval is not None:
            self.alg.tracking_interval = tracking_interval
    
    def run(self):
        self.alg.run()
    
    @property
    def fes(self):
        return self.alg.fes
    
    @property
    def best_fitness(self):
        return self.alg.best_fitness
    
    @property
    def memory_global_best(self):
        return self.alg.memory_global_best
    
    @property
    def num_solutions(self):
        return self.alg.num_solutions


# Feature: nelder-mead-thesis-refactor, Property 27: Function evaluation counter monotonicity
@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=5),
    max_fes=st.integers(min_value=10, max_value=50),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_27_fes_counter_monotonicity(dim, max_fes, seed):
    """
    Property 27: Function evaluation counter monotonicity.
    
    For any algorithm execution, the function evaluation counter must be 
    monotonically increasing: FES(t+1) >= FES(t) for all time steps t.
    
    Validates: Requirements 13.1
    """
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    
    # Create algorithm
    alg = MockAlgorithm(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=max_fes,
        num_solutions=dim + 1,
        seed=seed,
    )
    
    # Track FES values during execution
    fes_history = []
    
    # Manually step through algorithm to track FES
    alg.alg.population = alg.alg.initialize_population()
    
    # Record FES after each evaluation
    for i in range(len(alg.alg.population)):
        fes_before = alg.alg.fes
        alg.alg.evaluate(alg.alg.population[i])
        fes_after = alg.alg.fes
        fes_history.append(fes_after)
        
        # Verify monotonicity
        assert fes_after >= fes_before, \
            f"FES decreased from {fes_before} to {fes_after}"
        assert fes_after == fes_before + 1, \
            f"FES should increment by 1, got {fes_after - fes_before}"
    
    # Verify overall monotonicity
    for i in range(1, len(fes_history)):
        assert fes_history[i] >= fes_history[i-1], \
            f"FES not monotonic at step {i}: {fes_history[i-1]} -> {fes_history[i]}"


# Feature: nelder-mead-thesis-refactor, Property 28: Initial population counts toward FES
@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_28_initial_population_counts_toward_fes(dim, seed):
    """
    Property 28: Initial population counts toward FES.
    
    For any algorithm with population size n, after initialization and evaluation,
    the FES counter must equal n.
    
    Validates: Requirements 13.4
    """
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    num_solutions = dim + 1
    
    # Create algorithm
    alg = MockAlgorithm(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=1000,
        num_solutions=num_solutions,
        seed=seed,
    )
    
    # Initialize population
    alg.alg.population = alg.alg.initialize_population()
    
    # FES should still be 0 before evaluation
    assert alg.alg.fes == 0, \
        f"FES should be 0 before evaluation, got {alg.alg.fes}"
    
    # Evaluate population
    alg.alg.evaluate_population(alg.alg.population)
    
    # FES should equal population size
    assert alg.alg.fes == num_solutions, \
        f"FES should be {num_solutions} after evaluating initial population, got {alg.alg.fes}"


# Feature: nelder-mead-thesis-refactor, Property 29: Algorithm terminates at max FES
@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=5),
    max_fes=st.integers(min_value=10, max_value=50),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_29_algorithm_terminates_at_max_fes(dim, max_fes, seed):
    """
    Property 29: Algorithm terminates at max FES.
    
    For any algorithm with max_fes limit, the algorithm must terminate when 
    FES >= max_fes, and the final FES must not exceed max_fes by more than 
    the simplex size.
    
    Validates: Requirements 13.2
    """
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    num_solutions = dim + 1
    
    # Create and run algorithm
    alg = MockAlgorithm(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=max_fes,
        num_solutions=num_solutions,
        seed=seed,
    )
    
    alg.run()
    
    # Verify algorithm terminated
    assert alg.alg.should_terminate(), \
        f"Algorithm should have terminated at FES={alg.fes}"
    
    # Verify FES is at least max_fes
    assert alg.fes >= max_fes, \
        f"FES ({alg.fes}) should be >= max_fes ({max_fes})"
    
    # Verify FES doesn't exceed max_fes by more than simplex size
    # (last evaluation might push slightly over)
    assert alg.fes <= max_fes + num_solutions, \
        f"FES ({alg.fes}) exceeded max_fes ({max_fes}) by more than simplex size ({num_solutions})"


# Feature: nelder-mead-thesis-refactor, Property 30: Best solution monotonic improvement
@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=5),
    max_fes=st.integers(min_value=20, max_value=50),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_30_best_solution_monotonic_improvement(dim, max_fes, seed):
    """
    Property 30: Best solution monotonic improvement.
    
    For any algorithm execution, the best fitness value must be monotonically 
    non-increasing: best_fitness(t+1) <= best_fitness(t) for all time steps t.
    
    Validates: Requirements 14.1, 14.2
    """
    # Create simple sphere function (convex, easy to optimize)
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    
    # Create algorithm
    alg = MockAlgorithm(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=max_fes,
        num_solutions=dim + 1,
        seed=seed,
    )
    
    # Track best fitness during execution
    best_fitness_history = []
    
    # Initialize and evaluate population
    alg.alg.population = alg.alg.initialize_population()
    alg.alg.evaluate_population(alg.alg.population)
    best_fitness_history.append(alg.alg.best_fitness)
    
    # Continue evaluating solutions
    while not alg.alg.should_terminate():
        solution = alg.alg.lower_bounds + alg.alg.rng.random(alg.alg.dim) * (
            alg.alg.upper_bounds - alg.alg.lower_bounds
        )
        alg.alg.evaluate(solution)
        best_fitness_history.append(alg.alg.best_fitness)
    
    # Verify monotonic non-increasing property
    for i in range(1, len(best_fitness_history)):
        assert best_fitness_history[i] <= best_fitness_history[i-1], \
            f"Best fitness increased at step {i}: {best_fitness_history[i-1]} -> {best_fitness_history[i]}"


# Feature: nelder-mead-thesis-refactor, Property 31: Solution comparison considers constraints
@settings(max_examples=100, deadline=None)
@given(
    fitness1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    fitness2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    violation1=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    violation2=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_property_31_solution_comparison_considers_constraints(
    fitness1, fitness2, violation1, violation2
):
    """
    Property 31: Solution comparison considers constraints.
    
    For any two solutions s1 and s2, when comparing:
    - If both feasible: prefer lower objective value
    - If one feasible, one infeasible: prefer feasible
    - If both infeasible: prefer lower total violation
    
    Validates: Requirements 14.3
    """
    from src.nelder_mead.algorithms.base_algorithm import BaseAlgorithm
    
    # Create a dummy algorithm instance to access _is_better_solution
    def dummy_objective(x):
        return 0.0
    
    lower_bounds = np.array([0.0, 0.0])
    upper_bounds = np.array([1.0, 1.0])
    
    class TestAlgorithm(BaseAlgorithm):
        def run(self):
            pass
    
    alg = TestAlgorithm(
        objective_fn=dummy_objective,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=100,
        seed=42,
    )
    
    # Define feasibility threshold
    feasibility_tol = 1e-8
    
    feasible1 = violation1 < feasibility_tol
    feasible2 = violation2 < feasibility_tol
    
    result = alg._is_better_solution(fitness1, violation1, fitness2, violation2)
    
    if feasible1 and feasible2:
        # Both feasible: prefer lower fitness
        expected = fitness1 < fitness2
        assert result == expected, \
            f"Both feasible: expected {expected} (f1={fitness1}, f2={fitness2}), got {result}"
    elif feasible1 and not feasible2:
        # Solution 1 feasible, solution 2 infeasible: prefer solution 1
        assert result is True, \
            f"Solution 1 feasible, solution 2 infeasible: should prefer solution 1"
    elif not feasible1 and feasible2:
        # Solution 1 infeasible, solution 2 feasible: prefer solution 2
        assert result is False, \
            f"Solution 1 infeasible, solution 2 feasible: should prefer solution 2"
    else:
        # Both infeasible: prefer lower violation
        expected = violation1 < violation2
        assert result == expected, \
            f"Both infeasible: expected {expected} (v1={violation1}, v2={violation2}), got {result}"


# Feature: nelder-mead-thesis-refactor, Property 32: Convergence history records best at intervals
@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=5),
    max_fes=st.integers(min_value=50, max_value=200),
    tracking_interval=st.integers(min_value=5, max_value=20),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_32_convergence_history_records_best_at_intervals(
    dim, max_fes, tracking_interval, seed
):
    """
    Property 32: Convergence history records best at intervals.
    
    For any algorithm with tracking interval k, the convergence history must 
    contain an entry every k function evaluations.
    
    Validates: Requirements 14.5
    """
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    
    # Create algorithm with custom tracking interval
    alg = MockAlgorithm(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=max_fes,
        num_solutions=dim + 1,
        seed=seed,
        tracking_interval=tracking_interval,
    )
    
    # Run algorithm
    alg.run()
    
    # Verify convergence history exists
    assert len(alg.memory_global_best) > 0, \
        "Convergence history should not be empty"
    
    # Verify first entry is at FES=1 (initial evaluation)
    first_fes, _ = alg.memory_global_best[0]
    assert first_fes == 1, \
        f"First convergence history entry should be at FES=1, got {first_fes}"
    
    # Verify entries are recorded at intervals
    for i in range(1, len(alg.memory_global_best)):
        fes, best_fitness = alg.memory_global_best[i]
        
        # FES should be a multiple of tracking_interval (or close to max_fes)
        if fes < max_fes:
            assert fes % tracking_interval == 0, \
                f"FES {fes} should be multiple of tracking_interval {tracking_interval}"
        
        # Verify monotonicity of FES in history
        prev_fes, _ = alg.memory_global_best[i-1]
        assert fes > prev_fes, \
            f"FES in history should be increasing: {prev_fes} -> {fes}"
    
    # Verify best fitness is monotonically non-increasing in history
    for i in range(1, len(alg.memory_global_best)):
        _, current_best = alg.memory_global_best[i]
        _, prev_best = alg.memory_global_best[i-1]
        assert current_best <= prev_best, \
            f"Best fitness should be non-increasing in history: {prev_best} -> {current_best}"


# ============================================================================
# Restart Mechanism Property Tests (Property 17)
# ============================================================================


# Feature: nelder-mead-thesis-refactor, Property 17: Restart increments function evaluations
@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000),
    restart_strategy=st.sampled_from(["uniform", "gaussian", "gaussian_best"]),
)
def test_property_17_restart_increments_fes(dim, seed, restart_strategy):
    """
    Property 17: Restart increments function evaluations.
    
    For any simplex with n+1 vertices, restarting must increase the function 
    evaluation counter by exactly n+1 (since all vertices are re-evaluated).
    
    This test verifies that when a restart is triggered, the FES counter is 
    incremented by the number of simplex vertices (n+1 for n-dimensional problem).
    
    Validates: Requirements 5.4, 5.5, 13.5
    """
    from src.nelder_mead.algorithms.nelder_mead import NelderMead
    
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    num_solutions = dim + 1
    
    # Create NelderMead algorithm
    # Note: Stopping criteria tolerances are hardcoded in the algorithm
    alg = NelderMead(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=1000,  # High enough to allow multiple restarts
        num_solutions=num_solutions,
        seed=seed,
        restart_strategy=restart_strategy,
        stopping_criteria=["fminsearch_fun", "fminsearch_x"],
    )
    
    # Initialize the simplex
    alg.simplex = alg.initialize_population()
    (
        alg.fitness_values,
        alg.eq_violations_list,
        alg.ineq_violations_list,
        alg.total_violations,
    ) = alg.evaluate_population(alg.simplex)
    alg._sort_simplex()
    
    # Record FES before restart
    fes_before_restart = alg.fes
    
    # Manually trigger a restart
    alg._restart_simplex()
    
    # Record FES after restart
    fes_after_restart = alg.fes
    
    # Verify that FES increased by exactly num_solutions
    # The restart re-evaluates all vertices in the simplex
    fes_increment = fes_after_restart - fes_before_restart
    
    assert fes_increment == num_solutions, \
        f"Restart should increment FES by {num_solutions} (simplex size), " \
        f"but incremented by {fes_increment}. " \
        f"FES before: {fes_before_restart}, FES after: {fes_after_restart}"
    
    # Verify FES is monotonically increasing
    assert fes_after_restart > fes_before_restart, \
        f"FES should increase after restart: {fes_before_restart} -> {fes_after_restart}"
    
    # Verify the simplex was actually modified (at least some vertices changed)
    # We can't directly compare old vs new simplex since we don't save it,
    # but we can verify that the simplex is still valid
    assert alg.simplex.shape == (num_solutions, dim), \
        f"Simplex shape should be ({num_solutions}, {dim}), got {alg.simplex.shape}"
    
    # Verify all vertices are within bounds after restart
    assert np.all(alg.simplex >= lower_bounds), \
        "All simplex vertices should be >= lower bounds after restart"
    assert np.all(alg.simplex <= upper_bounds), \
        "All simplex vertices should be <= upper bounds after restart"
    
    # Verify fitness values were re-evaluated (should have num_solutions values)
    assert len(alg.fitness_values) == num_solutions, \
        f"Should have {num_solutions} fitness values after restart, got {len(alg.fitness_values)}"


@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_17_multiple_restarts_accumulate_fes(dim, seed):
    """
    Property 17: Restart increments function evaluations (Multiple restarts).
    
    Verify that multiple restarts correctly accumulate FES increments.
    Each restart should add exactly n+1 to the FES counter.
    
    Validates: Requirements 5.4, 5.5, 13.5
    """
    from src.nelder_mead.algorithms.nelder_mead import NelderMead
    
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    num_solutions = dim + 1
    
    # Create NelderMead algorithm
    alg = NelderMead(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=1000,
        num_solutions=num_solutions,
        seed=seed,
        restart_strategy="gaussian_best",
        stopping_criteria=["fminsearch_fun"],
    )
    
    # Initialize the simplex
    alg.simplex = alg.initialize_population()
    (
        alg.fitness_values,
        alg.eq_violations_list,
        alg.ineq_violations_list,
        alg.total_violations,
    ) = alg.evaluate_population(alg.simplex)
    alg._sort_simplex()
    
    # Track FES across multiple restarts
    fes_history = [alg.fes]
    num_restarts = 3
    
    for i in range(num_restarts):
        fes_before = alg.fes
        alg._restart_simplex()
        fes_after = alg.fes
        fes_history.append(fes_after)
        
        # Verify each restart increments by num_solutions
        increment = fes_after - fes_before
        assert increment == num_solutions, \
            f"Restart {i+1} should increment FES by {num_solutions}, got {increment}"
    
    # Verify total FES increase
    total_increase = fes_history[-1] - fes_history[0]
    expected_increase = num_restarts * num_solutions
    
    assert total_increase == expected_increase, \
        f"After {num_restarts} restarts, FES should increase by {expected_increase}, " \
        f"got {total_increase}"
    
    # Verify monotonicity across all restarts
    for i in range(1, len(fes_history)):
        assert fes_history[i] > fes_history[i-1], \
            f"FES should be monotonically increasing: {fes_history[i-1]} -> {fes_history[i]}"


@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=10000),
    restart_strategy=st.sampled_from(["uniform", "gaussian", "gaussian_best"]),
)
def test_property_17_restart_preserves_best_solution(dim, seed, restart_strategy):
    """
    Property 17: Restart increments function evaluations (Best solution preservation).
    
    Verify that restart preserves the best solution as the first vertex and 
    correctly increments FES while doing so.
    
    Validates: Requirements 5.4, 5.5, 13.5
    """
    from src.nelder_mead.algorithms.nelder_mead import NelderMead
    
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    num_solutions = dim + 1
    
    # Create NelderMead algorithm
    alg = NelderMead(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=1000,
        num_solutions=num_solutions,
        seed=seed,
        restart_strategy=restart_strategy,
    )
    
    # Initialize the simplex
    alg.simplex = alg.initialize_population()
    (
        alg.fitness_values,
        alg.eq_violations_list,
        alg.ineq_violations_list,
        alg.total_violations,
    ) = alg.evaluate_population(alg.simplex)
    alg._sort_simplex()
    
    # Record best solution before restart
    best_solution_before = alg.simplex[0].copy()
    best_fitness_before = alg.fitness_values[0]
    fes_before = alg.fes
    
    # Trigger restart
    alg._restart_simplex()
    
    # Verify FES increment
    fes_after = alg.fes
    assert fes_after - fes_before == num_solutions, \
        f"FES should increment by {num_solutions}, got {fes_after - fes_before}"
    
    # Verify best solution is preserved as first vertex
    # Note: After restart and re-evaluation, the simplex is sorted again,
    # so the best solution should still be at index 0
    best_solution_after = alg.simplex[0]
    
    # The best solution should be preserved (it's kept as first vertex before re-evaluation)
    # After sorting, it should still be the best or close to it
    # We verify that the best fitness didn't get worse
    best_fitness_after = alg.fitness_values[0]
    
    # Best fitness should not increase (may stay same or improve)
    assert best_fitness_after <= best_fitness_before + 1e-10, \
        f"Best fitness should not worsen after restart: {best_fitness_before} -> {best_fitness_after}"
    
    # Verify all vertices are within bounds
    assert np.all(alg.simplex >= lower_bounds), \
        "All vertices should be within lower bounds after restart"
    assert np.all(alg.simplex <= upper_bounds), \
        "All vertices should be within upper bounds after restart"


# ============================================================================
# Adaptive Nelder-Mead Property Tests (Property 18)
# ============================================================================


# Feature: nelder-mead-thesis-refactor, Property 18: Adaptive NM uses dimension-dependent coefficients
@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_18_adaptive_nm_dimension_dependent_coefficients(dim, seed):
    """
    Property 18: Adaptive NM uses dimension-dependent coefficients.
    
    For any dimension n, the Adaptive Nelder-Mead algorithm must set:
    - δₑ = 1 + 2/n (expansion coefficient)
    - δₒc = 0.75 - 1/(2n) (outside contraction coefficient)
    - δᵢc = -(0.75 - 1/(2n)) (inside contraction coefficient)
    - γₛ = 1 - 1/n (shrink coefficient)
    
    These dimension-dependent formulas were derived by Gao & Han (2012) to improve
    convergence properties across different problem dimensions.
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.4
    """
    from src.nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
    
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    
    # Create AdaptiveNelderMead algorithm
    alg = AdaptiveNelderMead(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=100,
        seed=seed,
    )
    
    # Compute expected coefficients based on dimension
    n = dim
    expected_delta_e = 1.0 + (2.0 / n)
    expected_delta_oc = 0.75 - (1.0 / (2.0 * n))
    expected_delta_ic = -(0.75 - (1.0 / (2.0 * n)))
    expected_gamma_s = 1.0 - (1.0 / n)
    
    # Verify expansion coefficient (Requirement 6.1)
    assert np.isclose(alg.delta_e, expected_delta_e, rtol=1e-10), \
        f"Expansion coefficient should be 1 + 2/{n} = {expected_delta_e}, " \
        f"got {alg.delta_e}"
    
    # Verify outside contraction coefficient (Requirement 6.2)
    assert np.isclose(alg.delta_oc, expected_delta_oc, rtol=1e-10), \
        f"Outside contraction coefficient should be 0.75 - 1/(2*{n}) = {expected_delta_oc}, " \
        f"got {alg.delta_oc}"
    
    # Verify inside contraction coefficient (Requirement 6.3)
    assert np.isclose(alg.delta_ic, expected_delta_ic, rtol=1e-10), \
        f"Inside contraction coefficient should be -(0.75 - 1/(2*{n})) = {expected_delta_ic}, " \
        f"got {alg.delta_ic}"
    
    # Verify shrink coefficient (Requirement 6.4)
    assert np.isclose(alg.gamma_s, expected_gamma_s, rtol=1e-10), \
        f"Shrink coefficient should be 1 - 1/{n} = {expected_gamma_s}, " \
        f"got {alg.gamma_s}"
    
    # Verify reflection coefficient remains standard (not dimension-dependent)
    assert np.isclose(alg.delta_r, 1.0, rtol=1e-10), \
        f"Reflection coefficient should remain 1.0, got {alg.delta_r}"


@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_18_adaptive_coefficients_converge_to_classical(dim, seed):
    """
    Property 18: Adaptive NM uses dimension-dependent coefficients (Convergence to classical).
    
    Verify that for n=2 (2D problems), the adaptive coefficients converge to
    the classical Nelder-Mead coefficients:
    - δₑ = 2.0
    - δₒc = 0.5
    - δᵢc = -0.5
    - γₛ = 0.5
    
    This ensures backward compatibility with the classical algorithm for 2D problems.
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.4
    """
    from src.nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
    
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    
    # Create AdaptiveNelderMead algorithm
    alg = AdaptiveNelderMead(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=100,
        seed=seed,
    )
    
    # For 2D problems, verify coefficients match classical values
    if dim == 2:
        # Classical Nelder-Mead coefficients
        classical_delta_e = 2.0
        classical_delta_oc = 0.5
        classical_delta_ic = -0.5
        classical_gamma_s = 0.5
        
        assert np.isclose(alg.delta_e, classical_delta_e, rtol=1e-10), \
            f"For n=2, expansion coefficient should be {classical_delta_e}, got {alg.delta_e}"
        
        assert np.isclose(alg.delta_oc, classical_delta_oc, rtol=1e-10), \
            f"For n=2, outside contraction coefficient should be {classical_delta_oc}, got {alg.delta_oc}"
        
        assert np.isclose(alg.delta_ic, classical_delta_ic, rtol=1e-10), \
            f"For n=2, inside contraction coefficient should be {classical_delta_ic}, got {alg.delta_ic}"
        
        assert np.isclose(alg.gamma_s, classical_gamma_s, rtol=1e-10), \
            f"For n=2, shrink coefficient should be {classical_gamma_s}, got {alg.gamma_s}"


@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=3, max_value=50),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_18_adaptive_coefficients_monotonicity(dim, seed):
    """
    Property 18: Adaptive NM uses dimension-dependent coefficients (Monotonicity).
    
    Verify the monotonicity properties of adaptive coefficients as dimension increases:
    - δₑ decreases with dimension (expansion becomes more conservative)
    - δₒc increases with dimension (contraction becomes more aggressive)
    - |δᵢc| increases with dimension (inside contraction becomes more aggressive)
    - γₛ increases with dimension (shrink becomes less aggressive)
    
    These trends ensure the algorithm adapts appropriately to high-dimensional problems.
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.4
    """
    from src.nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
    
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    # Create algorithms for current dimension and dimension-1
    lower_bounds_curr = np.full(dim, -10.0)
    upper_bounds_curr = np.full(dim, 10.0)
    
    lower_bounds_prev = np.full(dim - 1, -10.0)
    upper_bounds_prev = np.full(dim - 1, 10.0)
    
    alg_curr = AdaptiveNelderMead(
        objective_fn=sphere,
        lower_bounds=lower_bounds_curr,
        upper_bounds=upper_bounds_curr,
        max_fes=100,
        seed=seed,
    )
    
    alg_prev = AdaptiveNelderMead(
        objective_fn=sphere,
        lower_bounds=lower_bounds_prev,
        upper_bounds=upper_bounds_prev,
        max_fes=100,
        seed=seed,
    )
    
    # Verify monotonicity: δₑ decreases with dimension
    assert alg_curr.delta_e < alg_prev.delta_e, \
        f"Expansion coefficient should decrease with dimension: " \
        f"δₑ(n={dim-1})={alg_prev.delta_e} > δₑ(n={dim})={alg_curr.delta_e}"
    
    # Verify monotonicity: δₒc increases with dimension
    assert alg_curr.delta_oc > alg_prev.delta_oc, \
        f"Outside contraction coefficient should increase with dimension: " \
        f"δₒc(n={dim-1})={alg_prev.delta_oc} < δₒc(n={dim})={alg_curr.delta_oc}"
    
    # Verify monotonicity: |δᵢc| increases with dimension (δᵢc is negative)
    assert abs(alg_curr.delta_ic) > abs(alg_prev.delta_ic), \
        f"Inside contraction coefficient magnitude should increase with dimension: " \
        f"|δᵢc(n={dim-1})|={abs(alg_prev.delta_ic)} < |δᵢc(n={dim})|={abs(alg_curr.delta_ic)}"
    
    # Verify monotonicity: γₛ increases with dimension
    assert alg_curr.gamma_s > alg_prev.gamma_s, \
        f"Shrink coefficient should increase with dimension: " \
        f"γₛ(n={dim-1})={alg_prev.gamma_s} < γₛ(n={dim})={alg_curr.gamma_s}"


@settings(max_examples=100, deadline=None)
@given(
    dim=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_18_adaptive_coefficients_valid_ranges(dim, seed):
    """
    Property 18: Adaptive NM uses dimension-dependent coefficients (Valid ranges).
    
    Verify that all adaptive coefficients remain within valid ranges for all dimensions:
    - δₑ > 1.0 (expansion must extend beyond reflection)
    - 0 < δₒc < 1.0 (outside contraction must be between centroid and reflection)
    - -1.0 < δᵢc < 0 (inside contraction must be between centroid and worst)
    - 0 < γₛ < 1.0 (shrink must move toward best point)
    
    These constraints ensure the geometric operations remain valid.
    
    Validates: Requirements 6.1, 6.2, 6.3, 6.4
    """
    from src.nelder_mead.algorithms.adaptive_nm import AdaptiveNelderMead
    
    # Create simple sphere function
    def sphere(x):
        return np.sum(x**2)
    
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    
    # Create AdaptiveNelderMead algorithm
    alg = AdaptiveNelderMead(
        objective_fn=sphere,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        max_fes=100,
        seed=seed,
    )
    
    # Verify expansion coefficient is greater than 1.0
    assert alg.delta_e > 1.0, \
        f"Expansion coefficient must be > 1.0, got {alg.delta_e}"
    
    # Verify outside contraction coefficient is in (0, 1)
    assert 0.0 < alg.delta_oc < 1.0, \
        f"Outside contraction coefficient must be in (0, 1), got {alg.delta_oc}"
    
    # Verify inside contraction coefficient is in (-1, 0)
    assert -1.0 < alg.delta_ic < 0.0, \
        f"Inside contraction coefficient must be in (-1, 0), got {alg.delta_ic}"
    
    # Verify shrink coefficient is in (0, 1)
    assert 0.0 < alg.gamma_s < 1.0, \
        f"Shrink coefficient must be in (0, 1), got {alg.gamma_s}"
    
    # Verify reflection coefficient is exactly 1.0
    assert alg.delta_r == 1.0, \
        f"Reflection coefficient must be 1.0, got {alg.delta_r}"
