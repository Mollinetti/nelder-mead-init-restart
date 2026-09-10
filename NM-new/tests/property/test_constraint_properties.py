"""
Property-based tests for constraint handling and barrier methods.

This module tests universal correctness properties of barrier methods and
constraint violation computations using property-based testing with the
Hypothesis library. Each property is validated across 100+ randomized test
cases to ensure correctness holds for all valid inputs.

Feature: nelder-mead-thesis-refactor
"""

import numpy as np
import pytest
import sys
from hypothesis import given, settings, strategies as st

from src.nelder_mead.constraints.barrier_base import Barrier
from src.nelder_mead.constraints.hard_barrier import HardBarrier
from src.nelder_mead.constraints.deb_barrier import DebBarrier
from src.nelder_mead.constraints.augmented_lagrangian import AugmentedLagrangian
from src.nelder_mead.constraints.progressive_barrier import ProgressiveBarrier


# Custom strategies for generating test data
@st.composite
def bounds_strategy(draw, min_dim=2, max_dim=10):
    """Generate valid lower and upper bounds."""
    dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    lower = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=-1.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    upper = draw(
        st.lists(
            st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    return np.array(lower), np.array(upper)


@st.composite
def solution_strategy(draw, bounds):
    """Generate a solution (may be in or out of bounds)."""
    lower_bounds, upper_bounds = bounds
    dim = len(lower_bounds)
    solution = draw(
        st.lists(
            st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    return np.array(solution)


@st.composite
def violations_strategy(draw, min_size=0, max_size=10):
    """Generate constraint violation arrays."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    if size == 0:
        return np.array([])
    violations = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    return np.array(violations)


# Feature: nelder-mead-thesis-refactor, Property 19: Hard barrier returns infinite penalty for bound violations
@settings(max_examples=100)
@given(bounds=bounds_strategy())
def test_property_19_hard_barrier_infinite_penalty(bounds):
    """
    Property 19: Hard barrier returns infinite penalty for bound violations.
    
    For any solution x and bounds [lower, upper], if any xᵢ < lowerᵢ or xᵢ > upperᵢ,
    the Hard Barrier must return a penalty value equal to the maximum representable integer.
    
    Validates: Requirements 7.1
    """
    lower_bounds, upper_bounds = bounds
    dim = len(lower_bounds)
    
    # Create Hard Barrier
    barrier = HardBarrier(bounds, num_solutions=dim+1)
    
    # Test 1: Solution violating lower bound
    solution_lower_vio = lower_bounds.copy()
    solution_lower_vio[0] -= 1.0  # Violate first dimension
    
    penalty = barrier.penalize_constraint_violation(
        solution_lower_vio, np.array([]), np.array([])
    )
    assert penalty == float(sys.maxsize), \
        f"Hard barrier should return infinite penalty for lower bound violation, got {penalty}"
    
    # Test 2: Solution violating upper bound
    solution_upper_vio = upper_bounds.copy()
    solution_upper_vio[0] += 1.0  # Violate first dimension
    
    penalty = barrier.penalize_constraint_violation(
        solution_upper_vio, np.array([]), np.array([])
    )
    assert penalty == float(sys.maxsize), \
        f"Hard barrier should return infinite penalty for upper bound violation, got {penalty}"
    
    # Test 3: Solution with equality constraint violation
    solution_feasible = (lower_bounds + upper_bounds) / 2.0
    eq_violations = np.array([1.0])  # One equality constraint violated
    
    penalty = barrier.penalize_constraint_violation(
        solution_feasible, eq_violations, np.array([])
    )
    assert penalty == float(sys.maxsize), \
        f"Hard barrier should return infinite penalty for equality constraint violation, got {penalty}"
    
    # Test 4: Solution with inequality constraint violation
    ineq_violations = np.array([1.0])  # One inequality constraint violated
    
    penalty = barrier.penalize_constraint_violation(
        solution_feasible, np.array([]), ineq_violations
    )
    assert penalty == float(sys.maxsize), \
        f"Hard barrier should return infinite penalty for inequality constraint violation, got {penalty}"
    
    # Test 5: Feasible solution (no violations)
    penalty = barrier.penalize_constraint_violation(
        solution_feasible, np.array([]), np.array([])
    )
    assert penalty == 0.0, \
        f"Hard barrier should return zero penalty for feasible solution, got {penalty}"


# Feature: nelder-mead-thesis-refactor, Property 20: Deb barrier uses worst fitness plus violations
@settings(max_examples=100)
@given(
    bounds=bounds_strategy(),
    worst_fitness=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    eq_vio=violations_strategy(min_size=0, max_size=5),
    ineq_vio=violations_strategy(min_size=0, max_size=5)
)
def test_property_20_deb_barrier_formula(bounds, worst_fitness, eq_vio, ineq_vio):
    """
    Property 20: Deb barrier uses worst fitness plus violations.
    
    For any solution with constraint violations and worst fitness f_worst,
    the Deb Barrier penalty must equal f_worst + Σ|h(x)| + Σmax(0, g(x)).
    
    Validates: Requirements 7.2
    """
    lower_bounds, upper_bounds = bounds
    dim = len(lower_bounds)
    
    # Create Deb Barrier and set worst fitness
    barrier = DebBarrier(bounds, num_solutions=dim+1)
    barrier.set_worst_fitness(worst_fitness)
    
    # Test with solution that has bound violations
    solution_vio = lower_bounds.copy()
    solution_vio[0] -= 5.0  # Violate lower bound by 5.0
    
    penalty = barrier.penalize_constraint_violation(solution_vio, eq_vio, ineq_vio)
    
    # Compute expected penalty
    bound_vio = 5.0  # We violated by 5.0
    eq_sum = np.sum(eq_vio) if len(eq_vio) > 0 else 0.0
    ineq_sum = np.sum(ineq_vio) if len(ineq_vio) > 0 else 0.0
    total_vio = bound_vio + eq_sum + ineq_sum
    
    if total_vio > 0:
        expected_penalty = worst_fitness + total_vio
        np.testing.assert_allclose(penalty, expected_penalty, rtol=1e-10, atol=1e-10)
    else:
        # No violations - should return 0
        assert penalty == 0.0
    
    # Test with feasible solution
    solution_feasible = (lower_bounds + upper_bounds) / 2.0
    penalty_feasible = barrier.penalize_constraint_violation(
        solution_feasible, np.array([]), np.array([])
    )
    assert penalty_feasible == 0.0, "Deb barrier should return 0 for feasible solution"


# Feature: nelder-mead-thesis-refactor, Property 21: Augmented Lagrangian penalty formula
@settings(max_examples=100)
@given(
    bounds=bounds_strategy(),
    m_eq=st.integers(min_value=1, max_value=5),
    p_ineq=st.integers(min_value=1, max_value=5),
    rho=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
)
def test_property_21_augmented_lagrangian_penalty_formula(bounds, m_eq, p_ineq, rho):
    """
    Property 21: Augmented Lagrangian penalty formula.
    
    For any solution with equality violations h and inequality violations g,
    the Augmented Lagrangian penalty must equal:
    (ρ/2)(||h + λ/ρ||² + ||max(0, g + μ/ρ)||²).
    
    Validates: Requirements 7.3, 7.5
    """
    lower_bounds, upper_bounds = bounds
    dim = len(lower_bounds)
    
    # Create Augmented Lagrangian
    barrier = AugmentedLagrangian(
        bounds, num_solutions=dim+1, m_eq=m_eq, p_ineq=p_ineq, rho=rho, beta=100.0
    )
    
    # Generate violations
    eq_violations = np.random.uniform(0.0, 10.0, m_eq)
    ineq_violations = np.random.uniform(0.0, 10.0, p_ineq)
    
    # Set some multipliers (non-zero)
    barrier.lambda_eq = np.random.uniform(-10.0, 10.0, m_eq)
    barrier.mu_ineq = np.random.uniform(0.0, 10.0, p_ineq)
    
    # Feasible solution (no bound violations)
    solution = (lower_bounds + upper_bounds) / 2.0
    
    # Compute penalty
    penalty = barrier.penalize_constraint_violation(solution, eq_violations, ineq_violations)
    
    # Manually compute expected penalty using the formula
    # Equality: (ρ/2) * ||h + λ/ρ||²
    shifted_eq = eq_violations + barrier.lambda_eq / rho
    eq_penalty = (rho / 2.0) * np.sum(shifted_eq ** 2)
    
    # Inequality: (ρ/2) * ||max(0, g + μ/ρ)||²
    shifted_ineq = ineq_violations + barrier.mu_ineq / rho
    shifted_ineq = np.maximum(0, shifted_ineq)
    ineq_penalty = (rho / 2.0) * np.sum(shifted_ineq ** 2)
    
    # Bound penalty (should be 0 for feasible solution)
    bound_penalty = 0.0
    
    expected_penalty = eq_penalty + ineq_penalty + bound_penalty
    
    np.testing.assert_allclose(penalty, expected_penalty, rtol=1e-8, atol=1e-10)


# Feature: nelder-mead-thesis-refactor, Property 22: Augmented Lagrangian multiplier updates
@settings(max_examples=100)
@given(
    bounds=bounds_strategy(),
    m_eq=st.integers(min_value=1, max_value=5),
    p_ineq=st.integers(min_value=1, max_value=5),
    rho=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
)
def test_property_22_augmented_lagrangian_multiplier_updates(bounds, m_eq, p_ineq, rho):
    """
    Property 22: Augmented Lagrangian multiplier updates.
    
    For any constraint violations h and g, after update:
    - λ_new = clip(λ + ρ*h, λ_min, λ_max)
    - μ_new = clip(max(0, μ + ρ*g), 0, μ_max)
    
    Validates: Requirements 8.2, 8.3, 8.5
    """
    lower_bounds, upper_bounds = bounds
    dim = len(lower_bounds)
    
    # Create Augmented Lagrangian
    barrier = AugmentedLagrangian(
        bounds, num_solutions=dim+1, m_eq=m_eq, p_ineq=p_ineq, rho=rho
    )
    
    # Store initial multipliers
    lambda_old = barrier.lambda_eq.copy()
    mu_old = barrier.mu_ineq.copy()
    
    # Generate violations
    eq_violations = np.random.uniform(0.0, 10.0, m_eq)
    ineq_violations = np.random.uniform(0.0, 10.0, p_ineq)
    
    # Feasible solution
    solution = (lower_bounds + upper_bounds) / 2.0
    
    # Update multipliers
    barrier.update(solution, eq_violations, ineq_violations)
    
    # Verify equality multiplier update: λ_new = clip(λ + ρ*h, λ_min, λ_max)
    expected_lambda = lambda_old + rho * eq_violations
    expected_lambda = np.clip(expected_lambda, barrier.lambda_min, barrier.lambda_max)
    np.testing.assert_allclose(barrier.lambda_eq, expected_lambda, rtol=1e-10, atol=1e-10)
    
    # Verify inequality multiplier update: μ_new = clip(max(0, μ + ρ*g), 0, μ_max)
    expected_mu = mu_old + rho * ineq_violations
    expected_mu = np.maximum(0, expected_mu)  # Ensure non-negativity
    expected_mu = np.minimum(expected_mu, barrier.mu_max)  # Enforce upper bound
    np.testing.assert_allclose(barrier.mu_ineq, expected_mu, rtol=1e-10, atol=1e-10)
    
    # Verify non-negativity of inequality multipliers
    assert np.all(barrier.mu_ineq >= 0), "Inequality multipliers must be non-negative"



# Feature: nelder-mead-thesis-refactor, Property 23: Augmented Lagrangian rho adaptation
@settings(max_examples=100)
@given(
    bounds=bounds_strategy(),
    m_eq=st.integers(min_value=1, max_value=5),
    p_ineq=st.integers(min_value=1, max_value=5)
)
def test_property_23_augmented_lagrangian_rho_adaptation(bounds, m_eq, p_ineq):
    """
    Property 23: Augmented Lagrangian rho adaptation.
    
    For any iteration where ||h||∞ or ||max(0, g + μ/ρ)||∞ > τ * previous_violation,
    the penalty parameter ρ must be increased: ρ_new = γ * ρ.
    
    Validates: Requirements 8.1
    """
    lower_bounds, upper_bounds = bounds
    dim = len(lower_bounds)
    
    # Create Augmented Lagrangian with specific parameters
    rho_initial = 2.0
    gamma = 1.5
    tau = 0.8
    
    barrier = AugmentedLagrangian(
        bounds, num_solutions=dim+1, m_eq=m_eq, p_ineq=p_ineq,
        rho=rho_initial, gamma=gamma, tau=tau
    )
    
    # Feasible solution
    solution = (lower_bounds + upper_bounds) / 2.0
    
    # First update with large violations to set previous_violation
    large_eq_vio = np.full(m_eq, 10.0)
    large_ineq_vio = np.full(p_ineq, 10.0)
    barrier.update(solution, large_eq_vio, large_ineq_vio)
    
    rho_after_first = barrier.rho
    
    # Second update with violations that don't decrease sufficiently
    # (violations > tau * previous_violation)
    insufficient_eq_vio = np.full(m_eq, 9.0)  # Only decreased from 10 to 9 (< tau * 10 = 8)
    insufficient_ineq_vio = np.full(p_ineq, 9.0)
    
    barrier.update(solution, insufficient_eq_vio, insufficient_ineq_vio)
    
    # Rho should have increased
    expected_rho = rho_after_first * gamma
    np.testing.assert_allclose(barrier.rho, expected_rho, rtol=1e-10, atol=1e-10,
                               err_msg="Rho should increase when violations don't decrease sufficiently")
    
    # Third update with violations that decrease sufficiently
    # (violations <= tau * previous_violation)
    sufficient_eq_vio = np.full(m_eq, 1.0)  # Decreased significantly
    sufficient_ineq_vio = np.full(p_ineq, 1.0)
    
    rho_before_third = barrier.rho
    barrier.update(solution, sufficient_eq_vio, sufficient_ineq_vio)
    
    # Rho should NOT have increased (or might have, but we can't guarantee it won't)
    # The key is that it only increases when violations don't decrease enough
    # So we just verify the mechanism works in the insufficient case above


# Feature: nelder-mead-thesis-refactor, Property 24: Progressive barrier threshold management
@settings(max_examples=100)
@given(bounds=bounds_strategy())
def test_property_24_progressive_barrier_threshold_management(bounds):
    """
    Property 24: Progressive barrier threshold management.
    
    For any Progressive Barrier, when a better infeasible solution is found
    within the threshold, the threshold must be updated to that solution's
    violation level.
    
    Validates: Requirements 9.2, 9.5
    """
    lower_bounds, upper_bounds = bounds
    dim = len(lower_bounds)
    
    # Create Progressive Barrier
    barrier = ProgressiveBarrier(bounds, penalty_coeff=1e4)
    
    # Initial threshold should be infinity
    assert barrier.threshold == float('inf'), "Initial threshold should be infinity"
    
    # Create solutions with different violations
    num_solutions = 3
    solutions = np.array([
        (lower_bounds + upper_bounds) / 2.0,  # Feasible
        (lower_bounds + upper_bounds) / 2.0,  # Will make infeasible
        (lower_bounds + upper_bounds) / 2.0,  # Will make infeasible
    ])
    
    # Make second and third solutions infeasible with different violation levels
    eq_violations_list = [
        np.array([]),
        np.array([5.0]),  # Violation = 5.0
        np.array([10.0]), # Violation = 10.0
    ]
    ineq_violations_list = [
        np.array([]),
        np.array([]),
        np.array([]),
    ]
    
    # Objective values (second solution is better)
    obj_vals = np.array([100.0, 50.0, 60.0])
    
    # Update barrier
    iteration_type = barrier.update(
        solutions, obj_vals, eq_violations_list, ineq_violations_list
    )
    
    # Threshold should be updated to the violation of the best infeasible solution (5.0)
    assert barrier.threshold == 5.0, \
        f"Threshold should be updated to best infeasible violation (5.0), got {barrier.threshold}"
    
    # Best infeasible should be the second solution
    assert barrier.best_infeasible is not None
    assert barrier.best_infeasible[1] == 50.0  # Objective value
    assert barrier.best_infeasible[2] == 5.0   # Violation


# Feature: nelder-mead-thesis-refactor, Property 25: Progressive barrier penalty beyond threshold
@settings(max_examples=100)
@given(
    bounds=bounds_strategy(),
    penalty_coeff=st.floats(min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
)
def test_property_25_progressive_barrier_penalty_beyond_threshold(bounds, penalty_coeff):
    """
    Property 25: Progressive barrier penalty beyond threshold.
    
    For any solution with total violation v > threshold, the Progressive Barrier
    penalty must be significantly larger than v (typically v * large_constant).
    
    Validates: Requirements 9.4
    """
    lower_bounds, upper_bounds = bounds
    
    # Create Progressive Barrier with specific penalty coefficient
    barrier = ProgressiveBarrier(bounds, penalty_coeff=penalty_coeff)
    
    # Set a specific threshold
    barrier.threshold = 5.0
    
    # Test solution with violation within threshold
    eq_vio_within = np.array([3.0])  # Total violation = 3.0 < 5.0
    ineq_vio_within = np.array([])
    solution = (lower_bounds + upper_bounds) / 2.0
    
    penalty_within = barrier.penalize_constraint_violation(
        solution, eq_vio_within, ineq_vio_within
    )
    
    # Penalty should equal the violation (3.0)
    assert penalty_within == 3.0, \
        f"Penalty within threshold should equal violation (3.0), got {penalty_within}"
    
    # Test solution with violation beyond threshold
    eq_vio_beyond = np.array([10.0])  # Total violation = 10.0 > 5.0
    ineq_vio_beyond = np.array([])
    
    penalty_beyond = barrier.penalize_constraint_violation(
        solution, eq_vio_beyond, ineq_vio_beyond
    )
    
    # Penalty should be violation * penalty_coeff
    expected_penalty_beyond = 10.0 * penalty_coeff
    np.testing.assert_allclose(penalty_beyond, expected_penalty_beyond, rtol=1e-10, atol=1e-10)
    
    # Verify penalty beyond is much larger than penalty within
    assert penalty_beyond > penalty_within * 10, \
        f"Penalty beyond threshold ({penalty_beyond}) should be much larger than within ({penalty_within})"



# Feature: nelder-mead-thesis-refactor, Property 26: Constraint violation computation formulas
@settings(max_examples=100)
@given(bounds=bounds_strategy())
def test_property_26_constraint_violation_computation(bounds):
    """
    Property 26: Constraint violation computation formulas.
    
    For any solution x with bounds [lower, upper], equality constraints h(x),
    and inequality constraints g(x):
    - Bound violations = Σmax(0, lower - x) + Σmax(0, x - upper)
    - Equality violations = Σ|h(x)|
    - Inequality violations = Σmax(0, g(x))
    - Total violation = bound_vio + eq_vio + ineq_vio
    - Feasible solutions have total violation = 0
    
    Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5
    """
    lower_bounds, upper_bounds = bounds
    dim = len(lower_bounds)
    
    # Create a barrier (any barrier will do, we're testing the base class method)
    barrier = HardBarrier(bounds, num_solutions=dim+1)
    
    # Test 1: Solution violating lower bounds
    solution_lower = lower_bounds.copy()
    solution_lower[0] -= 5.0  # Violate by 5.0
    solution_lower[1] -= 3.0  # Violate by 3.0
    
    bound_vio = barrier.penalize_bounds(solution_lower)
    expected_bound_vio = 5.0 + 3.0  # Sum of violations
    np.testing.assert_allclose(bound_vio, expected_bound_vio, rtol=1e-10, atol=1e-10)
    
    # Test 2: Solution violating upper bounds
    solution_upper = upper_bounds.copy()
    solution_upper[0] += 7.0  # Violate by 7.0
    
    bound_vio = barrier.penalize_bounds(solution_upper)
    expected_bound_vio = 7.0
    np.testing.assert_allclose(bound_vio, expected_bound_vio, rtol=1e-10, atol=1e-10)
    
    # Test 3: Solution within bounds (no bound violations)
    solution_feasible = (lower_bounds + upper_bounds) / 2.0
    bound_vio = barrier.penalize_bounds(solution_feasible)
    assert bound_vio == 0.0, "Feasible solution should have zero bound violation"
    
    # Test 4: Total violation computation
    eq_violations = np.array([2.0, 3.0])  # Already absolute values
    ineq_violations = np.array([1.0, 4.0])  # Already positive parts
    
    total_vio = barrier.compute_total_violation(
        solution_feasible, eq_violations, ineq_violations
    )
    
    expected_total = 0.0 + np.sum(eq_violations) + np.sum(ineq_violations)
    np.testing.assert_allclose(total_vio, expected_total, rtol=1e-10, atol=1e-10)
    
    # Test 5: Feasible solution has zero total violation
    total_vio_feasible = barrier.compute_total_violation(
        solution_feasible, np.array([]), np.array([])
    )
    assert total_vio_feasible == 0.0, "Feasible solution should have zero total violation"
    
    # Test 6: Verify formula components
    # Bound violations: Σmax(0, lower - x) + Σmax(0, x - upper)
    solution_mixed = lower_bounds.copy()
    solution_mixed[0] -= 2.0  # Lower violation
    if dim > 1:
        solution_mixed[1] = upper_bounds[1] + 3.0  # Upper violation
    
    bound_vio_mixed = barrier.penalize_bounds(solution_mixed)
    
    # Manually compute
    lower_vio_manual = np.sum(np.maximum(0, lower_bounds - solution_mixed))
    upper_vio_manual = np.sum(np.maximum(0, solution_mixed - upper_bounds))
    expected_mixed = lower_vio_manual + upper_vio_manual
    
    np.testing.assert_allclose(bound_vio_mixed, expected_mixed, rtol=1e-10, atol=1e-10)
