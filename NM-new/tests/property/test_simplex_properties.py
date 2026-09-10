"""
Property-based tests for simplex operations.

This module tests universal correctness properties of simplex geometric operations
using property-based testing with the Hypothesis library. Each property is validated
across 100+ randomized test cases to ensure correctness holds for all valid inputs.

Feature: nelder-mead-thesis-refactor
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.nelder_mead.core.simplex_operations import (
    compute_centroid,
    reflect,
    expand,
    contract_outside,
    contract_inside,
    shrink,
    enforce_bounds,
)


# Custom strategies for generating test data
@st.composite
def simplex_strategy(draw, min_dim=2, max_dim=10):
    """Generate a valid simplex with n+1 vertices in n dimensions."""
    dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    # Generate n+1 vertices for n-dimensional space
    simplex = draw(
        st.lists(
            st.lists(
                st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                min_size=dim,
                max_size=dim,
            ),
            min_size=dim + 1,
            max_size=dim + 1,
        )
    )
    return np.array(simplex)


@st.composite
def point_strategy(draw, min_dim=2, max_dim=10):
    """Generate a random point in n dimensions."""
    dim = draw(st.integers(min_value=min_dim, max_value=max_dim))
    point = draw(
        st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    return np.array(point)


@st.composite
def coefficient_strategy(draw):
    """Generate valid coefficients for simplex operations."""
    return draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))


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
    return np.array(lower), np.array(upper)


# Feature: nelder-mead-thesis-refactor, Property 1: Simplex operations produce geometrically correct transformations
@settings(max_examples=100)
@given(simplex=simplex_strategy())
def test_property_1_simplex_operations_correct_transformations(simplex):
    """
    Property 1: Simplex operations produce geometrically correct transformations.
    
    For any simplex with centroid c, worst point w, best point b, and reflection point r,
    the following must hold:
    - Reflection: r = c + δᵣ(c - w)
    - Expansion: e = c + δₑ(r - c)
    - Outside contraction: oc = c + δₒc(r - c)
    - Inside contraction: ic = c + δᵢc(w - c)
    - Shrink: for all vertices vᵢ, vᵢ' = b + γₛ(vᵢ - b)
    
    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
    """
    # Standard coefficients from Nelder & Mead (1965)
    delta_r = 1.0
    delta_e = 2.0
    delta_oc = 0.5
    delta_ic = -0.5
    gamma_s = 0.5
    
    # Get centroid, best, and worst points
    centroid = compute_centroid(simplex, exclude_worst=True)
    best_point = simplex[0]
    worst_point = simplex[-1]
    
    # Test reflection formula: r = c + δᵣ(c - w)
    reflected = reflect(centroid, worst_point, delta_r)
    expected_reflected = centroid + delta_r * (centroid - worst_point)
    np.testing.assert_allclose(reflected, expected_reflected, rtol=1e-10, atol=1e-10)
    
    # Test expansion formula: e = c + δₑ(r - c)
    expanded = expand(centroid, reflected, delta_e)
    expected_expanded = centroid + delta_e * (reflected - centroid)
    np.testing.assert_allclose(expanded, expected_expanded, rtol=1e-10, atol=1e-10)
    
    # Test outside contraction formula: oc = c + δₒc(r - c)
    outside_contracted = contract_outside(centroid, reflected, delta_oc)
    expected_outside = centroid + delta_oc * (reflected - centroid)
    np.testing.assert_allclose(outside_contracted, expected_outside, rtol=1e-10, atol=1e-10)
    
    # Test inside contraction formula: ic = c + δᵢc(c - w)
    inside_contracted = contract_inside(centroid, worst_point, delta_ic)
    expected_inside = centroid + delta_ic * (centroid - worst_point)
    np.testing.assert_allclose(inside_contracted, expected_inside, rtol=1e-10, atol=1e-10)
    
    # Test shrink formula: vᵢ' = b + γₛ(vᵢ - b) for all vertices
    shrunk = shrink(best_point, simplex, gamma_s)
    for i in range(len(simplex)):
        expected_vertex = best_point + gamma_s * (simplex[i] - best_point)
        np.testing.assert_allclose(shrunk[i], expected_vertex, rtol=1e-10, atol=1e-10)


# Feature: nelder-mead-thesis-refactor, Property 2: Centroid computation excludes worst point
@settings(max_examples=100)
@given(simplex=simplex_strategy())
def test_property_2_centroid_excludes_worst(simplex):
    """
    Property 2: Centroid computation excludes worst point.
    
    For any simplex with n+1 vertices, the centroid must equal the mean of the
    first n vertices (excluding the worst vertex after sorting by fitness).
    
    Validates: Requirements 2.6
    """
    # Compute centroid excluding worst (last vertex)
    centroid = compute_centroid(simplex, exclude_worst=True)
    
    # Manually compute mean of first n vertices
    expected_centroid = np.mean(simplex[:-1], axis=0)
    
    np.testing.assert_allclose(centroid, expected_centroid, rtol=1e-10, atol=1e-10)
    
    # Also verify that when exclude_worst=False, we get the mean of all vertices
    centroid_all = compute_centroid(simplex, exclude_worst=False)
    expected_all = np.mean(simplex, axis=0)
    np.testing.assert_allclose(centroid_all, expected_all, rtol=1e-10, atol=1e-10)


# Feature: nelder-mead-thesis-refactor, Property 3: Reflection places point on opposite side of centroid
@settings(max_examples=100)
@given(simplex=simplex_strategy(), delta_r=coefficient_strategy())
def test_property_3_reflection_opposite_side(simplex, delta_r):
    """
    Property 3: Reflection places point on opposite side of centroid.
    
    For any simplex, the reflected point r must satisfy: (r - c) · (w - c) < 0,
    where c is the centroid and w is the worst point (i.e., reflection and worst
    point are on opposite sides of centroid).
    
    Validates: Requirements 11.1
    """
    centroid = compute_centroid(simplex, exclude_worst=True)
    worst_point = simplex[-1]
    
    # Compute reflected point
    reflected = reflect(centroid, worst_point, delta_r)
    
    # Vectors from centroid to reflected and worst points
    vec_to_reflected = reflected - centroid
    vec_to_worst = worst_point - centroid
    
    # Dot product should be negative (opposite directions)
    # Allow for numerical tolerance and handle zero vectors
    dot_product = np.dot(vec_to_reflected, vec_to_worst)
    
    # If worst point is at centroid (degenerate case), skip this check
    if np.linalg.norm(vec_to_worst) > 1e-10:
        assert dot_product <= 1e-8, f"Dot product {dot_product} should be negative or near zero"


# Feature: nelder-mead-thesis-refactor, Property 4: Expansion extends beyond reflection
@settings(max_examples=100)
@given(simplex=simplex_strategy(), delta_e=coefficient_strategy())
def test_property_4_expansion_extends_beyond_reflection(simplex, delta_e):
    """
    Property 4: Expansion extends beyond reflection.
    
    For any simplex with centroid c, reflected point r, and expanded point e,
    the distance from expansion to centroid must exceed the distance from
    reflection to centroid: ||e - c|| > ||r - c||.
    
    Validates: Requirements 11.2
    """
    # Ensure delta_e > 1 for expansion to extend beyond reflection
    if delta_e <= 1.0:
        delta_e = 2.0
    
    centroid = compute_centroid(simplex, exclude_worst=True)
    worst_point = simplex[-1]
    
    # Compute reflected and expanded points
    reflected = reflect(centroid, worst_point, delta_r=1.0)
    expanded = expand(centroid, reflected, delta_e)
    
    # Compute distances from centroid
    dist_reflected = np.linalg.norm(reflected - centroid)
    dist_expanded = np.linalg.norm(expanded - centroid)
    
    # Expansion should be farther from centroid than reflection
    # Allow small tolerance for numerical errors
    assert dist_expanded >= dist_reflected * (delta_e - 0.01), \
        f"Expansion distance {dist_expanded} should exceed reflection distance {dist_reflected}"


# Feature: nelder-mead-thesis-refactor, Property 5: Contraction moves toward centroid
@settings(max_examples=100)
@given(simplex=simplex_strategy())
def test_property_5_contraction_moves_toward_centroid(simplex):
    """
    Property 5: Contraction moves toward centroid.
    
    For any contraction operation (inside or outside), the contracted point must
    be closer to the centroid than the original point: ||contracted - c|| < ||original - c||.
    
    Validates: Requirements 11.3
    """
    centroid = compute_centroid(simplex, exclude_worst=True)
    worst_point = simplex[-1]
    reflected = reflect(centroid, worst_point, delta_r=1.0)
    
    # Test outside contraction (from reflected point)
    delta_oc = 0.5
    outside_contracted = contract_outside(centroid, reflected, delta_oc)
    
    dist_reflected = np.linalg.norm(reflected - centroid)
    dist_outside_contracted = np.linalg.norm(outside_contracted - centroid)
    
    assert dist_outside_contracted < dist_reflected + 1e-8, \
        f"Outside contraction distance {dist_outside_contracted} should be less than reflection distance {dist_reflected}"
    
    # Test inside contraction (from worst point)
    delta_ic = -0.5
    inside_contracted = contract_inside(centroid, worst_point, delta_ic)
    
    dist_worst = np.linalg.norm(worst_point - centroid)
    dist_inside_contracted = np.linalg.norm(inside_contracted - centroid)
    
    assert dist_inside_contracted < dist_worst + 1e-8, \
        f"Inside contraction distance {dist_inside_contracted} should be less than worst point distance {dist_worst}"


# Feature: nelder-mead-thesis-refactor, Property 6: Shrink moves all points toward best
@settings(max_examples=100)
@given(simplex=simplex_strategy(), gamma_s=st.floats(min_value=0.1, max_value=0.9))
def test_property_6_shrink_moves_toward_best(simplex, gamma_s):
    """
    Property 6: Shrink moves all points toward best.
    
    For any simplex after shrink operation, all vertices except the best must be
    closer to the best vertex: ||vᵢ' - v₀|| < ||vᵢ - v₀|| for all i > 0.
    
    Validates: Requirements 11.4
    """
    best_point = simplex[0]
    
    # Perform shrink operation
    shrunk = shrink(best_point, simplex, gamma_s)
    
    # Check that best point remains unchanged
    np.testing.assert_allclose(shrunk[0], best_point, rtol=1e-10, atol=1e-10)
    
    # Check that all other points moved closer to best
    for i in range(1, len(simplex)):
        dist_before = np.linalg.norm(simplex[i] - best_point)
        dist_after = np.linalg.norm(shrunk[i] - best_point)
        
        # After shrink, distance should be gamma_s times the original distance
        expected_dist = gamma_s * dist_before
        
        # Allow small numerical tolerance
        np.testing.assert_allclose(dist_after, expected_dist, rtol=1e-8, atol=1e-10)
        
        # Verify it's actually closer (unless original point was at best point)
        if dist_before > 1e-10:
            assert dist_after < dist_before + 1e-8, \
                f"Point {i} distance after shrink {dist_after} should be less than before {dist_before}"



# Import stopping criteria functions
from src.nelder_mead.core.stopping_criteria import (
    check_oriented_length,
    check_std_dev,
    check_small_simplex,
    check_flat_simplex,
    check_degenerate_simplex,
)


# Feature: nelder-mead-thesis-refactor, Property 11: Oriented length criterion uses correct formula
@settings(max_examples=100)
@given(simplex=simplex_strategy())
def test_property_11_oriented_length_formula(simplex):
    """
    Property 11: Oriented length criterion uses correct formula.
    
    For any simplex, the oriented length criterion must compute:
    (1/max(1, ||v₀||)) * max(||vᵢ - v₀||) for i = 1 to n, where v₀ is the first vertex.
    
    Validates: Requirements 4.1
    """
    v0 = simplex[0]
    v0_norm = np.linalg.norm(v0)
    
    # Manually compute the oriented length
    edge_lengths = np.array([np.linalg.norm(simplex[i] - v0) for i in range(1, len(simplex))])
    max_edge = np.max(edge_lengths)
    expected_oriented_length = max_edge / max(1.0, v0_norm)
    
    # Skip degenerate case where all vertices are identical (max_edge = 0)
    if max_edge < 1e-15:
        # For degenerate simplex, criterion should always return True
        result = check_oriented_length(simplex, epsilon=1e-10)
        assert result, "Degenerate simplex (all vertices identical) should satisfy criterion"
        return
    
    # Test with various epsilon values
    # If expected_oriented_length < epsilon, criterion should return True
    epsilon_small = expected_oriented_length / 2.0  # Should return False
    epsilon_large = expected_oriented_length * 2.0  # Should return True
    
    result_small = check_oriented_length(simplex, epsilon=epsilon_small)
    result_large = check_oriented_length(simplex, epsilon=epsilon_large)
    
    # Verify the criterion behaves correctly
    assert not result_small, f"Criterion should return False when epsilon ({epsilon_small}) < oriented_length ({expected_oriented_length})"
    assert result_large, f"Criterion should return True when epsilon ({epsilon_large}) > oriented_length ({expected_oriented_length})"
    
    # Test exact threshold
    result_exact = check_oriented_length(simplex, epsilon=expected_oriented_length)
    assert result_exact, f"Criterion should return True when epsilon equals oriented_length"


# Feature: nelder-mead-thesis-refactor, Property 12: Standard deviation criterion uses fitness variance
@settings(max_examples=100)
@given(
    fitness_values=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=20
    )
)
def test_property_12_std_dev_uses_variance(fitness_values):
    """
    Property 12: Standard deviation criterion uses fitness variance.
    
    For any simplex with fitness values f, the standard deviation criterion must compute:
    (1/(n+1)) * Σ(fᵢ - f̄)² where f̄ is the mean fitness.
    
    Validates: Requirements 4.2
    """
    fitness_array = np.array(fitness_values)
    n = len(fitness_array)
    f_mean = np.mean(fitness_array)
    
    # Manually compute variance: (1/n) * Σ(fᵢ - f̄)²
    expected_variance = np.sum((fitness_array - f_mean) ** 2) / n
    
    # Test with various epsilon values
    epsilon_small = expected_variance / 2.0 if expected_variance > 0 else 1e-10
    epsilon_large = expected_variance * 2.0 + 1e-10
    
    result_small = check_std_dev(fitness_array, epsilon=epsilon_small)
    result_large = check_std_dev(fitness_array, epsilon=epsilon_large)
    
    # Verify the criterion behaves correctly
    if expected_variance > 1e-15:  # Only test when variance is non-negligible
        assert not result_small, f"Criterion should return False when epsilon ({epsilon_small}) < variance ({expected_variance})"
    assert result_large, f"Criterion should return True when epsilon ({epsilon_large}) > variance ({expected_variance})"


# Feature: nelder-mead-thesis-refactor, Property 13: Small simplex criterion normalizes by search space
@settings(max_examples=100)
@given(simplex=simplex_strategy(), bounds=bounds_strategy())
def test_property_13_small_simplex_normalizes_by_bounds(simplex, bounds):
    """
    Property 13: Small simplex criterion normalizes by search space.
    
    For any simplex and bounds, the small simplex criterion must compute the maximum
    edge length normalized by the corresponding bound range for each dimension.
    
    Validates: Requirements 4.3
    """
    lower_bounds, upper_bounds = bounds
    dim = simplex.shape[1]
    
    # Ensure bounds match simplex dimension
    if len(lower_bounds) != dim:
        lower_bounds = np.full(dim, -100.0)
        upper_bounds = np.full(dim, 100.0)
    
    # Manually compute maximum normalized edge length
    max_normalized_edge = 0.0
    for d in range(dim):
        coords = simplex[:, d]
        coord_range = np.max(coords) - np.min(coords)
        bound_range = upper_bounds[d] - lower_bounds[d]
        if bound_range > 0:
            normalized = coord_range / bound_range
            max_normalized_edge = max(max_normalized_edge, normalized)
    
    # Test with various epsilon values
    epsilon_small = max_normalized_edge / 2.0 if max_normalized_edge > 0 else 1e-10
    epsilon_large = max_normalized_edge * 2.0 + 1e-10
    
    result_small = check_small_simplex(simplex, (lower_bounds, upper_bounds), epsilon=epsilon_small)
    result_large = check_small_simplex(simplex, (lower_bounds, upper_bounds), epsilon=epsilon_large)
    
    # Verify the criterion behaves correctly
    if max_normalized_edge > 1e-15:  # Only test when normalized edge is non-negligible
        assert not result_small, f"Criterion should return False when epsilon ({epsilon_small}) < normalized_edge ({max_normalized_edge})"
    assert result_large, f"Criterion should return True when epsilon ({epsilon_large}) > normalized_edge ({max_normalized_edge})"


# Feature: nelder-mead-thesis-refactor, Property 14: Flat simplex detects identical fitness values
@settings(max_examples=100)
@given(
    fitness_low=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    fitness_diff=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
def test_property_14_flat_simplex_detects_identical_fitness(fitness_low, fitness_diff):
    """
    Property 14: Flat simplex detects identical fitness values.
    
    For any simplex, the flat simplex criterion must return true when
    |f_max - f_min| < ε, where f_max and f_min are the maximum and minimum fitness values.
    
    Validates: Requirements 4.4
    """
    fitness_high = fitness_low + fitness_diff

    # Use the realized difference, not the requested one. Adding a tiny diff to a
    # large magnitude rounds straight back (256.0 + 2.7e-14 == 256.0, because the
    # float spacing at 256 is 5.7e-14), so the criterion correctly sees a
    # difference of zero while `fitness_diff` still claims to be positive.
    actual_diff = fitness_high - fitness_low

    # Test with various epsilon values
    epsilon_small = actual_diff / 2.0 if actual_diff > 0 else 1e-10
    epsilon_large = actual_diff * 2.0 + 1e-10
    
    result_small = check_flat_simplex(fitness_high, fitness_low, epsilon=epsilon_small)
    result_large = check_flat_simplex(fitness_high, fitness_low, epsilon=epsilon_large)
    
    # Verify the criterion behaves correctly
    if actual_diff > 1e-15:  # Only test when difference is non-negligible
        assert not result_small, f"Criterion should return False when epsilon ({epsilon_small}) < |f_high - f_low| ({actual_diff})"
        assert result_large, f"Criterion should return True when epsilon ({epsilon_large}) > |f_high - f_low| ({actual_diff})"


# Feature: nelder-mead-thesis-refactor, Property 15: Degenerate simplex detects loss of structure
@settings(max_examples=100)
@given(simplex=simplex_strategy())
def test_property_15_degenerate_simplex_detects_structure_loss(simplex):
    """
    Property 15: Degenerate simplex detects loss of structure.
    
    For any simplex, the degenerate criterion must check both edge ratio
    (min_edge/max_edge < ε₁) and determinant ratio (|det(V)|/Π||vᵢ|| < ε₂).
    
    Validates: Requirements 4.5
    """
    # Test 1: Non-degenerate simplex should not trigger with very strict epsilon values
    epsilon1 = 1e-6  # Very small edge ratio threshold
    epsilon2 = 1e-6  # Very small volume ratio threshold
    
    result_strict = check_degenerate_simplex(simplex, epsilon1, epsilon2)
    
    # Test 2: Very loose thresholds should detect most simplices as "degenerate"
    epsilon1_loose = 0.99  # Almost any simplex will have edge ratio < 0.99
    epsilon2_loose = 0.99  # Almost any simplex will have volume ratio < 0.99
    
    result_loose = check_degenerate_simplex(simplex, epsilon1_loose, epsilon2_loose)
    
    # We can't assert specific results without knowing if the simplex is actually degenerate,
    # but we can verify the function runs and returns a boolean
    assert isinstance(result_strict, (bool, np.bool_)), "Result should be boolean"
    assert isinstance(result_loose, (bool, np.bool_)), "Result should be boolean"
    
    # Test 3: Create a known degenerate simplex (all points nearly collinear)
    dim = simplex.shape[1]
    degenerate_simplex = np.zeros((dim + 1, dim))
    for i in range(dim + 1):
        degenerate_simplex[i, 0] = i * 1.0  # All points along first axis
        # Add tiny perturbations in other dimensions
        for j in range(1, dim):
            degenerate_simplex[i, j] = i * 1e-8
    
    result_degenerate = check_degenerate_simplex(degenerate_simplex, epsilon1=0.1, epsilon2=0.1)
    assert result_degenerate, "Known degenerate simplex should be detected as degenerate"
    
    # Test 4: Create a known non-degenerate regular simplex (only for 2D case)
    # For 2D: equilateral triangle centered away from origin
    if dim == 2:
        regular_simplex = np.array([
            [10.0, 10.0],
            [11.0, 10.0],
            [10.5, 10.0 + np.sqrt(3)/2]
        ])
        result_regular = check_degenerate_simplex(regular_simplex, epsilon1=0.01, epsilon2=0.01)
        assert not result_regular, "Regular simplex should not be detected as degenerate with strict thresholds"


# Import initialization functions
from src.nelder_mead.initialization.simplex_init import SimplexInitializer


# Feature: nelder-mead-thesis-refactor, Property 7: All initialization methods produce solutions within bounds
@settings(max_examples=100)
@given(
    dim=st.integers(min_value=2, max_value=10),
    method=st.sampled_from(['uniform', 'gaussian', 'spendleySimplex', 'pfefferSimplex', 'adaptiveSimplex'])
)
def test_property_7_initialization_within_bounds(dim, method):
    """
    Property 7: All initialization methods produce solutions within bounds.
    
    For any initialization method (uniform, Gaussian, Spendley, Pfeffer, adaptive)
    and any problem bounds, all generated solutions must satisfy
    lower ≤ x ≤ upper for all dimensions.
    
    Validates: Requirements 3.1, 3.2, 16.2, 16.3
    """
    # Generate random bounds
    lower_bounds = np.random.uniform(-100.0, -10.0, dim)
    upper_bounds = np.random.uniform(10.0, 100.0, dim)
    
    # Number of solutions for simplex (dim + 1)
    num_solutions = dim + 1
    
    # Create RNG with fixed seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Initialize simplex using the specified method
    if method in ['uniform', 'gaussian']:
        solutions = SimplexInitializer.initialize(
            method, num_solutions, lower_bounds, upper_bounds, dim, rng=rng
        )
    else:
        # Structured methods need x0
        x0 = (lower_bounds + upper_bounds) / 2.0
        solutions = SimplexInitializer.initialize(
            method, num_solutions, lower_bounds, upper_bounds, dim, x0=x0, rng=rng
        )
    
    # Verify all solutions are within bounds
    for i in range(num_solutions):
        for j in range(dim):
            assert lower_bounds[j] <= solutions[i, j] <= upper_bounds[j], \
                f"Solution {i}, dimension {j}: {solutions[i, j]} not in [{lower_bounds[j]}, {upper_bounds[j]}]"
    
    # Also verify using numpy operations
    assert np.all(solutions >= lower_bounds), \
        f"Some solutions violate lower bounds. Min values: {np.min(solutions, axis=0)}, Lower bounds: {lower_bounds}"
    assert np.all(solutions <= upper_bounds), \
        f"Some solutions violate upper bounds. Max values: {np.max(solutions, axis=0)}, Upper bounds: {upper_bounds}"


# Feature: nelder-mead-thesis-refactor, Property 8: Spendley simplex has regular geometry
@settings(max_examples=100)
@given(dim=st.integers(min_value=2, max_value=10))
def test_property_8_spendley_regular_geometry(dim):
    """
    Property 8: Spendley simplex has regular geometry.
    
    For any Spendley simplex initialization, all edges from the first vertex
    to other vertices must have equal length (within numerical tolerance).
    
    Validates: Requirements 3.3
    """
    # Generate bounds
    lower_bounds = np.full(dim, -10.0)
    upper_bounds = np.full(dim, 10.0)
    
    # Use center as starting point
    x0 = np.zeros(dim)
    
    # Initialize Spendley simplex
    num_solutions = dim + 1
    simplex = SimplexInitializer.spendley_simplex(
        num_solutions, lower_bounds, upper_bounds, dim, x0=x0
    )
    
    # Compute distances from first vertex to all other vertices
    first_vertex = simplex[0]
    distances = []
    for i in range(1, num_solutions):
        dist = np.linalg.norm(simplex[i] - first_vertex)
        distances.append(dist)
    
    # All distances should be equal (within tolerance)
    distances = np.array(distances)
    mean_distance = np.mean(distances)
    
    # Check that all distances are close to the mean
    # Use relative tolerance for numerical stability
    for i, dist in enumerate(distances):
        if mean_distance > 1e-10:  # Avoid division by zero
            relative_error = abs(dist - mean_distance) / mean_distance
            assert relative_error < 1e-6, \
                f"Edge {i+1} has distance {dist}, expected {mean_distance} (relative error: {relative_error})"
        else:
            # If mean distance is very small, use absolute tolerance
            assert abs(dist - mean_distance) < 1e-10, \
                f"Edge {i+1} has distance {dist}, expected {mean_distance}"


# Feature: nelder-mead-thesis-refactor, Property 9: Adaptive simplex uses dimension-dependent step sizes
@settings(max_examples=100)
@given(dim=st.integers(min_value=2, max_value=10))
def test_property_9_adaptive_dimension_dependent_steps(dim):
    """
    Property 9: Adaptive simplex uses dimension-dependent step sizes.
    
    For any adaptive simplex initialization with dimension n, the step size σ
    must equal min(max(||x₀||∞, 1), 10).
    
    Validates: Requirements 3.5
    """
    # Generate bounds
    lower_bounds = np.full(dim, -50.0)
    upper_bounds = np.full(dim, 50.0)
    
    # Test with different starting points to verify step size formula
    test_points = [
        np.zeros(dim),  # Origin: σ = min(max(0, 1), 10) = 1
        np.full(dim, 0.5),  # Small values: σ = min(max(0.5, 1), 10) = 1
        np.full(dim, 5.0),  # Medium values: σ = min(max(5, 1), 10) = 5
        np.full(dim, 15.0),  # Large values: σ = min(max(15, 1), 10) = 10
    ]
    
    for x0 in test_points:
        # Compute expected step size
        x0_inf_norm = np.linalg.norm(x0, ord=np.inf)
        expected_sigma = min(max(x0_inf_norm, 1.0), 10.0)
        
        # Initialize adaptive simplex
        num_solutions = dim + 1
        simplex = SimplexInitializer.adaptive_simplex(
            num_solutions, lower_bounds, upper_bounds, dim, x0=x0
        )
        
        # Verify first vertex is x0
        np.testing.assert_allclose(simplex[0], x0, rtol=1e-10, atol=1e-10)
        
        # For each other vertex, check that it differs from x0 by sigma in exactly one dimension
        for i in range(1, num_solutions):
            diff = simplex[i] - x0
            
            # Find the dimension where the step was applied
            # (should be dimension i-1, but might be clipped to bounds)
            non_zero_dims = np.where(np.abs(diff) > 1e-10)[0]
            
            # The step should be applied in dimension i-1
            target_dim = i - 1
            
            # Check if the step was applied (might be clipped by bounds)
            if x0[target_dim] + expected_sigma <= upper_bounds[target_dim]:
                # Step was not clipped
                expected_value = x0[target_dim] + expected_sigma
                np.testing.assert_allclose(
                    simplex[i, target_dim], expected_value, rtol=1e-10, atol=1e-10,
                    err_msg=f"Vertex {i}, dimension {target_dim}: expected step of {expected_sigma}"
                )
            else:
                # Step was clipped to upper bound
                np.testing.assert_allclose(
                    simplex[i, target_dim], upper_bounds[target_dim], rtol=1e-10, atol=1e-10,
                    err_msg=f"Vertex {i}, dimension {target_dim}: expected clipping to upper bound"
                )


# Feature: nelder-mead-thesis-refactor, Property 10: Bound enforcement clips to nearest boundary
@settings(max_examples=100)
@given(
    dim=st.integers(min_value=2, max_value=10),
    point=st.lists(
        st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10
    )
)
def test_property_10_bound_enforcement_clips_correctly(dim, point):
    """
    Property 10: Bound enforcement clips to nearest boundary.
    
    For any point x and bounds [lower, upper], enforcing bounds must produce x'
    where x'ᵢ = clip(xᵢ, lowerᵢ, upperᵢ) for all dimensions i.
    
    Validates: Requirements 16.1, 16.4
    """
    # Ensure point has correct dimension
    if len(point) != dim:
        point = point[:dim] if len(point) > dim else point + [0.0] * (dim - len(point))
    point = np.array(point)
    
    # Generate bounds
    lower_bounds = np.random.uniform(-100.0, -10.0, dim)
    upper_bounds = np.random.uniform(10.0, 100.0, dim)
    
    # Enforce bounds
    clipped_point = enforce_bounds(point, lower_bounds, upper_bounds)
    
    # Verify clipping formula: x'ᵢ = clip(xᵢ, lowerᵢ, upperᵢ)
    expected_clipped = np.clip(point, lower_bounds, upper_bounds)
    np.testing.assert_allclose(clipped_point, expected_clipped, rtol=1e-10, atol=1e-10)
    
    # Verify each dimension individually
    for i in range(dim):
        if point[i] < lower_bounds[i]:
            # Should be clipped to lower bound
            assert clipped_point[i] == lower_bounds[i], \
                f"Dimension {i}: {point[i]} should be clipped to lower bound {lower_bounds[i]}, got {clipped_point[i]}"
        elif point[i] > upper_bounds[i]:
            # Should be clipped to upper bound
            assert clipped_point[i] == upper_bounds[i], \
                f"Dimension {i}: {point[i]} should be clipped to upper bound {upper_bounds[i]}, got {clipped_point[i]}"
        else:
            # Should remain unchanged
            assert clipped_point[i] == point[i], \
                f"Dimension {i}: {point[i]} is within bounds and should remain unchanged, got {clipped_point[i]}"
    
    # Verify all values are within bounds
    assert np.all(clipped_point >= lower_bounds), "Some values below lower bounds"
    assert np.all(clipped_point <= upper_bounds), "Some values above upper bounds"
