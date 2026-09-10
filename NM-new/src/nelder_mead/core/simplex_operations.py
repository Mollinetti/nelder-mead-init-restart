"""
Pure simplex geometric operations for Nelder-Mead algorithms.

This module provides stateless, deterministic functions for all simplex transformations.
All operations return new arrays without mutating inputs.

References:
    - Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization.
      The Computer Journal, 7(4), 308-313.
    - Gao, F., & Han, L. (2012). Implementing the Nelder-Mead simplex algorithm with
      adaptive parameters. Computational Optimization and Applications, 51(1), 259-277.
"""

import math

import numpy as np


def compute_centroid(simplex: np.ndarray, exclude_worst: bool = True) -> np.ndarray:
    """
    Compute the centroid of simplex vertices.

    By default, excludes the worst point (last vertex after sorting by fitness).
    This is the standard behavior in Nelder-Mead algorithms.

    Args:
        simplex: Array of shape (n+1, n) containing simplex vertices
        exclude_worst: If True, compute centroid of first n vertices only

    Returns:
        Centroid as array of shape (n,)

    Example:
        >>> simplex = np.array([[0, 0], [1, 0], [0, 1]])
        >>> compute_centroid(simplex, exclude_worst=True)
        array([0.5, 0. ])  # Mean of first 2 vertices
    """
    if exclude_worst:
        # Exclude last vertex (worst point after sorting)
        return np.mean(simplex[:-1], axis=0)
    else:
        return np.mean(simplex, axis=0)


def reflect(
    centroid: np.ndarray, worst_point: np.ndarray, delta_r: float = 1.0
) -> np.ndarray:
    """
    Compute reflected point across centroid from worst point.

    Formula: r = c + δᵣ(c - w)
    where c is centroid, w is worst point, δᵣ is reflection coefficient

    The reflection operation is the fundamental move in Nelder-Mead, attempting to
    replace the worst point by reflecting it through the centroid of the remaining
    points. With δᵣ = 1.0 (standard), the reflected point is equidistant from the
    centroid as the worst point, but on the opposite side.

    Reference: Nelder & Mead (1965), Section 2, Equation (1)

    Args:
        centroid: Centroid of simplex (excluding worst point)
        worst_point: Worst vertex in simplex
        delta_r: Reflection coefficient (default: 1.0, standard value from Nelder & Mead 1965)

    Returns:
        Reflected point as array of shape (n,)

    Example:
        >>> centroid = np.array([0.5, 0.0])
        >>> worst = np.array([0.0, 1.0])
        >>> reflect(centroid, worst, delta_r=1.0)
        array([1., -1.])  # Point on opposite side of centroid
    """
    # Compute direction from worst to centroid: (c - w)
    # Then extend by δᵣ from centroid: c + δᵣ(c - w)
    return centroid + delta_r * (centroid - worst_point)


def expand(
    centroid: np.ndarray, reflected_point: np.ndarray, delta_e: float = 2.0
) -> np.ndarray:
    """
    Compute expanded point beyond reflected point.

    Formula: e = c + δₑ(r - c)
    where c is centroid, r is reflected point, δₑ is expansion coefficient

    Expansion is attempted when the reflected point is better than all current
    simplex vertices, suggesting the search direction is promising. The expansion
    extends further in the same direction to potentially find an even better point.
    With δₑ = 2.0 (standard), the expanded point is twice as far from the centroid
    as the reflected point.

    Reference: Nelder & Mead (1965), Section 2, Equation (2)

    Args:
        centroid: Centroid of simplex
        reflected_point: Previously computed reflected point
        delta_e: Expansion coefficient (default: 2.0, standard value from Nelder & Mead 1965)

    Returns:
        Expanded point as array of shape (n,)

    Example:
        >>> centroid = np.array([0.5, 0.0])
        >>> reflected = np.array([1.0, -1.0])
        >>> expand(centroid, reflected, delta_e=2.0)
        array([1.5, -2.])  # Extends beyond reflection
    """
    # Compute direction from centroid to reflection: (r - c)
    # Then extend by δₑ from centroid: c + δₑ(r - c)
    return centroid + delta_e * (reflected_point - centroid)


def contract_outside(
    centroid: np.ndarray, reflected_point: np.ndarray, delta_oc: float = 0.5
) -> np.ndarray:
    """
    Compute outside contraction point between centroid and reflected point.

    Formula: oc = c + δₒc(r - c)
    where c is centroid, r is reflected point, δₒc is outside contraction coefficient

    Outside contraction is attempted when the reflected point is better than the
    worst point but not good enough to accept directly. It contracts the reflected
    point back toward the centroid. With δₒc = 0.5 (standard), the contracted point
    is halfway between the centroid and the reflected point. The term "outside"
    refers to the fact that this point is on the opposite side of the centroid
    from the worst point (outside the original simplex).

    Reference: Nelder & Mead (1965), Section 2, Equation (3)

    Args:
        centroid: Centroid of simplex
        reflected_point: Previously computed reflected point
        delta_oc: Outside contraction coefficient (default: 0.5, standard value from Nelder & Mead 1965)

    Returns:
        Outside contracted point as array of shape (n,)

    Example:
        >>> centroid = np.array([0.5, 0.0])
        >>> reflected = np.array([1.0, -1.0])
        >>> contract_outside(centroid, reflected, delta_oc=0.5)
        array([0.75, -0.5])  # Halfway between centroid and reflection
    """
    # Compute direction from centroid to reflection: (r - c)
    # Then contract by δₒc from centroid: c + δₒc(r - c)
    # With δₒc = 0.5, this gives the midpoint
    return centroid + delta_oc * (reflected_point - centroid)


def contract_inside(
    centroid: np.ndarray, worst_point: np.ndarray, delta_ic: float = -0.5
) -> np.ndarray:
    """
    Compute inside contraction point between centroid and worst point.

    Formula: ic = c + δᵢc(c - w)
    where c is centroid, w is worst point, δᵢc is inside contraction coefficient

    Inside contraction is attempted when the reflected point is worse than the
    worst point, indicating the reflection went in the wrong direction. Instead,
    we contract toward the centroid from the worst point. The term "inside" refers
    to the fact that this point is on the same side of the centroid as the worst
    point (inside the original simplex).

    Note: δᵢc is negative (e.g., -0.5), matching Lagarias et al. (1998) and the
    thesis eq. (2.7) y' = c + α(c - y^n). With δᵢc = -0.5 this gives
    ic = c - 0.5(c - w) = ½c + ½w, the midpoint of the centroid and the worst
    vertex — a point strictly *inside* the simplex, which is what makes the step
    an inside contraction.

    Reference: Nelder & Mead (1965), Section 2, Equation (4);
               Lagarias et al. (1998), inside contraction x_cc = c̄ - γ(c̄ - x_{n+1})

    Args:
        centroid: Centroid of simplex
        worst_point: Worst vertex in simplex
        delta_ic: Inside contraction coefficient (default: -0.5, standard value from Nelder & Mead 1965)

    Returns:
        Inside contracted point as array of shape (n,)

    Example:
        >>> centroid = np.array([0.5, 0.0])
        >>> worst = np.array([0.0, 1.0])
        >>> contract_inside(centroid, worst, delta_ic=-0.5)
        array([0.25, 0.5])  # Midpoint of centroid and worst point
    """
    # Direction from worst toward centroid: (c - w).
    # With δᵢc < 0 this steps back from the centroid toward the worst vertex,
    # landing between the two — inside the simplex.
    return centroid + delta_ic * (centroid - worst_point)


def shrink(
    best_point: np.ndarray, simplex: np.ndarray, gamma_s: float = 0.5
) -> np.ndarray:
    """
    Shrink all simplex vertices toward the best point.

    Formula: vᵢ' = b + γₛ(vᵢ - b) for all vertices vᵢ
    where b is best point, γₛ is shrink coefficient

    Shrink is the most expensive operation in Nelder-Mead, as it requires
    re-evaluating all n vertices (except the best). It is used as a last resort
    when all other operations (reflection, expansion, contraction) fail to produce
    an acceptable point. The shrink operation contracts the entire simplex toward
    the best vertex, reducing its size. With γₛ = 0.5 (standard), each vertex
    moves halfway toward the best point.

    Reference: Nelder & Mead (1965), Section 2, Equation (5)

    Args:
        best_point: Best vertex in simplex (typically first after sorting)
        simplex: Array of shape (n+1, n) containing all vertices
        gamma_s: Shrink coefficient (default: 0.5, standard value from Nelder & Mead 1965)

    Returns:
        New simplex with all vertices shrunk toward best point

    Example:
        >>> best = np.array([0.0, 0.0])
        >>> simplex = np.array([[0, 0], [2, 0], [0, 2]])
        >>> shrink(best, simplex, gamma_s=0.5)
        array([[0., 0.],
               [1., 0.],
               [0., 1.]])  # All points moved halfway toward best
    """
    # Create new simplex with all vertices shrunk toward best
    new_simplex = np.zeros_like(simplex)
    for i in range(len(simplex)):
        # For each vertex vᵢ, compute: b + γₛ(vᵢ - b)
        # This moves the vertex a fraction γₛ of the way from best to vᵢ
        # Note: The best point itself (i=0) remains unchanged since (b - b) = 0
        new_simplex[i] = best_point + gamma_s * (simplex[i] - best_point)
    return new_simplex


def enforce_bounds(
    point: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """
    Enforce box constraints by clipping point to nearest boundary.

    Formula: xᵢ' = clip(xᵢ, lowerᵢ, upperᵢ) for all dimensions i

    Args:
        point: Point to enforce bounds on
        lower: Lower bounds for each dimension
        upper: Upper bounds for each dimension

    Returns:
        Point with all coordinates within bounds

    Raises:
        ValueError: If lower bounds are not less than upper bounds

    Example:
        >>> point = np.array([1.5, -0.5, 0.5])
        >>> lower = np.array([0.0, 0.0, 0.0])
        >>> upper = np.array([1.0, 1.0, 1.0])
        >>> enforce_bounds(point, lower, upper)
        array([1., 0., 0.5])  # Clipped to [0, 1] in each dimension
    """
    if np.any(lower >= upper):
        raise ValueError(
            "Lower bounds must be less than upper bounds in all dimensions"
        )

    return np.clip(point, lower, upper)


def edge_matrix(points: np.ndarray) -> np.ndarray:
    """
    Build the matrix of edge vectors radiating from the first point.

    Formula: L(Y) = [y¹ - y⁰, y² - y⁰, ..., yᵏ - y⁰] ∈ ℝ^(n×k)

    This is the matrix that appears in both the simplex volume and the simplex
    gradient. `points` is expected to be ordered by objective value, so that
    y⁰ is the best point.

    Args:
        points: Array of shape (k+1, n) containing k+1 points in n dimensions

    Returns:
        Edge matrix of shape (n, k)

    Example:
        >>> pts = np.array([[0., 0.], [1., 0.], [0., 2.]])
        >>> edge_matrix(pts)
        array([[1., 0.],
               [0., 2.]])
    """
    Y = np.asarray(points, dtype=float)
    return (Y[1:] - Y[0]).T


def oriented_length(simplex: np.ndarray) -> float:
    """
    Compute the oriented length of a simplex.

    Formula: σ⁺(Y) = max_{1≤i≤n} ‖yⁱ - y⁰‖

    The oriented length is a cheaper stand-in for the diameter, bounded by
    σ⁺(Y) ≤ diam(Y) ≤ 2σ⁺(Y). It is the natural length scale of a simplex and
    is what a step size should be measured against.

    Args:
        simplex: Array of shape (n+1, n) containing simplex vertices, ordered
                 by objective value so that y⁰ is the best point

    Returns:
        Oriented length (0.0 for a fully collapsed simplex)

    Example:
        >>> simplex = np.array([[0., 0.], [3., 0.], [0., 4.]])
        >>> oriented_length(simplex)
        4.0
    """
    Y = np.asarray(simplex, dtype=float)
    if len(Y) < 2:
        return 0.0

    return float(np.max(np.linalg.norm(Y[1:] - Y[0], axis=1)))


def simplex_volume(simplex: np.ndarray) -> float:
    """
    Compute the volume of a simplex.

    Formula: vol(Y) = |det(L(Y))| / n!

    A simplex is nondegenerate exactly when its volume is strictly positive.
    Nelder-Mead relies on nondegeneracy being preserved across iterations, so
    this is the natural quantity to assert on.

    Args:
        simplex: Array of shape (n+1, n) containing simplex vertices

    Returns:
        Volume of the simplex (0.0 if degenerate)

    Raises:
        ValueError: If the simplex is not (n+1, n) shaped

    Example:
        >>> simplex = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> simplex_volume(simplex)
        0.5
    """
    Y = np.asarray(simplex, dtype=float)
    n_vertices, dim = Y.shape
    if n_vertices != dim + 1:
        raise ValueError(
            f"Volume requires an (n+1, n) simplex, got {Y.shape}"
        )

    return abs(np.linalg.det(edge_matrix(Y))) / math.factorial(dim)


def simplex_gradient(points: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Compute the simplex gradient of f with respect to a set of sampled points.

    The simplex gradient is the least-squares solution g of

        L(Y)ᵀ g = δ_f,    δ_f = [f(y¹) - f(y⁰), ..., f(yᵏ) - f(y⁰)]ᵀ

    When k = n and L is invertible this is the determined case and reduces to
    ∇ₛf(Y) = (Lᵀ)⁻¹ δ_f. When k > n it is the overdetermined case and the least
    squares solution is returned. `np.linalg.lstsq` covers both, and also
    degrades gracefully to the minimum-norm solution when L is rank deficient.

    Custódio and Vicente showed the simplex gradient approximates the true
    gradient well when the sample set is well poised, i.e. when L has full rank.
    Callers that need that guarantee should check `np.linalg.matrix_rank(L)`.

    Args:
        points: Array of shape (k+1, n), ordered by objective value (best first)
        values: Array of shape (k+1,) with the objective value of each point

    Returns:
        Simplex gradient as array of shape (n,)

    Raises:
        ValueError: If fewer than 2 points are given, or shapes disagree

    Example:
        >>> pts = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> vals = np.array([0., 1., 2.])   # f(x, y) = x + 2y
        >>> np.round(simplex_gradient(pts, vals), 6)
        array([1., 2.])
    """
    Y = np.asarray(points, dtype=float)
    f = np.asarray(values, dtype=float)

    if Y.shape[0] < 2:
        raise ValueError("Simplex gradient needs at least 2 points")
    if Y.shape[0] != f.shape[0]:
        raise ValueError(
            f"Got {Y.shape[0]} points but {f.shape[0]} values"
        )

    L = edge_matrix(Y)  # (n, k)
    delta_f = f[1:] - f[0]  # (k,)

    gradient, *_ = np.linalg.lstsq(L.T, delta_f, rcond=None)
    return gradient


def merge_simplices(
    simplex1: np.ndarray, simplex2: np.ndarray, gamma_m: float = 0.8
) -> np.ndarray:
    """
    Merge two simplices into one, keeping the best vertex of the first.

    Formula: Yₙ = {y₁⁰} ∪ {y₁ⁱ + γᵐ(y₂ⁱ⁻¹ - y₁ⁱ) : i = 1, ..., n}

    Each non-best vertex of Y₁ is moved a fraction γᵐ of the way toward a vertex
    of Y₂. The best vertex y₁⁰ is retained unchanged, so the merge never
    discards the incumbent.

    Args:
        simplex1: Array of shape (n+1, n), ordered by objective value
        simplex2: Array of shape (n+1, n), ordered by objective value
        gamma_m: Merge step length in [0, 1] (default: 0.8)

    Returns:
        Merged simplex of shape (n+1, n)

    Raises:
        ValueError: If the two simplices have different shapes

    Example:
        >>> y1 = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> y2 = np.array([[2., 2.], [3., 2.], [2., 3.]])
        >>> merge_simplices(y1, y2, gamma_m=0.5)
        array([[0. , 0. ],
               [1.5, 1. ],
               [1.5, 1.5]])
    """
    Y1 = np.asarray(simplex1, dtype=float)
    Y2 = np.asarray(simplex2, dtype=float)

    if Y1.shape != Y2.shape:
        raise ValueError(
            f"Cannot merge simplices of shape {Y1.shape} and {Y2.shape}"
        )

    merged = Y1.copy()
    # y₁ⁱ + γᵐ(y₂ⁱ⁻¹ - y₁ⁱ) for i = 1..n; Y2[:-1] supplies y₂⁰..y₂ⁿ⁻¹
    merged[1:] = Y1[1:] + gamma_m * (Y2[:-1] - Y1[1:])
    return merged


def simplex_conditioning(simplex: np.ndarray) -> tuple[float, float]:
    """
    Geometric conditioning of a simplex, as the two GBNM degeneracy measures.

    Returns (edge_ratio, hadamard_ratio) for the edge matrix L(Y) = [y¹-y⁰, ...]:

        edge_ratio     = min_k ‖eᵏ‖ / max_k ‖eᵏ‖        ∈ [0, 1]
        hadamard_ratio = |det L(Y)| / ∏_k ‖eᵏ‖          ∈ [0, 1]

    Both are scale- and translation-invariant, and both equal their maximum on a
    simplex with equal-length orthogonal edges. The Hadamard ratio is the shape
    term proper: it is 1 iff the edges are orthogonal and 0 iff they are linearly
    dependent, independent of how long they are. The edge ratio catches the
    separate failure of edges of wildly different lengths.

    These are exactly the two quantities in eq. (9) of Luersen & Le Riche,
    "Globalized Nelder-Mead method for engineering optimization", Computers &
    Structures 82(23):2251-2260, 2004, which declares a simplex degenerate when
    either falls below a tolerance (their εs3, εs4).

    Args:
        simplex: Array of shape (n+1, n) containing simplex vertices

    Returns:
        (edge_ratio, hadamard_ratio); (0.0, 0.0) if any edge has zero length

    Example:
        >>> unit = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> simplex_conditioning(unit)
        (1.0, 1.0)
    """
    L = edge_matrix(simplex)
    lengths = np.linalg.norm(L, axis=0)
    max_len = lengths.max(initial=0.0)
    if max_len <= 0.0 or not np.all(lengths > 0.0):
        return 0.0, 0.0

    edge_ratio = float(lengths.min() / max_len)
    if L.shape[0] != L.shape[1]:
        return edge_ratio, 0.0

    # In log space: for n=10 edges of length ~1e-40, both |det L| and the product
    # underflow to 0.0 and the ratio comes out NaN, which silently poisons the
    # trace exactly where the simplex is most degenerate -- the regime of interest.
    sign, log_abs_det = np.linalg.slogdet(L)
    if sign == 0:
        return edge_ratio, 0.0
    return edge_ratio, float(np.exp(log_abs_det - np.log(lengths).sum()))


def oriented_restart(
    simplex: np.ndarray,
    fitness_values: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """
    Rebuild a simplex as a smaller orthogonal one oriented along -∇ₛf.

    Kelley eq. (8.7), *Iterative Methods for Optimization*, SIAM 1999:

        y¹ = x¹,   yʲ = y¹ - β_{j-1} e_{j-1}  for 2 ≤ j ≤ n+1,

        βₗ = ½ σ⁻(S) · sign((Df)ₗ)   when (Df)ₗ ≠ 0,
        βₗ = ½ σ⁻(S)                 when (Df)ₗ = 0,

    where σ⁻(S) is the *smallest* edge length from the best vertex. Scaling by
    σ⁻ rather than the oriented length σ⁺ is what gives Kelley's guarantee that
    every edge is shortened, σ⁺(S_{k+1}) ≤ σ⁻(S_k).

    This is a *construction*: it says how to re-seed, not when. Kelley pairs it
    with his sufficient-decrease detector, but the two are separable, and keeping
    them separable is what lets a study attribute an effect to one or the other.

    Args:
        simplex: Array of shape (n+1, n), ordered so that row 0 is the best point
        fitness_values: Objective values matching `simplex`
        lower_bounds: Lower bounds, for clipping
        upper_bounds: Upper bounds, for clipping

    Returns:
        The new simplex, same shape, with the best vertex preserved in row 0

    Example:
        >>> Y = np.array([[0., 0.], [1., 0.], [0., 1.]])
        >>> f = np.array([0., 1., 2.])
        >>> new = oriented_restart(Y, f, np.array([-5., -5.]), np.array([5., 5.]))
        >>> np.allclose(new[0], Y[0])
        True
    """
    Y = np.asarray(simplex, dtype=float)
    best = Y[0].copy()
    gradient = simplex_gradient(Y, fitness_values)

    edges = np.linalg.norm(Y[1:] - Y[0], axis=1)
    sigma = 0.5 * float(edges.min()) if len(edges) else 0.0
    if sigma <= 0.0:
        sigma = 1e-4 * float(np.min(upper_bounds - lower_bounds))

    dim = Y.shape[1]
    new_simplex = np.zeros_like(Y)
    new_simplex[0] = best
    for i in range(1, Y.shape[0]):
        step = np.zeros(dim)
        axis = (i - 1) % dim
        g = gradient[axis]
        step[axis] = -sigma * (np.sign(g) if g != 0.0 else 1.0)
        new_simplex[i] = enforce_bounds(best + step, lower_bounds, upper_bounds)

    return new_simplex
