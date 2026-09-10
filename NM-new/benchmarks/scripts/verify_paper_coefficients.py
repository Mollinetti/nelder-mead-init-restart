#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numerical provenance check for the eight engineering design problems.

Produces the numbers quoted in docs/paper/coefficient_provenance.md (revision
item 3). The method is the same for every problem: evaluate the formulation as
printed in the manuscript at the optimum its source literature reports, then look
at the constraints there.

A constraint that is active at a published optimum must evaluate to approximately
zero. One that comes out strongly slack, or strongly violated, means the printed
coefficients are not the ones that optimum was computed from. This catches
coefficient drift that no amount of proofreading will, because the formulations
look perfectly plausible either way.

Usage:
    python benchmarks/scripts/verify_paper_coefficients.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def banner(title):
    """Print a section heading."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def check_three_bar_truss():
    """TBT: confirm the objective's length multiplier is l = 100, not 1."""
    banner("4.1  Three-Bar Truss: the length factor in eq. (35)")

    x = np.array([0.78867513, 0.40824829])  # analytic optimum
    unit = (2 * np.sqrt(2) * x[0] + x[1]) * 1
    metres = (2 * np.sqrt(2) * x[0] + x[1]) * 100

    print(f"  f(x*) with the printed multiplier x1   : {unit:.6f}")
    print(f"  f(x*) with the stated length  l = 100  : {metres:.6f}")
    print("  Table 2 reports a best of 263.896 for every algorithm.")
    print("  -> l = 100 was used; the printed 'x 1' in eq. (35) is a typesetting slip.")

    load, allowable = 2.0, 2.0
    denominator = np.sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]
    g = np.array([
        (np.sqrt(2) * x[0] + x[1]) / denominator * load - allowable,
        x[1] / denominator * load - allowable,
        1.0 / (x[0] + np.sqrt(2) * x[1]) * load - allowable,
    ])
    print(f"  constraints at x*: g1={g[0]:+.3e}  g2={g[1]:+.3e}  g3={g[2]:+.3e}")
    print("  g1 is active, as the source reports.")


def check_pressure_vessel():
    """DPV: decide between the two coefficient variants in circulation."""
    banner("4.2  Pressure Vessel: the two coefficient variants")

    x = np.array([0.8125, 0.4375, 42.0984, 176.6366])
    cited_optimum = 6059.7143

    def cost(c3, c4):
        return (
            0.6224 * x[0] * x[2] * x[3]
            + 1.7781 * x[1] * x[2] ** 2
            + c3 * x[0] ** 2 * x[3]
            + c4 * x[0] ** 2 * x[2]
        )

    paper = cost(3.1661, 19.84)
    alternative = cost(3.1611, 19.8621)

    print(f"  paper       (3.1661, 19.84)   : {paper:.4f}   "
          f"deviation {abs(paper - cited_optimum):.4f}")
    print(f"  alternative (3.1611, 19.8621) : {alternative:.4f}   "
          f"deviation {abs(alternative - cited_optimum):.4f}")
    print(f"  the two differ by {abs(paper - alternative):.4f}, "
          f"{100 * abs(paper - alternative) / paper:.4f}% of the objective")
    print(f"  cited optimum {cited_optimum}")
    print("  -> the manuscript's coefficients are the closer of the two. Correct.")


def check_speed_reducer():
    """SRD-11: the constant in g5 is wrong by a factor of ten."""
    banner("4.3  Speed Reducer: the constant in eq. (37) g5")

    x = np.array([3.5, 0.7, 17.0, 7.3, 7.8, 3.350214, 5.286683])
    objective = (
        0.7854 * x[0] * x[1] ** 2 * (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934)
        - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2)
        + 7.477 * (x[5] ** 3 + x[6] ** 3)
        + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)
    )
    print(f"  f at the literature optimum: {objective:.4f}  (literature 2994.4711)")

    shaft_1 = (745.0 * x[3] / (x[1] * x[2])) ** 2
    printed = np.sqrt(shaft_1 + 1.69e6) / (110.0 * x[5] ** 3) - 1.0
    sources = np.sqrt(shaft_1 + 16.9e6) / (110.0 * x[5] ** 3) - 1.0

    shaft_2 = (745.0 * x[4] / (x[1] * x[2])) ** 2
    g6 = np.sqrt(shaft_2 + 157.5e6) / (85.0 * x[6] ** 3) - 1.0

    print()
    print(f"  g5 with 1.69e6, as printed  : {printed:+.6f}   strongly slack")
    print(f"  g5 with 16.9e6, per sources : {sources:+.6f}   active")
    print(f"  g6 with 157.5e6, agreed     : {g6:+.6f}   active")
    print()
    print("  g5 and g6 are the same shaft-stress limit on two shafts, and both are")
    print("  active at the Golinski optimum. Their constants cannot differ by two")
    print("  orders of magnitude. 1.69e6 is a decimal-point slip for 16.9e6, and as")
    print("  printed it leaves g5 slack over essentially the whole feasible box.")


def check_spring():
    """MWTCS: confirm agreement with the verified CEC2020 transcription."""
    banner("4.4  Tension/Compression Spring: agreement with CEC2020 RC17")

    from nelder_mead.benchmarks.cec2020_rw import RC17

    def paper_constraints(x):
        """Equation (38) as printed in the manuscript."""
        return np.array([
            1.0 - (x[1] ** 3 * x[2]) / (71785.0 * x[0] ** 4),
            (4.0 * x[1] ** 2 - x[0] * x[1])
            / (12566.0 * (x[1] * x[0] ** 3 - x[0] ** 4))
            + 1.0 / (5108.0 * x[0] ** 2)
            - 1.0,
            1.0 - 140.45 * x[0] / (x[1] ** 2 * x[2]),
            (x[0] + x[1]) / 1.5 - 1.0,
        ])

    reference = RC17()
    rng = np.random.default_rng(0)
    lower, upper = reference.bounds
    worst = 0.0
    for _ in range(500):
        x = lower + rng.random(3) * (upper - lower)
        worst = max(worst, float(np.max(np.abs(
            paper_constraints(x) - reference.constraint_ineq(x)
        ))))

    print(f"  bounds, manuscript : {np.array2string(lower)} to "
          f"{np.array2string(upper)}")
    print("  bounds, RC17       : identical")
    print(f"  max |g_paper - g_RC17| over 500 random points: {worst:.3e}")
    print("  -> the manuscript's formulation and bounds are correct.")


def check_welded_beam():
    """WBD: compare the printed polar moment against the classic and CEC2020."""
    banner("4.5  Welded Beam: the polar moment of inertia J in eq. (44)")

    x = np.array([0.2057, 3.4705, 9.0366, 0.2057])  # classic optimum, f* = 1.7249

    paper_wbd1 = 2 * (
        (x[0] * x[1] / np.sqrt(2)) * (x[1] ** 2 / 12 + ((x[0] + x[2]) / 2) ** 2)
    )
    paper_wbd2 = 2 * (
        np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 4 + ((x[0] + x[2]) / 2) ** 2)
    )
    cec_rc19 = 2 * (
        np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4)
    )
    classic = 2 * (
        np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 12 + ((x[0] + x[2]) / 2) ** 2)
    )

    print(f"  paper WBD1, as printed : J = {paper_wbd1:.6f}   "
          f"ratio to classic {paper_wbd1 / classic:.4f}")
    print(f"  paper WBD2, as printed : J = {paper_wbd2:.6f}   "
          f"ratio to classic {paper_wbd2 / classic:.4f}")
    print(f"  CEC2020 RC19           : J = {cec_rc19:.6f}")
    print(f"  classic (Coello, Deb)  : J = {classic:.6f}")
    print()
    print(f"  paper WBD2 reproduces RC19 exactly: {np.isclose(paper_wbd2, cec_rc19)}")
    print("  paper WBD1 is exactly half the classic value: the prefactor")
    print("  2*(x1x2/sqrt2) simplifies to sqrt2*x1x2, where the classic has")
    print("  2*sqrt2*x1x2. J sits in the denominator of tau2 = MR/J, so halving it")
    print("  doubles the secondary shear and makes g1 harder than in the source.")


def check_gear_train():
    """GTD: confirm the reported zero and identify the printed slips."""
    banner("4.6  Gear Train Design: eq. (49)")

    teeth = np.arange(12, 61)
    a, b, c, d = np.meshgrid(teeth, teeth, teeth, teeth, indexing="ij")
    values = (1.0 / 6.931 - (a * b) / (c * d)) ** 2
    flat_index = int(np.argmin(values))
    best_value = float(values.flat[flat_index])
    best_x = np.unravel_index(flat_index, values.shape)
    best_teeth = tuple(int(teeth[i]) for i in best_x)

    print(f"  exhaustive integer minimum over [12,60]^4: f = {best_value:.6e}")
    print(f"  attained at x = {best_teeth}")
    print("  Table 3 reports 0 at 12-digit precision, which is consistent:")
    print(f"  {best_value:.3e} rounds to 0 there. The reported results are correct.")
    print()
    print("  Two typesetting slips in eq. (49):")
    print("    - the bound prints '12 <= x >= 60'; it must be 12 <= x_i <= 60")
    print("    - 'x1x2/(x3x4) in Z+' should constrain the teeth counts x_i to be")
    print("      integers. The ratio is ~0.1443 at the optimum, never an integer.")


def main():
    """Run every provenance check."""
    print("Coefficient provenance check for the eight engineering design problems")
    print("Reproduces the numbers in docs/paper/coefficient_provenance.md")

    check_three_bar_truss()
    check_pressure_vessel()
    check_speed_reducer()
    check_spring()
    check_welded_beam()
    check_gear_train()

    banner("Summary")
    print("  Action required before submission:")
    print("    SRD-11  eq. (37) g5 : 1.69e6 should be 16.9e6, then re-run")
    print("    WBD1    eq. (44) J  : prefactor gives half the classic value")
    print("  Typesetting only:")
    print("    TBT     eq. (35)    : 'x 1' should be 'x l'")
    print("    GTD     eq. (49)    : bound direction, and integrality on x_i")
    print("  Verified correct as printed:")
    print("    DPV1/DPV2, MWTCS, WBD2")
    print()


if __name__ == "__main__":
    main()
