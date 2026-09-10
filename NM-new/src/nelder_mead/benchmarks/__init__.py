"""
Benchmark optimization problems.

This module contains standard test problems for validating optimization algorithms.
"""

from nelder_mead.benchmarks.problem_suite import OptimizationProblem
from nelder_mead.benchmarks.unconstrained import (
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
    Griewank,
    Schwefel,
)
from nelder_mead.benchmarks.constrained import (
    G01,
    G04,
    G05,
    G06,
    G07,
    G08,
    G09,
    G10,
    G11,
    PressureVessel,
    WeldedBeam,
    get_all_constrained_problems,
    get_cec_problems,
    get_engineering_problems,
)
from nelder_mead.benchmarks.cec2020_rw import (
    RC01,
    RC02,
    RC03,
    RC04,
    RC05,
    RC06,
    RC08,
    RC14,
    RC15,
    RC16,
    RC17,
    RC19,
    get_cec2020_rw_problems,
    get_cec2020_rw_by_dimension,
)
from nelder_mead.benchmarks.scalable_constrained import (
    ScalableSphereInequality,
    ScalableRotatedInequality,
    ScalableEqualityManifold,
    ScalableActiveSetInequality,
    SCALABLE_FAMILIES,
    DEFAULT_DIMENSIONS,
    get_scalability_sweep,
)
from nelder_mead.benchmarks.blackbox import (
    CantileverBracketFEA,
    get_blackbox_problems,
)
from nelder_mead.benchmarks.mopta08 import (
    MOPTA08,
    download_executable,
    executable_name,
    is_available,
)

__all__ = [
    "OptimizationProblem",
    # Unconstrained problems
    "Sphere",
    "Rosenbrock",
    "Rastrigin",
    "Ackley",
    "Griewank",
    "Schwefel",
    # Constrained problems
    "G01",
    "G04",
    "G05",
    "G06",
    "G07",
    "G08",
    "G09",
    "G10",
    "G11",
    "PressureVessel",
    "WeldedBeam",
    # CEC2020 real-world constrained problems
    "RC01",
    "RC02",
    "RC03",
    "RC04",
    "RC05",
    "RC06",
    "RC08",
    "RC14",
    "RC15",
    "RC16",
    "RC17",
    "RC19",
    # Scalable families for the dimension sweep
    "ScalableSphereInequality",
    "ScalableRotatedInequality",
    "ScalableEqualityManifold",
    "ScalableActiveSetInequality",
    "SCALABLE_FAMILIES",
    "DEFAULT_DIMENSIONS",
    # Simulation-based black-box problems
    "CantileverBracketFEA",
    "MOPTA08",
    # Helper functions
    "get_all_constrained_problems",
    "get_cec_problems",
    "get_engineering_problems",
    "get_cec2020_rw_problems",
    "get_cec2020_rw_by_dimension",
    "get_scalability_sweep",
    "get_blackbox_problems",
    "download_executable",
    "executable_name",
    "is_available",
]
