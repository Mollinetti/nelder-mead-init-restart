"""
Nelder-Mead algorithm implementations.

This module contains the classic and adaptive Nelder-Mead algorithms.
"""

from .base_algorithm import BaseAlgorithm
from .nelder_mead import NelderMead
from .adaptive_nm import AdaptiveNelderMead
from .gradient_nm import GradientNelderMead
from .safeguarded_nm import SafeguardedNelderMead
from .kelley_nm import KelleyNelderMead

__all__ = [
    "BaseAlgorithm",
    "NelderMead",
    "AdaptiveNelderMead",
    "GradientNelderMead",
    "SafeguardedNelderMead",
    "KelleyNelderMead",
]
