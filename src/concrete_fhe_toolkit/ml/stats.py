"""Deprecated location: these helpers moved to concrete_fhe_toolkit.stats.

This module re-exports them for backwards compatibility.
"""

from concrete_fhe_toolkit.stats import (
    array_count_greater,
    array_max,
    array_mean,
    array_min,
    array_range,
    array_variance,
)

__all__ = [
    "array_count_greater",
    "array_max",
    "array_mean",
    "array_min",
    "array_range",
    "array_variance",
]
