"""Encrypted statistics and private-analytics helpers.

These are traceable building blocks over bounded encrypted integers. The
order-statistics helpers (median, percentile) reuse the bitonic sorting
network, so they require a power-of-two array length. Helpers that build
lookup tables take explicit ``min_value`` / ``max_value`` bounds.
"""

from __future__ import annotations

from typing import Any, List

from ._utils import validate_bounds, validate_integer, validate_size
from .arrays import array_sum, make_argmax, make_sort
from .math import equal, greater, maximum, minimum
from .math.number_theory import make_isqrt


def array_mean(array: List[Any]) -> Any:
    """Calculate the floor mean of an encrypted array."""
    return array_sum(array) // len(array)


def array_variance(array: List[Any]) -> Any:
    """Calculate the floor variance of an encrypted array."""
    mean_value = array_mean(array)
    diff_sum: Any = 0
    for item in array:
        diff_sum = diff_sum + (item - mean_value) * (item - mean_value)
    return diff_sum // len(array)


def array_std(array: List[Any], min_value: int, max_value: int) -> Any:
    """Calculate the integer standard deviation (isqrt of the variance).

    ``min_value`` / ``max_value`` bound the array elements; the isqrt lookup
    is built over the worst-case variance for that range.
    """
    minimum_value, maximum_value = validate_bounds(min_value, max_value)
    span = maximum_value - minimum_value
    variance = array_variance(array)
    if span == 0:
        return variance
    isqrt = make_isqrt(span * span)
    return isqrt(variance)


def array_covariance(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the floor covariance of two encrypted arrays."""
    if len(array1) != len(array2):
        raise ValueError("The array sizes must be equal")
    mean1 = array_mean(array1)
    mean2 = array_mean(array2)
    product_sum: Any = 0
    for item1, item2 in zip(array1, array2):
        product_sum = product_sum + (item1 - mean1) * (item2 - mean2)
    return product_sum // len(array1)


def array_max(elements: List[Any]) -> Any:
    """Find the maximum value in an encrypted array using a tournament reduction."""
    current_round = list(elements)
    if len(current_round) == 0:
        return 0
    while len(current_round) > 1:
        next_round = []
        for index in range(0, len(current_round) - 1, 2):
            next_round.append(maximum(current_round[index], current_round[index + 1]))
        if len(current_round) % 2 == 1:
            next_round.append(current_round[-1])
        current_round = next_round
    return current_round[0]


def array_min(elements: List[Any]) -> Any:
    """Find the minimum value in an encrypted array using a tournament reduction."""
    current_round = list(elements)
    if len(current_round) == 0:
        return 0
    while len(current_round) > 1:
        next_round = []
        for index in range(0, len(current_round) - 1, 2):
            next_round.append(minimum(current_round[index], current_round[index + 1]))
        if len(current_round) % 2 == 1:
            next_round.append(current_round[-1])
        current_round = next_round
    return current_round[0]


def array_range(array: List[Any]) -> Any:
    """Calculate the range (max - min) of an encrypted array."""
    return array_max(array) - array_min(array)


def array_count_greater(array: List[Any], threshold: Any) -> Any:
    """Count how many elements are strictly greater than a threshold."""
    count: Any = 0
    for item in array:
        count = count + greater(item, threshold)
    return count


def array_median(array: List[Any], min_value: int, max_value: int) -> Any:
    """Return the median of an encrypted array (floor average of the middle pair).

    Uses the bitonic sorting network, so the array length must be a power
    of two.
    """
    items = list(array)
    size = validate_size(len(items), power_of_two=True)
    minimum_value, maximum_value = validate_bounds(min_value, max_value)
    sort_values = make_sort(size, minimum_value, maximum_value)
    ordered = sort_values(items)
    middle = size // 2
    if size % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def array_percentile(array: List[Any], q: int, min_value: int, max_value: int) -> Any:
    """Return the nearest-rank q-th percentile (0..100) of an encrypted array.

    Uses the bitonic sorting network, so the array length must be a power
    of two.
    """
    items = list(array)
    size = validate_size(len(items), power_of_two=True)
    percent = validate_integer("q", q, minimum=0)
    if percent > 100:
        raise ValueError("q must be between 0 and 100")
    minimum_value, maximum_value = validate_bounds(min_value, max_value)
    sort_values = make_sort(size, minimum_value, maximum_value)
    ordered = sort_values(items)
    rank = round(percent / 100 * (size - 1))
    return ordered[rank]


def array_histogram(array: List[Any], min_value: int, max_value: int) -> List[Any]:
    """Count occurrences of every value in [min_value, max_value] (bincount)."""
    minimum_value, maximum_value = validate_bounds(min_value, max_value)
    items = list(array)
    return [
        array_sum([equal(item, value) for item in items])
        for value in range(minimum_value, maximum_value + 1)
    ]


def array_mode(array: List[Any], min_value: int, max_value: int) -> Any:
    """Return the most frequent value (smallest value wins ties)."""
    minimum_value, maximum_value = validate_bounds(min_value, max_value)
    counts = array_histogram(array, minimum_value, maximum_value)
    if len(counts) == 1:
        return minimum_value + counts[0] - counts[0]
    arg_max = make_argmax(len(counts), 0, len(list(array)))
    return minimum_value + arg_max(counts)


def array_normalize(array: List[Any], mean: Any, scale: int) -> List[Any]:
    """Return (x - mean) * scale for every element (z-score style affine transform)."""
    normalized_scale = validate_integer("scale", scale, minimum=1)
    return [(item - mean) * normalized_scale for item in array]


__all__ = [
    "array_count_greater",
    "array_covariance",
    "array_histogram",
    "array_max",
    "array_mean",
    "array_median",
    "array_min",
    "array_mode",
    "array_normalize",
    "array_percentile",
    "array_range",
    "array_std",
    "array_variance",
]
