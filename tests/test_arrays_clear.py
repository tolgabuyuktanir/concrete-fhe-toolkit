from itertools import product

import numpy as np
import pytest

from concrete_fhe_toolkit import (
    make_argmax,
    make_argmin,
    make_compare_swap,
    make_array_maximum,
    make_array_minimum,
    make_sort,
)
from concrete_fhe_toolkit.arrays.arithmetic import array_sum, array_scale, array_add, array_sub, array_multiply
from concrete_fhe_toolkit.arrays.manipulation import array_slice, array_all_equal, make_array_pad
from concrete_fhe_toolkit.arrays.search import make_array_contains, make_array_index_of


def test_compare_swap_exhaustive():
    compare_swap = make_compare_swap(-2, 3)

    for left, right in product(range(-2, 4), repeat=2):
        assert tuple(compare_swap(left, right)) == (
            min(left, right),
            max(left, right),
        )


@pytest.mark.parametrize("descending", [False, True])
def test_sort_exhaustive(descending):
    sort_values = make_sort(4, -2, 2, descending=descending)

    for values in product(range(-2, 3), repeat=4):
        actual = sort_values(np.array(values, dtype=np.int64))
        expected = sorted(values, reverse=descending)
        assert np.array_equal(actual, expected)


def test_minimum_and_maximum_support_odd_sizes():
    values = np.array([3, -2, 7, -2, 4], dtype=np.int64)

    assert int(make_array_minimum(5, -2, 7)(values)) == -2
    assert int(make_array_maximum(5, -2, 7)(values)) == 7


@pytest.mark.parametrize(
    ("factory", "tie_break", "expected"),
    [
        (make_argmin, "first", 1),
        (make_argmin, "last", 2),
        (make_argmax, "first", 0),
        (make_argmax, "last", 4),
    ],
)
def test_arg_extrema_have_explicit_tie_behavior(factory, tie_break, expected):
    values = np.array([3, 1, 1, 2, 3], dtype=np.int64)
    function = factory(5, 1, 3, tie_break=tie_break)

    assert int(function(values)) == expected


def test_arg_extrema_support_single_element_arrays():
    values = np.array([-4], dtype=np.int64)

    assert int(make_argmin(1, -4, -4)(values)) == 0
    assert int(make_argmax(1, -4, -4)(values)) == 0


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="power of two"):
        make_sort(3)

    with pytest.raises(ValueError, match="min_value"):
        make_array_minimum(4, 3, 2)

    with pytest.raises(ValueError, match="tie_break"):
        make_argmin(4, tie_break="middle")


def test_array_arithmetic_functions():
    sample1 = np.array([1, 2, 3], dtype=np.int64)
    sample2 = np.array([4, 5, 6], dtype=np.int64)
    
    assert int(array_sum(sample1)) == 6
    assert np.array_equal(array_scale(sample1, 2), [2, 4, 6])
    assert np.array_equal(array_add(sample1, sample2), [5, 7, 9])
    assert np.array_equal(array_sub(sample1, sample2), [-3, -3, -3])
    assert np.array_equal(array_multiply(sample1, sample2), [4, 10, 18])


def test_array_manipulation_functions():
    sample = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    
    # Slice
    sliced = array_slice(sample, 1, 4)
    assert np.array_equal(sliced, [2, 3, 4])
    
    # Pad
    pad_fn = make_array_pad(size=5, target_size=7, min_value=1, max_value=9)
    padded = pad_fn(sample)
    assert np.array_equal(padded, [1, 2, 3, 4, 5, 0, 0])
    
    # All equal
    assert array_all_equal(np.array([1, 1, 1], dtype=np.int64), np.array([1, 1, 1], dtype=np.int64)) == 1
    assert array_all_equal(sample, np.array([1, 1, 1, 1, 1], dtype=np.int64)) == 0


def test_array_search_functions():
    sample = np.array([1, 2, 3, 2, 4], dtype=np.int64)
    
    contains_fn = make_array_contains(size=5, min_value=1, max_value=9)
    assert int(contains_fn(sample, 2)) == 1
    assert int(contains_fn(sample, 9)) == 0
    
    index_of_fn = make_array_index_of(size=5, min_value=1, max_value=4)
    assert int(index_of_fn(sample, 2)) == 1
