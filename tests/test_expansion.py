"""Tests for the layer-expansion batch: math, bits, arrays, stats, ml."""

import math
from itertools import product

import numpy as np
import pytest

from concrete_fhe_toolkit import (
    array_concat,
    array_cumsum,
    array_index,
    array_index_of,
    array_reverse,
    array_set,
    compile_top_k,
    make_top_k,
    ml,
    stats,
)
from concrete_fhe_toolkit import math as fhe_math


# ---------- math ----------


def test_copysign_clear():
    copysign = fhe_math.make_copysign(-10, 10)
    for x, y in product(range(-10, 11), (-3, 0, 4)):
        expected = abs(x) if y >= 0 else -abs(x)
        assert int(copysign(x, y)) == expected


def test_saturating_arithmetic_clear():
    add = fhe_math.make_saturating_add(-5, 5)
    subtract = fhe_math.make_saturating_subtract(-5, 5)
    multiply = fhe_math.make_saturating_multiply(-5, 5)

    for left, right in product(range(-5, 6), repeat=2):
        assert int(add(left, right)) == max(-5, min(left + right, 5))
        assert int(subtract(left, right)) == max(-5, min(left - right, 5))
        assert int(multiply(left, right)) == max(-5, min(left * right, 5))


def test_round_to_multiple_clear():
    quantize = fhe_math.make_round_to_multiple(0, 100, 25)
    assert int(quantize(30)) == 25
    assert int(quantize(40)) == 50
    assert int(quantize(0)) == 0
    assert int(quantize(100)) == 100

    floor_quantize = fhe_math.make_round_to_multiple(0, 100, 25, rounding="floor")
    assert int(floor_quantize(49)) == 25


def test_modf_clear():
    modf = fhe_math.make_modf(-40, 40, scale=10)
    fractional, integer = modf(-37)
    assert (int(fractional), int(integer)) == (-7, -3)
    fractional, integer = modf(26)
    assert (int(fractional), int(integer)) == (6, 2)


def test_fixed_point_multiply_clear():
    multiply = fhe_math.make_fixed_point_multiply(scale=10)
    assert int(multiply(15, 20)) == 30  # 1.5 * 2.0 = 3.0
    nearest = fhe_math.make_fixed_point_multiply(scale=10, rounding="nearest")
    assert int(nearest(15, 25)) == 38  # 1.5 * 2.5 = 3.75 -> 3.8


def test_fixed_point_codecs():
    assert fhe_math.encode_fixed_point(3.14, 100) == 314
    assert fhe_math.decode_fixed_point(314, 100) == 3.14


def test_powmod_clear():
    powmod = fhe_math.make_powmod(3, 7, 10)
    for exponent in range(11):
        assert int(powmod(exponent)) == pow(3, exponent, 7)


def test_atan2_clear():
    atan2 = fhe_math.make_atan2(-20, 20, input_scale=10, output_scale=100)
    for y, x in ((10, 10), (-10, 10), (0, -10), (20, -5)):
        expected = round(math.atan2(y / 10, x / 10) * 100)
        assert int(atan2(y, x)) == expected


def test_gamma_lgamma_clear():
    gamma = fhe_math.make_gamma(1, 50, input_scale=10, output_scale=100)
    assert int(gamma(50)) == 2400  # gamma(5.0) = 24
    assert int(gamma(1)) == round(math.gamma(0.1) * 100)

    lgamma = fhe_math.make_lgamma(1, 100, input_scale=10, output_scale=100)
    assert int(lgamma(100)) == round(math.lgamma(10.0) * 100)


def test_log_with_base():
    log2 = fhe_math.make_log(1, 100, input_scale=1, output_scale=100, base=2)
    assert int(log2(8)) == 300
    with pytest.raises(ValueError):
        fhe_math.make_log(1, 10, base=1)


# ---------- bits ----------


def test_bit_shifts_and_rotates():
    bits = (1, 0, 1, 0)  # value 5
    assert fhe_math.shift_left_bits(bits, 1) == (0, 1, 0, 1)  # 10
    assert fhe_math.shift_right_bits(bits, 1) == (0, 1, 0, 0)  # 2
    assert fhe_math.shift_right_bits((0, 1, 0, 1), 1, arithmetic=True) == (1, 0, 1, 1)
    assert fhe_math.rotate_left_bits((1, 1, 0, 0), 1) == (0, 1, 1, 0)
    assert fhe_math.rotate_right_bits((0, 1, 1, 0), 1) == (1, 1, 0, 0)


def test_popcount_parity_bit_length():
    assert int(fhe_math.popcount_bits((1, 1, 0, 1))) == 3
    assert int(fhe_math.parity_bits((1, 1, 0, 1))) == 1
    assert int(fhe_math.bit_length_bits((0, 0, 1, 0))) == 3
    assert int(fhe_math.bit_length_bits((0, 0, 0, 0))) == 0
    assert int(fhe_math.bit_length_bits((1, 0, 0, 1))) == 4


def test_unsigned_compare_bits():
    for left, right in product(range(16), repeat=2):
        left_bits = fhe_math.unsigned_to_bits(left, 4)
        right_bits = fhe_math.unsigned_to_bits(right, 4)
        is_less, is_equal = fhe_math.unsigned_compare_bits(left_bits, right_bits)
        assert int(is_less) == int(left < right)
        assert int(is_equal) == int(left == right)


def test_unsigned_mod_clear():
    modulo = fhe_math.make_unsigned_mod(4, 3, zero_result=7)
    for numerator, denominator in product(range(16), range(8)):
        expected = 7 if denominator == 0 else numerator % denominator
        assert int(modulo(numerator, denominator)) == expected


# ---------- arrays ----------


def test_oblivious_array_access():
    assert int(array_index([4, 7, 9], 2)) == 9
    assert [int(v) for v in array_set([4, 7, 9], 1, 5)] == [4, 5, 9]

    assert int(array_index_of([4, 7, 9, 7], 7)) == 1
    assert int(array_index_of([1, 2], 5)) == 2
    assert int(array_index_of([1, 2], 5, missing_result=-1)) == -1


def test_array_utilities():
    assert [int(v) for v in array_cumsum([1, 2, 3])] == [1, 3, 6]
    assert array_reverse([1, 2, 3]) == [3, 2, 1]
    assert array_concat([1, 2], [3], [4, 5]) == [1, 2, 3, 4, 5]


def test_top_k_clear():
    values = np.array([5, 17, 3, 9, 17], dtype=np.int64)
    top3 = make_top_k(5, 3, 0, 20)(values)
    assert [int(v) for v in top3] == [17, 17, 9]

    bottom2 = make_top_k(5, 2, 0, 20, largest=False)(values)
    assert [int(v) for v in bottom2] == [3, 5]

    with pytest.raises(ValueError):
        make_top_k(3, 4, 0, 20)


# ---------- stats ----------


def test_stats_order_statistics():
    values = [5, 1, 9, 3]
    assert int(stats.array_median(values, 0, 10)) == 4
    assert int(stats.array_percentile(values, 0, 0, 10)) == 1
    assert int(stats.array_percentile(values, 100, 0, 10)) == 9

    odd_like = [2, 4, 4, 4, 5, 5, 7, 9]
    assert int(stats.array_std(odd_like, 0, 10)) == 2


def test_stats_histogram_and_mode():
    counts = stats.array_histogram([1, 2, 2, 0], 0, 3)
    assert [int(c) for c in counts] == [1, 1, 2, 0]
    assert int(stats.array_mode([1, 2, 2, 0], 0, 3)) == 2


def test_stats_covariance_and_normalize():
    assert int(stats.array_covariance([1, 2, 3, 4], [2, 4, 6, 8])) == 2
    with pytest.raises(ValueError):
        stats.array_covariance([1, 2], [1])

    assert [int(v) for v in stats.array_normalize([1, 2, 3], 2, 10)] == [-10, 0, 10]


def test_ml_stats_backwards_compat():
    assert int(ml.array_mean([1, 2, 3, 4])) == 2
    assert int(stats.array_mean([1, 2, 3, 4])) == 2


# ---------- ml ----------


TREE_A = {"feature": 0, "threshold": 5, "left": 1, "right": 0}
TREE_B = {"feature": 1, "threshold": 3, "left": 1, "right": 0}
TREE_C = {"feature": 0, "threshold": 8, "left": 1, "right": 0}


def test_random_forest_inference_clear():
    forest = [TREE_A, TREE_B, TREE_C]
    assert int(ml.random_forest_inference([6, 4], forest)) == 1  # votes 1,1,0
    assert int(ml.random_forest_inference([2, 1], forest)) == 0  # votes 0,0,0
    with pytest.raises(ValueError):
        ml.random_forest_inference([1, 2], [])


def test_mlp_inference_clear():
    layers = [
        ([[1, 0], [0, 1]], [0, 0]),
        ([[1, 1]], [1]),
    ]
    assert [int(v) for v in ml.mlp_inference([3, -2], layers)] == [4]
    with pytest.raises(ValueError):
        ml.mlp_inference([1], [([[1]], [1, 2])])


def test_nearest_centroid_inference_clear():
    centroids = [[0, 0], [10, 10]]
    assert int(ml.nearest_centroid_inference([1, 2], centroids, max_distance=300)) == 0
    assert (
        int(
            ml.nearest_centroid_inference(
                [9, 8], centroids, [5, 9], max_distance=300
            )
        )
        == 9
    )


def test_naive_bayes_inference_clear():
    log_prob_tables = [
        [[10, 0, 0], [10, 0, 0]],
        [[0, 0, 10], [0, 0, 10]],
    ]
    priors = [0, 0]
    assert int(ml.naive_bayes_inference([0, 0], log_prob_tables, priors)) == 0
    assert int(ml.naive_bayes_inference([2, 2], log_prob_tables, priors)) == 1


def test_argmax_inference_clear():
    assert int(ml.argmax_inference([3, 9, 5], 0, 10)) == 1


def test_precision_recall_f1_clear():
    y_preds = [1, 1, 0, 0]
    y_trues = [1, 0, 1, 0]
    assert int(ml.precision_score(y_preds, y_trues)) == 50
    assert int(ml.recall_score(y_preds, y_trues)) == 50
    assert int(ml.f1_score(y_preds, y_trues)) == 50

    perfect = [1, 0, 1]
    assert int(ml.precision_score(perfect, perfect)) == 100
    assert int(ml.f1_score(perfect, perfect)) == 100

    all_negative = [0, 0, 0]
    assert int(ml.precision_score(all_negative, [0, 1, 0])) == 0
    assert int(ml.f1_score(all_negative, [0, 1, 0])) == 0


# ---------- compile / simulate ----------


def test_saturating_add_compiles_and_simulates():
    circuit = fhe_math.saturating_add.compile(-5, 5)
    assert int(circuit.simulate(4, 3)) == 5
    assert int(circuit.simulate(-4, -3)) == -5
    assert int(circuit.simulate(2, 1)) == 3


def test_copysign_compiles_and_simulates():
    circuit = fhe_math.copysign.compile(-7, 7)
    assert int(circuit.simulate(5, -2)) == -5
    assert int(circuit.simulate(-5, 3)) == 5


def test_modf_compiles_and_simulates():
    circuit = fhe_math.modf.compile(-40, 40, scale=10)
    fractional, integer = circuit.simulate(-37)
    assert (int(fractional), int(integer)) == (-7, -3)


def test_unsigned_mod_compiles_and_simulates():
    circuit = fhe_math.unsigned_mod.compile(4, 3)
    assert int(circuit.simulate(13, 5)) == 3
    assert int(circuit.simulate(13, 0)) == 0


def test_atan2_compiles_and_simulates():
    circuit = fhe_math.atan2.compile(-6, 6, input_scale=2, output_scale=10)
    assert int(circuit.simulate(2, 2)) == round(math.atan2(1, 1) * 10)
    assert int(circuit.simulate(0, -2)) == round(math.pi * 10)


def test_round_to_multiple_compiles_and_simulates():
    circuit = fhe_math.round_to_multiple.compile(0, 50, 10)
    assert int(circuit.simulate(37)) == 40
    assert int(circuit.simulate(12)) == 10


def test_top_k_compiles_and_simulates():
    circuit = compile_top_k(4, 2, 0, 10)
    result = circuit.simulate(np.array([3, 9, 2, 7], dtype=np.int64))
    assert [int(v) for v in result] == [9, 7]
