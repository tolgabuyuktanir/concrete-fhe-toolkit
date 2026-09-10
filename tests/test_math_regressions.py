"""Regression cases for operand widths and matrix arithmetic."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from concrete import fhe

from concrete_fhe_toolkit import math as fm
from concrete_fhe_toolkit.ml import matrix as mm


@pytest.mark.parametrize("n_width,d_width", list(product(range(1, 6), repeat=2)))
def test_division_and_modulo_all_small_widths(n_width, d_width):
    divide = fm.make_unsigned_floor_divide(n_width, d_width)
    modulo = fm.make_unsigned_mod(n_width, d_width)
    for n, d in product(range(1 << n_width), range(1 << d_width)):
        assert int(divide(n, d)) == (n // d if d else 0)
        assert int(modulo(n, d)) == (n % d if d else 0)


def test_wide_denominator_fractional_and_zero_fallback():
    divide = fm.make_fixed_point_divide(2, 6, fractional_bits=1, zero_result=7)
    modulo = fm.make_unsigned_mod(2, 6, zero_result=63)
    for n, d in product(range(4), range(64)):
        assert int(divide(n, d)) == ((n << 1) // d if d else 7)
        assert int(modulo(n, d)) == (n % d if d else 63)


def test_wide_denominator_compiles_and_simulates():
    config = fhe.Configuration(p_error=2**-40)
    divide = fm.compile_unsigned_floor_divide(3, 5, configuration=config)
    modulo = fm.compile_unsigned_mod(3, 5, configuration=config)
    for n, d in [(7, 16), (7, 17), (7, 31), (7, 3), (0, 16), (7, 0)]:
        assert int(divide.simulate(n, d)) == (n // d if d else 0)
        assert int(modulo.simulate(n, d)) == (n % d if d else 0)


def _covariance_reference(data):
    n = len(data)
    columns = list(zip(*data))
    means = [sum(Fraction(v) for v in col) / n for col in columns]
    return [
        [
            sum(
                (Fraction(x) - means[i]) * (Fraction(y) - means[j])
                for x, y in zip(columns[i], columns[j])
            )
            // (n - 1)
            for j in range(len(columns))
        ]
        for i in range(len(columns))
    ]


def test_covariance_floors_only_final_result():
    for values in product(range(-2, 3), repeat=4):
        data = [list(values[:2]), list(values[2:])]
        actual = mm.covariance_matrix(data)
        assert actual == _covariance_reference(data)
        assert actual[0][1] == actual[1][0]
    assert mm.covariance_matrix([[1, 1], [2, 3]]) == [[0, 1], [1, 2]]


def test_covariance_compiles_and_simulates():
    def covariance(x):
        return fhe.array(mm.covariance_matrix(x))

    samples = [np.array(v).reshape(2, 2) for v in product(range(-1, 2), repeat=4)]
    circuit = fhe.Compiler(covariance, {"x": "encrypted"}).compile(
        samples, configuration=fhe.Configuration(p_error=2**-40)
    )
    for sample in [np.array([[0, 0], [1, -1]]), np.array([[-1, 0], [1, 1]])]:
        np.testing.assert_array_equal(circuit.simulate(sample), _covariance_reference(sample))


@pytest.mark.parametrize(
    "operation", [mm.matrix_add, mm.matrix_subtract, mm.matrix_elementwise_multiply]
)
@pytest.mark.parametrize(
    "a,b", [([[1]], [[2], [3]]), ([[1], [2]], [[3]]), ([[1, 2], [3]], [[4, 5], [6]]), ([], [[1]])]
)
def test_elementwise_matrix_shape_validation(operation, a, b):
    with pytest.raises(ValueError):
        operation(a, b)


def test_empty_and_ragged_matrices():
    for operation in [mm.matrix_add, mm.matrix_subtract, mm.matrix_elementwise_multiply]:
        assert operation([], []) == []
    with pytest.raises(ValueError):
        mm.matrix_transpose([[1, 2], [3]])
    with pytest.raises(ValueError):
        mm.matrix_multiply([[1, 2]], [[1, 2], [3]])
    for data in [[], [[1]], [[], []], [[1, 2], [3]]]:
        with pytest.raises(ValueError):
            mm.covariance_matrix(data)
