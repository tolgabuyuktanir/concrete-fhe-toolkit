"""Tests for the C-math-parity additions."""

import math
from itertools import product

import pytest

from concrete_fhe_toolkit import math as fhe_math


def test_inverse_hyperbolics_clear():
    asinh = fhe_math.make_asinh(-500, 500, input_scale=100, output_scale=100)
    for encoded in range(-500, 501, 125):
        assert int(asinh(encoded)) == round(math.asinh(encoded / 100) * 100)

    acosh = fhe_math.make_acosh(100, 1000, input_scale=100, output_scale=100)
    for encoded in (100, 200, 550, 1000):
        assert int(acosh(encoded)) == round(math.acosh(encoded / 100) * 100)

    atanh = fhe_math.make_atanh(-99, 99, input_scale=100, output_scale=100)
    for encoded in (-99, -50, 0, 50, 99):
        assert int(atanh(encoded)) == round(math.atanh(encoded / 100) * 100)


def test_inverse_hyperbolic_domains():
    with pytest.raises(ValueError):
        fhe_math.make_acosh(0, 200, input_scale=100, output_scale=100)
    acosh = fhe_math.make_acosh(
        0, 200, input_scale=100, output_scale=100, invalid_result=-1
    )
    assert int(acosh(50)) == -1

    with pytest.raises(ValueError):
        fhe_math.make_atanh(-100, 100, input_scale=100, output_scale=100)
    atanh = fhe_math.make_atanh(
        -100, 100, input_scale=100, output_scale=100, invalid_result=-999
    )
    assert int(atanh(100)) == -999


def test_exp2_clear():
    exp2 = fhe_math.make_exp2(-50, 50, input_scale=10, output_scale=100)
    for encoded in (-50, -10, 0, 10, 30, 50):
        assert int(exp2(encoded)) == round(2.0 ** (encoded / 10) * 100)


def test_fdim_clear():
    fdim = fhe_math.make_fdim(-6, 6)
    for left, right in product(range(-6, 7), repeat=2):
        assert int(fdim(left, right)) == max(left - right, 0)


def test_fma_clear():
    for left, right, addend in product(range(-3, 4), repeat=3):
        assert int(fhe_math.fma(left, right, addend)) == left * right + addend


def test_remainder_clear():
    remainder = fhe_math.make_remainder(-10, 10, -5, 5, zero_result=99)
    for numerator, denominator in product(range(-10, 11), range(-5, 6)):
        if denominator == 0:
            assert int(remainder(numerator, denominator)) == 99
        else:
            expected = int(math.remainder(numerator, denominator))
            assert int(remainder(numerator, denominator)) == expected


def test_ilogb_clear():
    ilogb = fhe_math.make_ilogb(-16, 16, invalid_result=-99)
    for value in range(-16, 17):
        if value == 0:
            assert int(ilogb(value)) == -99
        else:
            assert int(ilogb(value)) == int(math.floor(math.log2(abs(value))))


def test_ldexp_clear():
    times_eight = fhe_math.make_ldexp(3)
    assert int(times_eight(5)) == 40
    div_four = fhe_math.make_ldexp(-2)
    assert int(div_four(20)) == 5
    assert int(div_four(-20)) == -5
    assert fhe_math.make_scalbn is fhe_math.make_ldexp


def test_cmath_aliases():
    assert fhe_math.tgamma is fhe_math.gamma
    assert fhe_math.fabs is fhe_math.absolute
    assert fhe_math.fmax is fhe_math.maximum
    assert fhe_math.fmin is fhe_math.minimum
    assert fhe_math.scalbn is fhe_math.ldexp


def test_fdim_and_remainder_compile_and_simulate():
    fdim = fhe_math.fdim.compile(-5, 5)
    assert int(fdim.simulate(4, -2)) == 6
    assert int(fdim.simulate(-2, 4)) == 0

    remainder = fhe_math.remainder.compile(-8, 8, 1, 4)
    assert int(remainder.simulate(7, 4)) == -1  # round(7/4)=2 -> 7-8
    assert int(remainder.simulate(5, 2)) == 1


def test_fma_and_ldexp_compile_and_simulate():
    fma_circuit = fhe_math.compile_fma(-4, 4)
    assert int(fma_circuit.simulate(3, -2, 4)) == -2

    ldexp_circuit = fhe_math.ldexp.compile(-10, 10, 2)
    assert int(ldexp_circuit.simulate(-3)) == -12


def test_ilogb_and_exp2_compile_and_simulate():
    ilogb = fhe_math.ilogb.compile(1, 32)
    assert int(ilogb.simulate(32)) == 5
    assert int(ilogb.simulate(7)) == 2

    exp2 = fhe_math.exp2.compile(0, 60, input_scale=10, output_scale=10)
    assert int(exp2.simulate(30)) == 80  # 2^3 = 8.0
