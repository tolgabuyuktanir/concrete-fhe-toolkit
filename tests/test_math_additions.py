"""Tests for the math additions: select, abs_diff, number theory, inverse trig."""

import math
from itertools import product

import pytest

from concrete_fhe_toolkit import math as fhe_math


def test_select_clear():
    for control in (0, 1):
        for when_true, when_false in product(range(-3, 4), repeat=2):
            expected = when_true if control else when_false
            assert int(fhe_math.select(control, when_true, when_false)) == expected


def test_abs_diff_clear():
    abs_diff = fhe_math.make_abs_diff(-5, 5)
    for left, right in product(range(-5, 6), repeat=2):
        assert int(abs_diff(left, right)) == abs(left - right)


def test_totient_clear():
    totient = fhe_math.make_totient(0, 20)
    expected = {0: 0, 1: 1, 2: 1, 6: 2, 9: 6, 10: 4, 12: 4, 17: 16, 20: 8}
    for value, phi in expected.items():
        assert int(totient(value)) == phi


def test_next_prime_clear():
    next_prime = fhe_math.make_next_prime(0, 30)
    expected = {0: 2, 1: 2, 2: 3, 3: 5, 8: 11, 13: 17, 23: 29, 30: 31}
    for value, prime in expected.items():
        assert int(next_prime(value)) == prime


def test_mod_inverse_clear():
    mod_inverse = fhe_math.make_mod_inverse(0, 12, invalid_result=99)
    for value, modulus in product(range(13), repeat=2):
        result = int(mod_inverse(value, modulus))
        if modulus <= 1 or math.gcd(value, modulus) != 1:
            assert result == 99
        else:
            assert result == pow(value, -1, modulus)
            assert (result * value) % modulus == 1


def test_hypot_clear():
    hypot = fhe_math.make_hypot(0, 12)
    for left, right in product(range(13), repeat=2):
        assert int(hypot(left, right)) == round(math.hypot(left, right))


def test_inverse_trig_clear():
    asin = fhe_math.make_asin(-100, 100, input_scale=100, output_scale=1000)
    acos = fhe_math.make_acos(-100, 100, input_scale=100, output_scale=1000)
    atan = fhe_math.make_atan(-300, 300, input_scale=100, output_scale=1000)

    for encoded in range(-100, 101, 25):
        real = encoded / 100
        assert int(asin(encoded)) == round(math.asin(real) * 1000)
        assert int(acos(encoded)) == round(math.acos(real) * 1000)

    for encoded in range(-300, 301, 75):
        real = encoded / 100
        assert int(atan(encoded)) == round(math.atan(real) * 1000)


def test_asin_domain_handling():
    with pytest.raises(ValueError):
        fhe_math.make_asin(-200, 200, input_scale=100, output_scale=1000)

    asin = fhe_math.make_asin(
        -200,
        200,
        input_scale=100,
        output_scale=1000,
        invalid_result=-9999,
    )
    assert int(asin(150)) == -9999
    assert int(asin(100)) == round(math.asin(1.0) * 1000)


def test_cbrt_degrees_radians_clear():
    cbrt = fhe_math.make_cbrt(-1000, 1000, input_scale=1, output_scale=100)
    assert int(cbrt(27)) == 300
    assert int(cbrt(-27)) == -300
    assert int(cbrt(1000)) == 1000
    assert int(cbrt(2)) == round((2 ** (1 / 3)) * 100)

    degrees = fhe_math.make_degrees(-628, 628, input_scale=100, output_scale=1)
    assert int(degrees(314)) == round(math.degrees(3.14))
    assert int(degrees(-157)) == round(math.degrees(-1.57))

    radians = fhe_math.make_radians(-360, 360, input_scale=1, output_scale=100)
    assert int(radians(180)) == round(math.pi * 100)
    assert int(radians(-90)) == -round(math.pi / 2 * 100)


def test_select_compiles_and_simulates():
    circuit = fhe_math.compile_select(-7, 7)

    assert int(circuit.simulate(1, 5, -3)) == 5
    assert int(circuit.simulate(0, 5, -3)) == -3
    assert int(circuit.simulate(1, -7, 7)) == -7


def test_abs_diff_compiles_and_simulates():
    circuit = fhe_math.abs_diff.compile(-5, 5)

    assert int(circuit.simulate(3, -4)) == 7
    assert int(circuit.simulate(-5, 5)) == 10
    assert int(circuit.simulate(2, 2)) == 0


def test_hypot_compiles_and_simulates():
    circuit = fhe_math.hypot.compile(0, 8)

    assert int(circuit.simulate(3, 4)) == 5
    assert int(circuit.simulate(0, 7)) == 7
    assert int(circuit.simulate(1, 1)) == 1


def test_mod_inverse_compiles_and_simulates():
    circuit = fhe_math.mod_inverse.compile(0, 10, invalid_result=0)

    assert int(circuit.simulate(3, 7)) == 5
    assert int(circuit.simulate(2, 4)) == 0
    assert int(circuit.simulate(9, 10)) == 9


def test_atan_compiles_and_simulates():
    circuit = fhe_math.atan.compile(-200, 200, input_scale=100, output_scale=100)

    assert int(circuit.simulate(100)) == round(math.atan(1.0) * 100)
    assert int(circuit.simulate(0)) == 0
    assert int(circuit.simulate(-200)) == round(math.atan(-2.0) * 100)


def test_totient_and_next_prime_compile_and_simulate():
    totient = fhe_math.totient.compile(0, 30)
    assert int(totient.simulate(12)) == 4
    assert int(totient.simulate(29)) == 28

    next_prime = fhe_math.next_prime.compile(0, 30)
    assert int(next_prime.simulate(24)) == 29
    assert int(next_prime.simulate(0)) == 2
