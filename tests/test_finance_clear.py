"""Cleartext behavior tests for the finance subpackage."""

import pytest

from concrete_fhe_toolkit import finance


def test_apply_rate_is_exact_for_two_decimal_rates():
    # 0.29 * 100 is 28.999... in binary floating point; the helper must
    # still resolve it to exactly 29.
    assert int(finance.apply_rate(100, 0.29)) == 2900
    assert int(finance.apply_rate(100, 0.07)) == 700
    assert int(finance.apply_rate(250, 0.5)) == 12500

    assert finance.return_actual_value(2900) == 29.0


def test_apply_rate_rejects_unsupported_precision():
    with pytest.raises(ValueError):
        finance.apply_rate(100, 0.125)


def test_calculate_tax_and_discount():
    assert int(finance.calculate_tax(200, 0.18)) == 3600
    assert finance.return_actual_value(finance.calculate_tax(200, 0.18)) == 36.0

    discounted = finance.discount(200, 0.25)
    assert int(discounted) == 15000
    assert finance.return_actual_value(discounted) == 150.0


def test_simple_interest():
    interest = finance.simple_interest(100, 0.1, 3)
    assert int(interest) == 3000
    assert finance.return_actual_value(interest) == 30.0


def test_transfer():
    sender, receiver = finance.transfer(100, 50, 30)
    assert (int(sender), int(receiver)) == (70, 80)

    sender, receiver = finance.transfer(10, 50, 30)
    assert (int(sender), int(receiver)) == (10, 50)

    sender, receiver = finance.transfer(30, 0, 30)
    assert (int(sender), int(receiver)) == (0, 30)
