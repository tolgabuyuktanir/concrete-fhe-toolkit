"""Bounded finance helpers for encrypted integer amounts.

Money convention: encrypted amounts are plain integers in the smallest unit
you choose (for example cents/kurus). Rates are public floats with at most
two decimal digits and are applied through the fixed ``RATE_SCALE`` factor,
so every rate-scaled result returned by this module is ``RATE_SCALE`` times
larger than the real value. Use :func:`return_actual_value` after decryption
to recover the real value.
"""

from typing import Any

RATE_SCALE = 100


def _scaled_rate(rate: float) -> int:
    """Convert a public float rate into an exact integer at RATE_SCALE."""
    scaled = round(rate * RATE_SCALE)
    if abs(rate * RATE_SCALE - scaled) > 1e-6:
        raise ValueError(
            "rate must have at most two decimal digits "
            f"(got {rate!r}; supported resolution is 1/{RATE_SCALE})"
        )
    return scaled


def apply_rate(amount: Any, rate: float) -> Any:
    """Multiply an encrypted amount by a public rate.

    The result is scaled by ``RATE_SCALE``; decode with
    :func:`return_actual_value` after decryption.
    """
    return amount * _scaled_rate(rate)


def return_actual_value(value: Any) -> float:
    """Decode a RATE_SCALE-scaled cleartext result back to its real value."""
    return value / RATE_SCALE


def calculate_tax(amount: Any, rate: float) -> Any:
    """Calculate the tax amount at RATE_SCALE scaling."""
    return apply_rate(amount, rate)


def discount(amount: Any, rate: float) -> Any:
    """Calculate the discounted amount at RATE_SCALE scaling."""
    return (amount * RATE_SCALE) - apply_rate(amount, rate)


def simple_interest(amount: Any, rate: float, time_period: Any) -> Any:
    """Calculate simple interest (amount * rate * time) at RATE_SCALE scaling."""
    return apply_rate(amount, rate) * time_period
