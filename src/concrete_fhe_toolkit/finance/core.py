"""Bounded finance helpers for encrypted integer amounts.

Money convention: encrypted amounts are plain integers in the smallest unit
you choose (for example cents/kurus). Rates are public floats with at most
two decimal digits and are applied through the fixed ``RATE_SCALE`` factor,
so every rate-scaled result returned by this module is ``RATE_SCALE`` times
larger than the real value. Use :func:`return_actual_value` after decryption
to recover the real value.
"""

from .._utils import client_side_helper

from typing import Any

RATE_SCALE = 100


def _scaled_rate(rate: float) -> int:
    """Convert a public float rate into an exact integer at RATE_SCALE.
    
    Args:
        rate (float): The public float rate.
        
    Returns:
        int: The exact integer representation of the rate at RATE_SCALE.
        
    Raises:
        ValueError: If the rate has more than two decimal digits.
        
    Example:
        ```python
        scaled = _scaled_rate(0.05)
        ```
    """
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
    
    Args:
        amount (Any): The encrypted amount.
        rate (float): The public rate to multiply by.
        
    Returns:
        Any: The encrypted result scaled by RATE_SCALE.
    
    Example:
        ```python
        from concrete_fhe_toolkit.finance.core import apply_rate
        
        # Inside an FHE circuit (e.g. applying a 5% rate)
        # result = apply_rate(enc_amount, rate=0.05)
        ```
    """
    return amount * _scaled_rate(rate)


@client_side_helper
def return_actual_value(value: Any) -> float:
    """Decode a RATE_SCALE-scaled cleartext result back to its real value.
    
    Args:
        value (Any): The cleartext value scaled by RATE_SCALE.
        
    Returns:
        float: The actual real value.
    
    Example:
        ```python
        from concrete_fhe_toolkit.finance.core import return_actual_value
        
        # After decryption
        real_value = return_actual_value(decrypted_result)
        ```
    """
    return value / RATE_SCALE


def calculate_tax(amount: Any, rate: float) -> Any:
    """Calculate the tax amount at RATE_SCALE scaling.
    
    Args:
        amount (Any): The encrypted amount.
        rate (float): The tax rate.
        
    Returns:
        Any: The encrypted tax amount scaled by RATE_SCALE.
    
    Example:
        ```python
        from concrete_fhe_toolkit.finance.core import calculate_tax
        
        # Inside an FHE circuit (18% tax)
        # tax_amount = calculate_tax(enc_amount, rate=0.18)
        ```
    """
    return apply_rate(amount, rate)


def discount(amount: Any, rate: float) -> Any:
    """Calculate the discounted amount at RATE_SCALE scaling.
    
    Args:
        amount (Any): The encrypted original amount.
        rate (float): The discount rate.
        
    Returns:
        Any: The encrypted discounted amount scaled by RATE_SCALE.
    
    Example:
        ```python
        from concrete_fhe_toolkit.finance.core import discount
        
        # Inside an FHE circuit (20% discount)
        # final_price = discount(enc_amount, rate=0.20)
        ```
    """
    return (amount * RATE_SCALE) - apply_rate(amount, rate)


def simple_interest(amount: Any, rate: float, time_period: Any) -> Any:
    """Calculate simple interest (amount * rate * time) at RATE_SCALE scaling.
    
    Args:
        amount (Any): The encrypted principal amount.
        rate (float): The interest rate.
        time_period (Any): The encrypted or plaintext time period.
        
    Returns:
        Any: The encrypted simple interest scaled by RATE_SCALE.
    
    Example:
        ```python
        from concrete_fhe_toolkit.finance.core import simple_interest
        
        # Inside an FHE circuit (e.g. 5% interest over 3 years)
        # interest = simple_interest(enc_amount, rate=0.05, time_period=3)
        ```
    """
    return apply_rate(amount, rate) * time_period
