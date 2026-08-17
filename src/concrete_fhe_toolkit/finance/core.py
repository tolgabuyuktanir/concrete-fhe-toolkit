from typing import Any

def apply_rate(amount: Any, rate: float) -> Any:
    """Applies a rate to an amount."""
    scale_factor = 100
    rate = int(round(rate,2) * scale_factor)
    product = amount * rate

    return product

def return_actual_value(value: Any) -> float:
    """Returns the actual value of a scaled value."""
    return value / 100

def calculate_tax(amount: Any, rate: float) -> Any:
    """Calculates the tax amount."""
    return apply_rate(amount,rate)

def discount(amount: Any, rate: float) -> Any:
    """Calculates the discount amount."""
    return (amount * 100) - apply_rate(amount, rate)

def simple_interest(amount: Any, rate: float, time_period: Any) -> Any:
    """Calculates simple interest."""
    return apply_rate(amount,rate) * time_period