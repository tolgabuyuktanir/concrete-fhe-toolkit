from typing import Any, Optional

from concrete import fhe

from concrete_fhe_toolkit._utils import compile_function, validate_bounds
from concrete_fhe_toolkit.math.basic import greater, less, maximum
from concrete_fhe_toolkit.math import sigmoid, tanh
from concrete_fhe_toolkit.arithmetic import sign

def relu(value: Any):
    "return the relu result of a number"
    return maximum(0,value)

def leaky_relu(value: Any, alpha: Any=0.01):
    "return the leaky relu result of a number"
    divisor = int(1/alpha)
    return maximum(value//divisor,value)

def unit_step(value: Any) -> Any:
    "return the unit step function result of a number in interval[0-2](2 times the actual interval)"
    less_zero = less(value,0)
    greater_zero = greater(value,0)

    result = 1 + (greater_zero * 1) + (less_zero * -1)
    return result


def compile_relu(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted integer negation."""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        relu,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def compile_leaky_relu(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted integer negation."""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        leaky_relu,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def compile_unit_step(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted integer negation."""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        unit_step,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )

