"""Activation helpers for encrypted ML circuits."""

from typing import Any, Optional

from .._compat import fhe

from concrete_fhe_toolkit._utils import compile_function, validate_bounds
from concrete_fhe_toolkit.math import greater_equal
from concrete_fhe_toolkit.math.basic import maximum


def relu(value: Any) -> Any:
    """Return max(0, value)."""
    return maximum(0, value)


def leaky_relu(value: Any, alpha: float = 0.01) -> Any:
    """Return value when positive, otherwise alpha * value truncated toward zero.

    ``alpha`` must be the reciprocal of a positive integer (for example 0.01,
    0.1, or 0.5) because the slope is applied with integer division.
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    divisor = round(1 / alpha)
    if abs(1 / alpha - divisor) > 1e-9:
        raise ValueError("alpha must be the reciprocal of a positive integer")
    # -((-value) // divisor) truncates toward zero, so small negative inputs
    # scale toward 0 instead of sticking at -1.
    scaled = -((-value) // divisor)
    return maximum(scaled, value)


def unit_step(value: Any) -> Any:
    """Return the Heaviside step of a number: 0 when value < 0, otherwise 1."""
    return greater_equal(value, 0)


def threshold_activation(value: Any, threshold: Any) -> Any:
    """Return 1 when value >= threshold, otherwise 0."""
    return greater_equal(value, threshold)


def compile_relu(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted ReLU."""
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
    alpha: float = 0.01,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted leaky ReLU."""
    minimum, maximum = validate_bounds(min_value, max_value)

    def bound_leaky_relu(value: Any) -> Any:
        return leaky_relu(value, alpha)

    return compile_function(
        bound_leaky_relu,
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
    """Compile encrypted unit step."""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        unit_step,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def compile_threshold_activation(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted threshold activation over two encrypted inputs."""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        threshold_activation,
        {"value": "encrypted", "threshold": "encrypted"},
        [
            (minimum, minimum),
            (minimum, maximum),
            (maximum, minimum),
            (maximum, maximum),
        ],
        configuration,
    )
