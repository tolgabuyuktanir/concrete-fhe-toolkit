"""Helpers for scaled-integer fixed-point values."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Literal, Optional

from .._compat import fhe

from .._utils import compile_function, validate_bounds, validate_integer
from ._lookup import (
    UnaryFunction,
    check_lookup_cost,
    compile_unary_lookup,
    make_unary_lookup,
    unary_values,
)

RoundingMode = Literal["floor", "ceil", "trunc", "nearest"]


def _validate_scale(name: str, value: int) -> int:
    return validate_integer(name, value, minimum=1)


def _round_fraction(value: Fraction, mode: RoundingMode) -> int:
    if mode == "floor":
        return value.numerator // value.denominator
    if mode == "ceil":
        return -((-value.numerator) // value.denominator)
    if mode == "trunc":
        return int(value)
    if mode == "nearest":
        return round(value)
    raise ValueError("rounding must be 'floor', 'ceil', 'trunc', or 'nearest'")


def _scaled_values(
    min_input: int,
    max_input: int,
    scale: int,
    mode: RoundingMode,
) -> list[int]:
    input_minimum, input_maximum = validate_bounds(min_input, max_input)
    normalized_scale = _validate_scale("scale", scale)
    return unary_values(
        lambda value: _round_fraction(Fraction(value, normalized_scale), mode),
        input_minimum,
        input_maximum,
    )


def make_floor(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
) -> UnaryFunction:
    """Create floor(value / scale) for encrypted scaled integers."""
    values = _scaled_values(min_input, max_input, scale, "floor")
    return make_unary_lookup(values, min_input)


def compile_floor(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile floor(value / scale) for encrypted scaled integers."""
    values = _scaled_values(min_input, max_input, scale, "floor")
    return compile_unary_lookup(
        "compile_floor",
        values,
        min_input,
        max_input,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_ceil(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
) -> UnaryFunction:
    """Create ceil(value / scale) for encrypted scaled integers."""
    values = _scaled_values(min_input, max_input, scale, "ceil")
    return make_unary_lookup(values, min_input)


def compile_ceil(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile ceil(value / scale) for encrypted scaled integers."""
    values = _scaled_values(min_input, max_input, scale, "ceil")
    return compile_unary_lookup(
        "compile_ceil",
        values,
        min_input,
        max_input,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_trunc(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
) -> UnaryFunction:
    """Create trunc(value / scale) toward zero for encrypted scaled integers."""
    values = _scaled_values(min_input, max_input, scale, "trunc")
    return make_unary_lookup(values, min_input)


def compile_trunc(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile trunc(value / scale) toward zero for encrypted scaled integers."""
    values = _scaled_values(min_input, max_input, scale, "trunc")
    return compile_unary_lookup(
        "compile_trunc",
        values,
        min_input,
        max_input,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_round(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
) -> UnaryFunction:
    """Create round(value / scale) using Python's ties-to-even rule."""
    values = _scaled_values(min_input, max_input, scale, "nearest")
    return make_unary_lookup(values, min_input)


def compile_round(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile round(value / scale) using Python's ties-to-even rule."""
    values = _scaled_values(min_input, max_input, scale, "nearest")
    return compile_unary_lookup(
        "compile_round",
        values,
        min_input,
        max_input,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_floor_ceil(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
) -> UnaryFunction:
    """Create a function returning both floor and ceil for a scaled integer."""
    floor_values = _scaled_values(min_input, max_input, scale, "floor")
    ceil_values = _scaled_values(min_input, max_input, scale, "ceil")
    floor_lookup = make_unary_lookup(floor_values, min_input)
    ceil_lookup = make_unary_lookup(ceil_values, min_input)

    def floor_and_ceil(value: Any) -> Any:
        return floor_lookup(value), ceil_lookup(value)

    return floor_and_ceil


def compile_floor_ceil(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a circuit returning both floor and ceil for a scaled integer."""
    floor_values = _scaled_values(min_input, max_input, scale, "floor")
    ceil_values = _scaled_values(min_input, max_input, scale, "ceil")
    check_lookup_cost(
        "compile_floor_ceil floor",
        floor_values,
        allow_large_lookup=allow_large_lookup,
    )
    check_lookup_cost(
        "compile_floor_ceil ceil",
        ceil_values,
        allow_large_lookup=allow_large_lookup,
    )
    function = make_floor_ceil(min_input, max_input, scale=scale)
    minimum, maximum = validate_bounds(min_input, max_input)
    return compile_function(
        function,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def make_rescale(
    min_input: int,
    max_input: int,
    *,
    input_scale: int,
    output_scale: int,
    rounding: RoundingMode = "nearest",
) -> UnaryFunction:
    """Create rescaling from one fixed-point scale to another."""
    input_minimum, input_maximum = validate_bounds(min_input, max_input)
    source = _validate_scale("input_scale", input_scale)
    target = _validate_scale("output_scale", output_scale)
    values = unary_values(
        lambda value: _round_fraction(Fraction(value * target, source), rounding),
        input_minimum,
        input_maximum,
    )
    return make_unary_lookup(values, input_minimum)


def compile_rescale(
    min_input: int,
    max_input: int,
    *,
    input_scale: int,
    output_scale: int,
    rounding: RoundingMode = "nearest",
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile rescaling from one fixed-point scale to another."""
    input_minimum, input_maximum = validate_bounds(min_input, max_input)
    source = _validate_scale("input_scale", input_scale)
    target = _validate_scale("output_scale", output_scale)
    values = unary_values(
        lambda value: _round_fraction(Fraction(value * target, source), rounding),
        input_minimum,
        input_maximum,
    )
    return compile_unary_lookup(
        "compile_rescale",
        values,
        input_minimum,
        input_maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_round_to_multiple(
    min_input: int,
    max_input: int,
    step: int,
    *,
    rounding: RoundingMode = "nearest",
) -> UnaryFunction:
    """Create quantization of an encrypted integer to the nearest multiple of step."""
    minimum, maximum = validate_bounds(min_input, max_input)
    normalized_step = validate_integer("step", step)
    if normalized_step < 1:
        raise ValueError("step must be at least 1")
    values = unary_values(
        lambda value: _round_fraction(Fraction(value, normalized_step), rounding)
        * normalized_step,
        minimum,
        maximum,
    )
    return make_unary_lookup(values, minimum)


def compile_round_to_multiple(
    min_input: int,
    max_input: int,
    step: int,
    *,
    rounding: RoundingMode = "nearest",
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile quantization of an encrypted integer to the nearest multiple of step."""
    minimum, maximum = validate_bounds(min_input, max_input)
    normalized_step = validate_integer("step", step)
    if normalized_step < 1:
        raise ValueError("step must be at least 1")
    values = unary_values(
        lambda value: _round_fraction(Fraction(value, normalized_step), rounding)
        * normalized_step,
        minimum,
        maximum,
    )
    return compile_unary_lookup(
        "compile_round_to_multiple",
        values,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_modf(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
) -> UnaryFunction:
    """Create a function returning (fractional_part, integer_part) like math.modf.

    The integer part truncates toward zero; the fractional part keeps the
    input's sign and stays at the input scale (e.g. -37 with scale=10
    yields (-7, -3)).
    """
    minimum, maximum = validate_bounds(min_input, max_input)
    normalized_scale = _validate_scale("scale", scale)
    integer_values = _scaled_values(minimum, maximum, normalized_scale, "trunc")
    fractional_values = [
        value - integer * normalized_scale
        for value, integer in zip(range(minimum, maximum + 1), integer_values)
    ]
    integer_lookup = make_unary_lookup(integer_values, minimum)
    fractional_lookup = make_unary_lookup(fractional_values, minimum)

    def modf(value: Any) -> Any:
        return fractional_lookup(value), integer_lookup(value)

    return modf


def compile_modf(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a circuit returning (fractional_part, integer_part) like math.modf."""
    minimum, maximum = validate_bounds(min_input, max_input)
    normalized_scale = _validate_scale("scale", scale)
    integer_values = _scaled_values(minimum, maximum, normalized_scale, "trunc")
    fractional_values = [
        value - integer * normalized_scale
        for value, integer in zip(range(minimum, maximum + 1), integer_values)
    ]
    check_lookup_cost(
        "compile_modf integer",
        integer_values,
        allow_large_lookup=allow_large_lookup,
    )
    check_lookup_cost(
        "compile_modf fractional",
        fractional_values,
        allow_large_lookup=allow_large_lookup,
    )
    function = make_modf(minimum, maximum, scale=normalized_scale)
    # The fractional output is not monotonic, so the inputset must cover the
    # full domain for Concrete to infer the correct output range.
    return compile_function(
        function,
        {"value": "encrypted"},
        list(range(minimum, maximum + 1)),
        configuration,
    )


def make_fixed_point_multiply(
    *,
    scale: int = 10,
    rounding: RoundingMode = "floor",
) -> Any:
    """Create multiplication of two scaled values with the product rescaled back.

    Computes ``(left * right) / scale`` so that two inputs at the same
    fixed-point scale produce an output at that scale. This is the missing
    primitive for chaining fixed-point math without scale growth.
    """
    normalized_scale = _validate_scale("scale", scale)
    if rounding == "floor":
        def multiply_floor(left: Any, right: Any) -> Any:
            return (left * right) // normalized_scale

        return multiply_floor
    if rounding == "nearest":
        offset = normalized_scale // 2

        def multiply_nearest(left: Any, right: Any) -> Any:
            return (left * right + offset) // normalized_scale

        return multiply_nearest
    raise ValueError("rounding must be 'floor' or 'nearest'")


def compile_fixed_point_multiply(
    min_input: int,
    max_input: int,
    *,
    scale: int = 10,
    rounding: RoundingMode = "floor",
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile rescaled multiplication of two scaled encrypted values."""
    minimum, maximum = validate_bounds(min_input, max_input)
    function = make_fixed_point_multiply(scale=scale, rounding=rounding)
    inputset = [
        (minimum, minimum),
        (minimum, maximum),
        (maximum, minimum),
        (maximum, maximum),
    ]
    return compile_function(
        function,
        {"left": "encrypted", "right": "encrypted"},
        inputset,
        configuration,
    )


def encode_fixed_point(value: float, scale: int) -> int:
    """Encode a clear real value as a scaled integer for circuit inputs."""
    normalized_scale = _validate_scale("scale", scale)
    return int(round(value * normalized_scale))


def decode_fixed_point(value: Any, scale: int) -> float:
    """Decode a decrypted scaled integer back to its real value."""
    normalized_scale = _validate_scale("scale", scale)
    return value / normalized_scale
