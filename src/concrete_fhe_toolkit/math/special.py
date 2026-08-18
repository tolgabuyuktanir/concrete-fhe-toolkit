"""Scaled fixed-point approximations of common transcendental functions."""

from __future__ import annotations

from typing import Callable, Literal, Optional
import math

from .._compat import fhe

from .._utils import validate_bounds, validate_integer
from ._lookup import (
    BinaryFunction,
    UnaryFunction,
    binary_values,
    compile_binary_lookup,
    compile_unary_lookup,
    make_binary_lookup,
    make_unary_lookup,
    unary_values,
)

AngleUnit = Literal["radians", "degrees"]


def _validate_scale(name: str, value: int) -> int:
    return validate_integer(name, value, minimum=1)


def _angle(value: float, angle_unit: AngleUnit) -> float:
    if angle_unit == "radians":
        return value
    if angle_unit == "degrees":
        return math.radians(value)
    raise ValueError("angle_unit must be 'radians' or 'degrees'")


def _scaled_values(
    name: str,
    function: Callable[[float], float],
    min_input: int,
    max_input: int,
    *,
    input_scale: int,
    output_scale: int,
    invalid_result: Optional[int] = None,
    domain: Optional[Callable[[float], bool]] = None,
) -> list[int]:
    minimum, maximum = validate_bounds(min_input, max_input)
    source = _validate_scale("input_scale", input_scale)
    target = _validate_scale("output_scale", output_scale)
    if invalid_result is not None:
        invalid = validate_integer("invalid_result", invalid_result)
    else:
        invalid = None

    def compute(encoded: int) -> int:
        real_input = encoded / source
        if domain is not None and not domain(real_input):
            if invalid is None:
                raise ValueError(
                    f"{name} domain excludes encoded input {encoded}; "
                    "pass invalid_result to define encrypted-domain behavior"
                )
            return invalid

        try:
            real_output = function(real_input)
        except ValueError:
            if invalid is None:
                raise
            return invalid
        except OverflowError as error:
            raise ValueError(f"{name} output overflowed for input {encoded}") from error

        if not math.isfinite(real_output):
            if invalid is None:
                raise ValueError(f"{name} output is not finite for input {encoded}")
            return invalid
        return round(real_output * target)

    return unary_values(compute, minimum, maximum)


def _make_scaled_unary(
    name: str,
    function: Callable[[float], float],
    min_input: int,
    max_input: int,
    *,
    input_scale: int,
    output_scale: int,
    invalid_result: Optional[int] = None,
    domain: Optional[Callable[[float], bool]] = None,
) -> UnaryFunction:
    values = _scaled_values(
        name,
        function,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=domain,
    )
    return make_unary_lookup(values, min_input)


def _compile_scaled_unary(
    name: str,
    function: Callable[[float], float],
    min_input: int,
    max_input: int,
    *,
    input_scale: int,
    output_scale: int,
    invalid_result: Optional[int],
    domain: Optional[Callable[[float], bool]],
    allow_large_lookup: bool,
    configuration: Optional[fhe.Configuration],
) -> fhe.Circuit:
    values = _scaled_values(
        name,
        function,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=domain,
    )
    return compile_unary_lookup(
        name,
        values,
        min_input,
        max_input,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_sin(
    min_input: int = 0,
    max_input: int = 628,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    angle_unit: AngleUnit = "radians",
) -> UnaryFunction:
    """Create scaled sin for encrypted fixed-point angles."""
    return _make_scaled_unary(
        "make_sin",
        lambda value: math.sin(_angle(value, angle_unit)),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_sin(
    min_input: int = 0,
    max_input: int = 628,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    angle_unit: AngleUnit = "radians",
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled sin for encrypted fixed-point angles."""
    return _compile_scaled_unary(
        "compile_sin",
        lambda value: math.sin(_angle(value, angle_unit)),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_cos(
    min_input: int = 0,
    max_input: int = 628,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    angle_unit: AngleUnit = "radians",
) -> UnaryFunction:
    """Create scaled cos for encrypted fixed-point angles."""
    return _make_scaled_unary(
        "make_cos",
        lambda value: math.cos(_angle(value, angle_unit)),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_cos(
    min_input: int = 0,
    max_input: int = 628,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    angle_unit: AngleUnit = "radians",
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled cos for encrypted fixed-point angles."""
    return _compile_scaled_unary(
        "compile_cos",
        lambda value: math.cos(_angle(value, angle_unit)),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_tan(
    min_input: int = -100,
    max_input: int = 100,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    angle_unit: AngleUnit = "radians",
) -> UnaryFunction:
    """Create scaled tan for encrypted fixed-point angles."""
    return _make_scaled_unary(
        "make_tan",
        lambda value: math.tan(_angle(value, angle_unit)),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_tan(
    min_input: int = -100,
    max_input: int = 100,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    angle_unit: AngleUnit = "radians",
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled tan for encrypted fixed-point angles."""
    return _compile_scaled_unary(
        "compile_tan",
        lambda value: math.tan(_angle(value, angle_unit)),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_exp(
    min_input: int = -50,
    max_input: int = 50,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled exp for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_exp",
        math.exp,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_exp(
    min_input: int = -50,
    max_input: int = 50,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled exp for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_exp",
        math.exp,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_expm1(
    min_input: int = -50,
    max_input: int = 50,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled expm1 for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_expm1",
        math.expm1,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_expm1(
    min_input: int = -50,
    max_input: int = 50,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled expm1 for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_expm1",
        math.expm1,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def _validate_log_base(base: Optional[float]):
    if base is None:
        return math.log
    if base <= 0 or base == 1:
        raise ValueError("base must be positive and different from 1")
    return lambda value: math.log(value, base)


def make_log(
    min_input: int = 1,
    max_input: int = 1000,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    invalid_result: Optional[int] = None,
    base: Optional[float] = None,
) -> UnaryFunction:
    """Create scaled log (natural by default, or any base); invalid_result handles x <= 0."""
    return _make_scaled_unary(
        "make_log",
        _validate_log_base(base),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value > 0,
    )


def compile_log(
    min_input: int = 1,
    max_input: int = 1000,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    invalid_result: Optional[int] = None,
    base: Optional[float] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled log (natural by default, or any base); invalid_result handles x <= 0."""
    return _compile_scaled_unary(
        "compile_log",
        _validate_log_base(base),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value > 0,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_log2(
    min_input: int = 1,
    max_input: int = 1024,
    *,
    input_scale: int = 1,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled log2; invalid_result handles x <= 0 if needed."""
    return _make_scaled_unary(
        "make_log2",
        math.log2,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value > 0,
    )


def compile_log2(
    min_input: int = 1,
    max_input: int = 1024,
    *,
    input_scale: int = 1,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled log2; invalid_result handles x <= 0 if needed."""
    return _compile_scaled_unary(
        "compile_log2",
        math.log2,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value > 0,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_log10(
    min_input: int = 1,
    max_input: int = 1000,
    *,
    input_scale: int = 1,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled log10; invalid_result handles x <= 0 if needed."""
    return _make_scaled_unary(
        "make_log10",
        math.log10,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value > 0,
    )


def compile_log10(
    min_input: int = 1,
    max_input: int = 1000,
    *,
    input_scale: int = 1,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled log10; invalid_result handles x <= 0 if needed."""
    return _compile_scaled_unary(
        "compile_log10",
        math.log10,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value > 0,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_log1p(
    min_input: int = 0,
    max_input: int = 1000,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled log1p; invalid_result handles x <= -1 if needed."""
    return _make_scaled_unary(
        "make_log1p",
        math.log1p,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value > -1,
    )


def compile_log1p(
    min_input: int = 0,
    max_input: int = 1000,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled log1p; invalid_result handles x <= -1 if needed."""
    return _compile_scaled_unary(
        "compile_log1p",
        math.log1p,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value > -1,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_sqrt(
    min_input: int = 0,
    max_input: int = 1000,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled square root; invalid_result handles x < 0 if needed."""
    return _make_scaled_unary(
        "make_sqrt",
        math.sqrt,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value >= 0,
    )


def compile_sqrt(
    min_input: int = 0,
    max_input: int = 1000,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled square root; invalid_result handles x < 0 if needed."""
    return _compile_scaled_unary(
        "compile_sqrt",
        math.sqrt,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value >= 0,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_erf(
    min_input: int = -30,
    max_input: int = 30,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled erf for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_erf",
        math.erf,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_erf(
    min_input: int = -30,
    max_input: int = 30,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled erf for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_erf",
        math.erf,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_erfc(
    min_input: int = -30,
    max_input: int = 30,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled erfc for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_erfc",
        math.erfc,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_erfc(
    min_input: int = -30,
    max_input: int = 30,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled erfc for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_erfc",
        math.erfc,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_tanh(
    min_input: int = -40,
    max_input: int = 40,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled tanh for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_tanh",
        math.tanh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_tanh(
    min_input: int = -40,
    max_input: int = 40,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled tanh for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_tanh",
        math.tanh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_sinh(
    min_input: int = -30,
    max_input: int = 30,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled sinh for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_sinh",
        math.sinh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_sinh(
    min_input: int = -30,
    max_input: int = 30,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled sinh for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_sinh",
        math.sinh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_cosh(
    min_input: int = -30,
    max_input: int = 30,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled cosh for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_cosh",
        math.cosh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_cosh(
    min_input: int = -30,
    max_input: int = 30,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled cosh for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_cosh",
        math.cosh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_sigmoid(
    min_input: int = -60,
    max_input: int = 60,
    *,
    input_scale: int = 10,
    output_scale: int = 1000,
) -> UnaryFunction:
    """Create scaled logistic sigmoid for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_sigmoid",
        lambda value: 1 / (1 + math.exp(-value)),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_sigmoid(
    min_input: int = -60,
    max_input: int = 60,
    *,
    input_scale: int = 10,
    output_scale: int = 1000,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled logistic sigmoid for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_sigmoid",
        lambda value: 1 / (1 + math.exp(-value)),
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def _cbrt(value: float) -> float:
    return math.copysign(abs(value) ** (1.0 / 3.0), value)


def make_asin(
    min_input: int = -100,
    max_input: int = 100,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled arcsine (radians); invalid_result handles |x| > 1 if needed."""
    return _make_scaled_unary(
        "make_asin",
        math.asin,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: -1 <= value <= 1,
    )


def compile_asin(
    min_input: int = -100,
    max_input: int = 100,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled arcsine (radians); invalid_result handles |x| > 1 if needed."""
    return _compile_scaled_unary(
        "compile_asin",
        math.asin,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: -1 <= value <= 1,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_acos(
    min_input: int = -100,
    max_input: int = 100,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled arccosine (radians); invalid_result handles |x| > 1 if needed."""
    return _make_scaled_unary(
        "make_acos",
        math.acos,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: -1 <= value <= 1,
    )


def compile_acos(
    min_input: int = -100,
    max_input: int = 100,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled arccosine (radians); invalid_result handles |x| > 1 if needed."""
    return _compile_scaled_unary(
        "compile_acos",
        math.acos,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: -1 <= value <= 1,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_atan(
    min_input: int = -500,
    max_input: int = 500,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
) -> UnaryFunction:
    """Create scaled arctangent (radians) for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_atan",
        math.atan,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_atan(
    min_input: int = -500,
    max_input: int = 500,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled arctangent (radians) for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_atan",
        math.atan,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_cbrt(
    min_input: int = -1000,
    max_input: int = 1000,
    *,
    input_scale: int = 1,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled cube root (sign-preserving) for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_cbrt",
        _cbrt,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_cbrt(
    min_input: int = -1000,
    max_input: int = 1000,
    *,
    input_scale: int = 1,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled cube root (sign-preserving) for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_cbrt",
        _cbrt,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_degrees(
    min_input: int = -628,
    max_input: int = 628,
    *,
    input_scale: int = 100,
    output_scale: int = 1,
) -> UnaryFunction:
    """Create scaled radians-to-degrees conversion for encrypted inputs."""
    return _make_scaled_unary(
        "make_degrees",
        math.degrees,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_degrees(
    min_input: int = -628,
    max_input: int = 628,
    *,
    input_scale: int = 100,
    output_scale: int = 1,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled radians-to-degrees conversion for encrypted inputs."""
    return _compile_scaled_unary(
        "compile_degrees",
        math.degrees,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_radians(
    min_input: int = -360,
    max_input: int = 360,
    *,
    input_scale: int = 1,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled degrees-to-radians conversion for encrypted inputs."""
    return _make_scaled_unary(
        "make_radians",
        math.radians,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_radians(
    min_input: int = -360,
    max_input: int = 360,
    *,
    input_scale: int = 1,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled degrees-to-radians conversion for encrypted inputs."""
    return _compile_scaled_unary(
        "compile_radians",
        math.radians,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def _gamma_domain(value: float) -> bool:
    return not (value <= 0 and float(value).is_integer())


def make_gamma(
    min_input: int = 1,
    max_input: int = 50,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled gamma; invalid_result handles the non-positive-integer poles."""
    return _make_scaled_unary(
        "make_gamma",
        math.gamma,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=_gamma_domain,
    )


def compile_gamma(
    min_input: int = 1,
    max_input: int = 50,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled gamma; invalid_result handles the non-positive-integer poles."""
    return _compile_scaled_unary(
        "compile_gamma",
        math.gamma,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=_gamma_domain,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_lgamma(
    min_input: int = 1,
    max_input: int = 100,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled log-gamma; invalid_result handles the non-positive-integer poles."""
    return _make_scaled_unary(
        "make_lgamma",
        math.lgamma,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=_gamma_domain,
    )


def compile_lgamma(
    min_input: int = 1,
    max_input: int = 100,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled log-gamma; invalid_result handles the non-positive-integer poles."""
    return _compile_scaled_unary(
        "compile_lgamma",
        math.lgamma,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=_gamma_domain,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def _atan2_values(
    min_input: int,
    max_input: int,
    input_scale: int,
    output_scale: int,
) -> list:
    source = _validate_scale("input_scale", input_scale)
    target = _validate_scale("output_scale", output_scale)
    return binary_values(
        lambda y, x: round(math.atan2(y / source, x / source) * target),
        min_input,
        max_input,
        min_input,
        max_input,
    )


def make_atan2(
    min_input: int = -100,
    max_input: int = 100,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
) -> BinaryFunction:
    """Create scaled quadrant-aware atan2(y, x) for encrypted fixed-point inputs."""
    minimum, maximum = validate_bounds(min_input, max_input)
    values = _atan2_values(minimum, maximum, input_scale, output_scale)
    return make_binary_lookup(values, minimum, minimum, maximum - minimum + 1)


def compile_atan2(
    min_input: int = -100,
    max_input: int = 100,
    *,
    input_scale: int = 100,
    output_scale: int = 1000,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled quadrant-aware atan2(y, x) for encrypted fixed-point inputs."""
    minimum, maximum = validate_bounds(min_input, max_input)
    values = _atan2_values(minimum, maximum, input_scale, output_scale)
    return compile_binary_lookup(
        "compile_atan2",
        values,
        minimum,
        maximum,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_asinh(
    min_input: int = -500,
    max_input: int = 500,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled inverse hyperbolic sine for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_asinh",
        math.asinh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_asinh(
    min_input: int = -500,
    max_input: int = 500,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled inverse hyperbolic sine for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_asinh",
        math.asinh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_acosh(
    min_input: int = 100,
    max_input: int = 1000,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled inverse hyperbolic cosine; invalid_result handles x < 1."""
    return _make_scaled_unary(
        "make_acosh",
        math.acosh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value >= 1,
    )


def compile_acosh(
    min_input: int = 100,
    max_input: int = 1000,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled inverse hyperbolic cosine; invalid_result handles x < 1."""
    return _compile_scaled_unary(
        "compile_acosh",
        math.acosh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: value >= 1,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_atanh(
    min_input: int = -99,
    max_input: int = 99,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
) -> UnaryFunction:
    """Create scaled inverse hyperbolic tangent; invalid_result handles |x| >= 1."""
    return _make_scaled_unary(
        "make_atanh",
        math.atanh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: -1 < value < 1,
    )


def compile_atanh(
    min_input: int = -99,
    max_input: int = 99,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
    invalid_result: Optional[int] = None,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled inverse hyperbolic tangent; invalid_result handles |x| >= 1."""
    return _compile_scaled_unary(
        "compile_atanh",
        math.atanh,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=invalid_result,
        domain=lambda value: -1 < value < 1,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_exp2(
    min_input: int = -50,
    max_input: int = 50,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
) -> UnaryFunction:
    """Create scaled 2**x for encrypted fixed-point inputs."""
    return _make_scaled_unary(
        "make_exp2",
        lambda value: 2.0**value,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
    )


def compile_exp2(
    min_input: int = -50,
    max_input: int = 50,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile scaled 2**x for encrypted fixed-point inputs."""
    return _compile_scaled_unary(
        "compile_exp2",
        lambda value: 2.0**value,
        min_input,
        max_input,
        input_scale=input_scale,
        output_scale=output_scale,
        invalid_result=None,
        domain=None,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )
