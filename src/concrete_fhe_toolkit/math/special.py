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
    """Create scaled sin for encrypted fixed-point angles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_sin
        
        sin_fn = make_sin(min_input=0, max_input=628)
        # Use `sin_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled sin for encrypted fixed-point angles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_sin
        
        circuit = compile_sin(min_input=0, max_input=628)
        print(circuit.encrypt_run_decrypt(314))  # ~sin(pi) scaled
        ```
    """
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
    """Create scaled cos for encrypted fixed-point angles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_cos
        
        cos_fn = make_cos(min_input=0, max_input=628)
        # Use `cos_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled cos for encrypted fixed-point angles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_cos
        
        circuit = compile_cos(min_input=0, max_input=628)
        print(circuit.encrypt_run_decrypt(0))  # 1000 (scaled cos(0))
        ```
    """
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
    """Create scaled tan for encrypted fixed-point angles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_tan
        
        tan_fn = make_tan(min_input=-100, max_input=100)
        # Use `tan_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled tan for encrypted fixed-point angles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_tan
        
        circuit = compile_tan(min_input=-100, max_input=100)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled exp for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_exp
        
        exp_fn = make_exp(min_input=-50, max_input=50)
        # Use `exp_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled exp for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_exp
        
        circuit = compile_exp(min_input=-50, max_input=50)
        print(circuit.encrypt_run_decrypt(0))  # 100 (scaled exp(0))
        ```
    """
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
    """Create scaled expm1 for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_expm1
        
        expm1_fn = make_expm1(min_input=-50, max_input=50)
        # Use `expm1_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled expm1 for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_expm1
        
        circuit = compile_expm1(min_input=-50, max_input=50)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled log (natural by default, or any base); invalid_result handles x <= 0.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_log
        
        log_fn = make_log(min_input=1, max_input=1000)
        # Use `log_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled log (natural by default, or any base); invalid_result handles x <= 0.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_log
        
        circuit = compile_log(min_input=1, max_input=1000)
        print(circuit.encrypt_run_decrypt(100))  # scaled ln(1.0)
        ```
    """
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
    """Create scaled log2; invalid_result handles x <= 0 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_log2
        
        log2_fn = make_log2(min_input=1, max_input=1024)
        # Use `log2_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled log2; invalid_result handles x <= 0 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_log2
        
        circuit = compile_log2(min_input=1, max_input=1024)
        print(circuit.encrypt_run_decrypt(4))  # 200 (scaled log2(4))
        ```
    """
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
    """Create scaled log10; invalid_result handles x <= 0 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_log10
        
        log10_fn = make_log10(min_input=1, max_input=1000)
        # Use `log10_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled log10; invalid_result handles x <= 0 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_log10
        
        circuit = compile_log10(min_input=1, max_input=1000)
        print(circuit.encrypt_run_decrypt(100))  # 200 (scaled log10(100))
        ```
    """
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
    """Create scaled log1p; invalid_result handles x <= -1 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_log1p
        
        log1p_fn = make_log1p(min_input=0, max_input=1000)
        # Use `log1p_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled log1p; invalid_result handles x <= -1 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_log1p
        
        circuit = compile_log1p(min_input=0, max_input=1000)
        print(circuit.encrypt_run_decrypt(0))  # 0 (scaled log1p(0))
        ```
    """
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
    """Create scaled square root; invalid_result handles x < 0 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_sqrt
        
        sqrt_fn = make_sqrt(min_input=0, max_input=1000)
        # Use `sqrt_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled square root; invalid_result handles x < 0 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_sqrt
        
        circuit = compile_sqrt(min_input=0, max_input=1000)
        print(circuit.encrypt_run_decrypt(400))  # 200 (scaled sqrt(4.0))
        ```
    """
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
    """Create scaled erf for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_erf
        
        erf_fn = make_erf(min_input=-30, max_input=30)
        # Use `erf_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled erf for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_erf
        
        circuit = compile_erf(min_input=-30, max_input=30)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled erfc for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_erfc
        
        erfc_fn = make_erfc(min_input=-30, max_input=30)
        # Use `erfc_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled erfc for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_erfc
        
        circuit = compile_erfc(min_input=-30, max_input=30)
        print(circuit.encrypt_run_decrypt(0))  # 100 (scaled erfc(0))
        ```
    """
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
    """Create scaled tanh for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_tanh
        
        tanh_fn = make_tanh(min_input=-40, max_input=40)
        # Use `tanh_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled tanh for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_tanh
        
        circuit = compile_tanh(min_input=-40, max_input=40)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled sinh for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_sinh
        
        sinh_fn = make_sinh(min_input=-30, max_input=30)
        # Use `sinh_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled sinh for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_sinh
        
        circuit = compile_sinh(min_input=-30, max_input=30)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled cosh for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_cosh
        
        cosh_fn = make_cosh(min_input=-30, max_input=30)
        # Use `cosh_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled cosh for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_cosh
        
        circuit = compile_cosh(min_input=-30, max_input=30)
        print(circuit.encrypt_run_decrypt(0))  # 100 (scaled cosh(0))
        ```
    """
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
    """Create scaled logistic sigmoid for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_sigmoid
        
        sigmoid_fn = make_sigmoid(min_input=-60, max_input=60)
        # Use `sigmoid_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled logistic sigmoid for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_sigmoid
        
        circuit = compile_sigmoid(min_input=-60, max_input=60)
        print(circuit.encrypt_run_decrypt(0))  # 500 (scaled sigmoid(0)=0.5)
        ```
    """
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
    """Create scaled arcsine (radians); invalid_result handles |x| > 1 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_asin
        
        asin_fn = make_asin(min_input=-100, max_input=100)
        # Use `asin_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled arcsine (radians); invalid_result handles |x| > 1 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_asin
        
        circuit = compile_asin(min_input=-100, max_input=100)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled arccosine (radians); invalid_result handles |x| > 1 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_acos
        
        acos_fn = make_acos(min_input=-100, max_input=100)
        # Use `acos_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled arccosine (radians); invalid_result handles |x| > 1 if needed.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_acos
        
        circuit = compile_acos(min_input=-100, max_input=100)
        print(circuit.encrypt_run_decrypt(100))  # 0 (scaled acos(1.0))
        ```
    """
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
    """Create scaled arctangent (radians) for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_atan
        
        atan_fn = make_atan(min_input=-500, max_input=500)
        # Use `atan_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled arctangent (radians) for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_atan
        
        circuit = compile_atan(min_input=-500, max_input=500)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled cube root (sign-preserving) for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_cbrt
        
        cbrt_fn = make_cbrt(min_input=-1000, max_input=1000)
        # Use `cbrt_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled cube root (sign-preserving) for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_cbrt
        
        circuit = compile_cbrt(min_input=-1000, max_input=1000)
        print(circuit.encrypt_run_decrypt(8))  # 200 (scaled cbrt(8)=2.0)
        ```
    """
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
    """Create scaled radians-to-degrees conversion for encrypted inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_degrees
        
        deg_fn = make_degrees(min_input=-628, max_input=628)
        # Use `deg_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled radians-to-degrees conversion for encrypted inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_degrees
        
        circuit = compile_degrees(min_input=-628, max_input=628)
        print(circuit.encrypt_run_decrypt(314))  # 180 (degrees)
        ```
    """
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
    """Create scaled degrees-to-radians conversion for encrypted inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_radians
        
        rad_fn = make_radians(min_input=-360, max_input=360)
        # Use `rad_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled degrees-to-radians conversion for encrypted inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_radians
        
        circuit = compile_radians(min_input=-360, max_input=360)
        print(circuit.encrypt_run_decrypt(180))  # 314 (scaled radians)
        ```
    """
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
    """Create scaled gamma; invalid_result handles the non-positive-integer poles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_gamma
        
        gamma_fn = make_gamma(min_input=1, max_input=50)
        # Use `gamma_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled gamma; invalid_result handles the non-positive-integer poles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_gamma
        
        circuit = compile_gamma(min_input=1, max_input=50)
        print(circuit.encrypt_run_decrypt(50))  # scaled gamma(5)
        ```
    """
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
    """Create scaled log-gamma; invalid_result handles the non-positive-integer poles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_lgamma
        
        lgamma_fn = make_lgamma(min_input=1, max_input=100)
        # Use `lgamma_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled log-gamma; invalid_result handles the non-positive-integer poles.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_lgamma
        
        circuit = compile_lgamma(min_input=1, max_input=100)
        print(circuit.encrypt_run_decrypt(50))  # scaled lgamma(5)
        ```
    """
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
    """Create scaled quadrant-aware atan2(y, x) for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_atan2
        
        atan2_fn = make_atan2(min_input=-100, max_input=100)
        # Use `atan2_fn(y, x)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled quadrant-aware atan2(y, x) for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_atan2
        
        circuit = compile_atan2(min_input=-100, max_input=100)
        print(circuit.encrypt_run_decrypt(100, 100))  # ~785 (scaled pi/4)
        ```
    """
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
    """Create scaled inverse hyperbolic sine for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_asinh
        
        asinh_fn = make_asinh(min_input=-500, max_input=500)
        # Use `asinh_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled inverse hyperbolic sine for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_asinh
        
        circuit = compile_asinh(min_input=-500, max_input=500)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled inverse hyperbolic cosine; invalid_result handles x < 1.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_acosh
        
        acosh_fn = make_acosh(min_input=100, max_input=1000)
        # Use `acosh_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled inverse hyperbolic cosine; invalid_result handles x < 1.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_acosh
        
        circuit = compile_acosh(min_input=100, max_input=1000)
        print(circuit.encrypt_run_decrypt(100))  # 0 (scaled acosh(1.0))
        ```
    """
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
    """Create scaled inverse hyperbolic tangent; invalid_result handles |x| >= 1.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_atanh
        
        atanh_fn = make_atanh(min_input=-99, max_input=99)
        # Use `atanh_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled inverse hyperbolic tangent; invalid_result handles |x| >= 1.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_atanh
        
        circuit = compile_atanh(min_input=-99, max_input=99)
        print(circuit.encrypt_run_decrypt(0))  # 0
        ```
    """
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
    """Create scaled 2**x for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import make_exp2
        
        exp2_fn = make_exp2(min_input=-50, max_input=50)
        # Use `exp2_fn(val)` inside an FHE program compilation
        ```
    """
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
    """Compile scaled 2**x for encrypted fixed-point inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.special import compile_exp2
        
        circuit = compile_exp2(min_input=-50, max_input=50)
        print(circuit.encrypt_run_decrypt(10))  # 200 (scaled 2**1.0)
        ```
    """
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
