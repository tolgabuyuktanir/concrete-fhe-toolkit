"""Basic bounded scalar operations for encrypted integers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .._compat import fhe

from .._utils import compile_function, validate_bounds, validate_integer
from ._lookup import (
    check_lookup_domain,
    BinaryFunction,
    binary_values,
    check_lookup_cost,
    compile_binary_lookup,
    compile_unary_lookup,
    make_binary_lookup,
    make_unary_lookup,
    unary_values,
)

UnaryFunction = Callable[[Any], Any]


def add(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return left + right


def subtract(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return left - right


def multiply(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return left * right


def negate(value: Any) -> Any:
    """value: Any"""
    return -value


def square(value: Any) -> Any:
    """value: Any"""
    return value * value


def cube(value: Any) -> Any:
    """value: Any"""
    return value * value * value


def equal(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return (left == right) * 1

def not_equal(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return (left != right) * 1

def less(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return (left < right) * 1

def less_equal(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return (left <= right) * 1

def greater(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return (left > right) * 1

def greater_equal(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return (left >= right) * 1

def is_zero(value: Any) -> Any:
    """value: Any"""
    return (value == 0) * 1


def maximum(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return ((left+right)+abs(left-right))//2


def minimum(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
    return ((left+right)-abs(left-right))//2




def make_scalar_multiply(multiplier: int) -> UnaryFunction:
    """multiplier: int"""
    normalized = validate_integer("multiplier", multiplier)

    def scalar_multiply(value: Any) -> Any:
    """value: Any"""
        return value * normalized

    return scalar_multiply


def make_is_close(absolute_tolerance: int) -> BinaryFunction:
    """absolute_tolerance: int"""
    tolerance = validate_integer(
        "absolute_tolerance",
        absolute_tolerance,
        minimum=0,
    )

    def is_close(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
        difference = left - right
        return (difference >= -tolerance) * (difference <= tolerance)

    return is_close


def _binary_inputset(
    min_left: int,
    max_left: int,
    min_right: int,
    max_right: int,
) -> list[tuple[int, int]]:
    """
    min_left: int,
    max_left: int,
    min_right: int,
    max_right: int,
"""
    left_minimum, left_maximum = validate_bounds(min_left, max_left)
    right_minimum, right_maximum = validate_bounds(min_right, max_right)
    return [
        (left_minimum, right_minimum),
        (left_minimum, right_maximum),
        (left_maximum, right_minimum),
        (left_maximum, right_maximum),
    ]


def _compile_binary_native(
    function: BinaryFunction,
    min_left: int,
    max_left: int,
    min_right: int,
    max_right: int,
    configuration: Optional[fhe.Configuration],
) -> fhe.Circuit:
    """
    function: BinaryFunction,
    min_left: int,
    max_left: int,
    min_right: int,
    max_right: int,
    configuration: Optional[fhe.Configuration],
"""
    inputset = _binary_inputset(min_left, max_left, min_right, max_right)
    return compile_function(
        function,
        {"left": "encrypted", "right": "encrypted"},
        inputset,
        configuration,
    )


def compile_add(
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_binary_native(
        add,
        min_left,
        max_left,
        min_right,
        max_right,
        configuration,
    )


def compile_subtract(
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_binary_native(
        subtract,
        min_left,
        max_left,
        min_right,
        max_right,
        configuration,
    )


def compile_multiply(
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_binary_native(
        multiply,
        min_left,
        max_left,
        min_right,
        max_right,
        configuration,
    )


def compile_negate(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        negate,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def compile_square(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    inputset = [minimum, maximum]
    if minimum <= 0 <= maximum:
        inputset.append(0)
    return compile_function(
        square,
        {"value": "encrypted"},
        inputset,
        configuration,
    )

def compile_cube(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    inputset = [minimum, maximum]
    if minimum <= 0 <= maximum:
        inputset.append(0)
    return compile_function(
        cube,
        {"value": "encrypted"},
        inputset,
        configuration,
    )    


def compile_scalar_multiply(
    min_value: int,
    max_value: int,
    multiplier: int,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int,
    max_value: int,
    multiplier: int,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_scalar_multiply(multiplier)
    return compile_function(
        function,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def _compile_predicate(
    function: BinaryFunction,
    min_value: int,
    max_value: int,
    configuration: Optional[fhe.Configuration],
) -> fhe.Circuit:
    """
    function: BinaryFunction,
    min_value: int,
    max_value: int,
    configuration: Optional[fhe.Configuration],
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    inputset = [
        (minimum, minimum),
        (minimum, maximum),
        (maximum, minimum),
        (maximum, maximum),
    ]
    if minimum < maximum:
        inputset.append((minimum, minimum + 1))
    return compile_function(
        function,
        {"left": "encrypted", "right": "encrypted"},
        inputset,
        configuration,
    )


def compile_equal(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_predicate(equal, min_value, max_value, configuration)


def compile_not_equal(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_predicate(not_equal, min_value, max_value, configuration)


def compile_less(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_predicate(less, min_value, max_value, configuration)


def compile_less_equal(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_predicate(less_equal, min_value, max_value, configuration)


def compile_greater(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_predicate(greater, min_value, max_value, configuration)


def compile_greater_equal(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_predicate(greater_equal, min_value, max_value, configuration)


def compile_is_zero(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        is_zero,
        {"value": "encrypted"},
        [minimum, 0, maximum],
        configuration,
    )


def compile_maximum(
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_binary_native(
        maximum,
        min_left,
        max_left,
        min_right,
        max_right,
        configuration,
    )


def compile_minimum(
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_left: int = 0,
    max_left: int = 15,
    min_right: int = 0,
    max_right: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_binary_native(
        minimum,
        min_left,
        max_left,
        min_right,
        max_right,
        configuration,
    )


def compile_is_close(
    min_value: int,
    max_value: int,
    *,
    absolute_tolerance: int,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int,
    max_value: int,
    *,
    absolute_tolerance: int,
    configuration: Optional[fhe.Configuration] = None,
"""
    function = make_is_close(absolute_tolerance)
    return _compile_predicate(function, min_value, max_value, configuration)


def make_absolute(min_value: int, max_value: int) -> UnaryFunction:
    """min_value: int, max_value: int"""
    values = unary_values(abs, min_value, max_value)
    return make_unary_lookup(values, min_value)


def compile_absolute(
    min_value: int,
    max_value: int,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int,
    max_value: int,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    check_lookup_domain(
        "compile_absolute",
        (min_value, max_value),
        allow_large_lookup=allow_large_lookup,
    )
    values = unary_values(abs, min_value, max_value)
    return compile_unary_lookup(
        "compile_absolute",
        values,
        min_value,
        max_value,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )

def make_clamp(
    min_input: int,
    max_input: int,
    min_value: int,
    max_value: int,
) -> UnaryFunction:
    """
    min_input: int,
    max_input: int,
    min_value: int,
    max_value: int,
"""
    input_minimum, input_maximum = validate_bounds(min_input, max_input)
    clamp_minimum, clamp_maximum = validate_bounds(min_value, max_value)
    values = unary_values(
        lambda value: max(clamp_minimum, min(value, clamp_maximum)),
        input_minimum,
        input_maximum,
    )
    return make_unary_lookup(values, input_minimum)

def compile_clamp(
    min_input: int,
    max_input: int,
    min_value: int,
    max_value: int,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_input: int,
    max_input: int,
    min_value: int,
    max_value: int,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    check_lookup_domain(
        "compile_clamp",
        (min_input, max_input),
        allow_large_lookup=allow_large_lookup,
    )
    input_minimum, input_maximum = validate_bounds(min_input, max_input)
    clamp_minimum, clamp_maximum = validate_bounds(min_value, max_value)
    values = unary_values(
        lambda value: max(clamp_minimum, min(value, clamp_maximum)),
        input_minimum,
        input_maximum,
    )
    return compile_unary_lookup(
        "compile_clamp",
        values,
        input_minimum,
        input_maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )

def make_modulo(
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
) -> BinaryFunction:
    """
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
"""
    zero = validate_integer("zero_result", zero_result)
    values = binary_values(
        lambda numerator, denominator: (
            zero if denominator == 0 else numerator % denominator
        ),
        min_numerator,
        max_numerator,
        min_denominator,
        max_denominator,
    )
    denominator_minimum, denominator_maximum = validate_bounds(
        min_denominator,
        max_denominator,
    )
    return make_binary_lookup(
        values,
        min_numerator,
        denominator_minimum,
        denominator_maximum - denominator_minimum + 1,
    )


def compile_modulo(
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    check_lookup_domain(
        "compile_modulo",
        (min_numerator, max_numerator),
        (min_denominator, max_denominator),
        allow_large_lookup=allow_large_lookup,
    )
    zero = validate_integer("zero_result", zero_result)
    values = binary_values(
        lambda numerator, denominator: (
            zero if denominator == 0 else numerator % denominator
        ),
        min_numerator,
        max_numerator,
        min_denominator,
        max_denominator,
    )
    return compile_binary_lookup(
        "compile_modulo",
        values,
        min_numerator,
        max_numerator,
        min_denominator,
        max_denominator,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_divmod(
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_quotient: int = 0,
    zero_remainder: int = 0,
) -> BinaryFunction:
    """
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_quotient: int = 0,
    zero_remainder: int = 0,
"""
    zero_q = validate_integer("zero_quotient", zero_quotient)
    zero_r = validate_integer("zero_remainder", zero_remainder)
    denominator_minimum, denominator_maximum = validate_bounds(
        min_denominator,
        max_denominator,
    )
    quotient_values = binary_values(
        lambda numerator, denominator: (
            zero_q if denominator == 0 else numerator // denominator
        ),
        min_numerator,
        max_numerator,
        denominator_minimum,
        denominator_maximum,
    )
    remainder_values = binary_values(
        lambda numerator, denominator: (
            zero_r if denominator == 0 else numerator % denominator
        ),
        min_numerator,
        max_numerator,
        denominator_minimum,
        denominator_maximum,
    )
    width = denominator_maximum - denominator_minimum + 1
    quotient = make_binary_lookup(
        quotient_values,
        min_numerator,
        denominator_minimum,
        width,
    )
    remainder = make_binary_lookup(
        remainder_values,
        min_numerator,
        denominator_minimum,
        width,
    )

    def quotient_and_remainder(numerator: Any, denominator: Any) -> Any:
    """numerator: Any, denominator: Any"""
        return quotient(numerator, denominator), remainder(numerator, denominator)

    return quotient_and_remainder


def compile_divmod(
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_quotient: int = 0,
    zero_remainder: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_quotient: int = 0,
    zero_remainder: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    check_lookup_domain(
        "compile_divmod",
        (min_numerator, max_numerator),
        (min_denominator, max_denominator),
        allow_large_lookup=allow_large_lookup,
    )
    function = make_divmod(
        min_numerator,
        max_numerator,
        min_denominator,
        max_denominator,
        zero_quotient=zero_quotient,
        zero_remainder=zero_remainder,
    )
    numerator_minimum, numerator_maximum = validate_bounds(
        min_numerator,
        max_numerator,
    )
    denominator_minimum, denominator_maximum = validate_bounds(
        min_denominator,
        max_denominator,
    )
    quotient_values = binary_values(
        lambda numerator, denominator: (
            zero_quotient if denominator == 0 else numerator // denominator
        ),
        numerator_minimum,
        numerator_maximum,
        denominator_minimum,
        denominator_maximum,
    )
    remainder_values = binary_values(
        lambda numerator, denominator: (
            zero_remainder if denominator == 0 else numerator % denominator
        ),
        numerator_minimum,
        numerator_maximum,
        denominator_minimum,
        denominator_maximum,
    )
    check_lookup_cost(
        "compile_divmod quotient",
        quotient_values,
        allow_large_lookup=allow_large_lookup,
    )
    check_lookup_cost(
        "compile_divmod remainder",
        remainder_values,
        allow_large_lookup=allow_large_lookup,
    )
    inputset = _binary_inputset(
        numerator_minimum,
        numerator_maximum,
        denominator_minimum,
        denominator_maximum,
    )
    
    if denominator_minimum <= 0 <= denominator_maximum:
        inputset.append((numerator_minimum, 0))
        inputset.append((numerator_maximum, 0))
        
    return compile_function(
        function,
        {"numerator": "encrypted", "denominator": "encrypted"},
        inputset,
        configuration,
    )


compile_abs = compile_absolute
compile_sub = compile_subtract
make_abs = make_absolute
compile_scalar_mul = compile_scalar_multiply


def select(control: Any, when_true: Any, when_false: Any) -> Any:
    """control: Any, when_true: Any, when_false: Any"""
    return control * (when_true - when_false) + when_false


def compile_select(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    inputset = [
        (0, minimum, minimum),
        (0, maximum, maximum),
        (0, minimum, maximum),
        (1, minimum, minimum),
        (1, maximum, maximum),
        (1, maximum, minimum),
    ]
    return compile_function(
        select,
        {
            "control": "encrypted",
            "when_true": "encrypted",
            "when_false": "encrypted",
        },
        inputset,
        configuration,
    )


def make_abs_diff(min_value: int = 0, max_value: int = 15) -> BinaryFunction:
    """min_value: int = 0, max_value: int = 15"""
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum

    if span == 0:
        def constant_abs_diff(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
            return left - right

        return constant_abs_diff

    values = unary_values(abs, -span, span)
    lookup = make_unary_lookup(values, -span)

    def abs_diff(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
        return lookup(left - right)

    return abs_diff


def compile_abs_diff(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    low, high = validate_bounds(min_value, max_value)
    check_lookup_domain(
        "compile_abs_diff",
        (-(high - low), high - low),
        allow_large_lookup=allow_large_lookup,
    )
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum
    if span:
        check_lookup_cost(
            "compile_abs_diff",
            unary_values(abs, -span, span),
            allow_large_lookup=allow_large_lookup,
        )
    function = make_abs_diff(minimum, maximum)
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


def make_copysign(min_value: int = -15, max_value: int = 15) -> BinaryFunction:
    """min_value: int = -15, max_value: int = 15"""
    minimum, maximum = validate_bounds(min_value, max_value)
    absolute = make_absolute(minimum, maximum)

    def copysign(x: Any, y: Any) -> Any:
    """x: Any, y: Any"""
        sign_factor = (y >= 0) * 2 - 1
        return absolute(x) * sign_factor

    return copysign


def compile_copysign(
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    check_lookup_domain(
        "compile_copysign",
        (min_value, max_value),
        allow_large_lookup=allow_large_lookup,
    )
    minimum, maximum = validate_bounds(min_value, max_value)
    check_lookup_cost(
        "compile_copysign",
        unary_values(abs, minimum, maximum),
        allow_large_lookup=allow_large_lookup,
    )
    function = make_copysign(minimum, maximum)
    inputset = [
        (minimum, minimum),
        (minimum, maximum),
        (maximum, minimum),
        (maximum, maximum),
    ]
    return compile_function(
        function,
        {"x": "encrypted", "y": "encrypted"},
        inputset,
        configuration,
    )


def _saturating_output_range(operation: str, minimum: int, maximum: int) -> tuple:
    """operation: str, minimum: int, maximum: int"""
    if operation == "add":
        return 2 * minimum, 2 * maximum
    if operation == "subtract":
        return minimum - maximum, maximum - minimum
    corners = [
        minimum * minimum,
        minimum * maximum,
        maximum * minimum,
        maximum * maximum,
    ]
    return min(corners), max(corners)


def _make_saturating(operation: str, min_value: int, max_value: int) -> BinaryFunction:
    """operation: str, min_value: int, max_value: int"""
    minimum, maximum = validate_bounds(min_value, max_value)
    low, high = _saturating_output_range(operation, minimum, maximum)
    clamp_result = make_clamp(low, high, minimum, maximum)

    if operation == "add":
        def saturating(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
            return clamp_result(left + right)
    elif operation == "subtract":
        def saturating(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
            return clamp_result(left - right)
    else:
        def saturating(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
            return clamp_result(left * right)

    return saturating


def make_saturating_add(min_value: int = -15, max_value: int = 15) -> BinaryFunction:
    """min_value: int = -15, max_value: int = 15"""
    return _make_saturating("add", min_value, max_value)


def make_saturating_subtract(min_value: int = -15, max_value: int = 15) -> BinaryFunction:
    """min_value: int = -15, max_value: int = 15"""
    return _make_saturating("subtract", min_value, max_value)


def make_saturating_multiply(min_value: int = -15, max_value: int = 15) -> BinaryFunction:
    """min_value: int = -15, max_value: int = 15"""
    return _make_saturating("multiply", min_value, max_value)


def _compile_saturating(
    operation: str,
    min_value: int,
    max_value: int,
    allow_large_lookup: bool,
    configuration: Optional[fhe.Configuration],
) -> fhe.Circuit:
    """
    operation: str,
    min_value: int,
    max_value: int,
    allow_large_lookup: bool,
    configuration: Optional[fhe.Configuration],
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    low, high = _saturating_output_range(operation, minimum, maximum)
    check_lookup_domain('_compile_saturating', (low, high), allow_large_lookup=allow_large_lookup)
    check_lookup_cost(
        f"compile_saturating_{operation}",
        unary_values(
            lambda value: max(minimum, min(value, maximum)),
            low,
            high,
        ),
        allow_large_lookup=allow_large_lookup,
    )
    function = _make_saturating(operation, minimum, maximum)
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


def compile_saturating_add(
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_saturating("add", min_value, max_value, allow_large_lookup, configuration)


def compile_saturating_subtract(
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_saturating(
        "subtract", min_value, max_value, allow_large_lookup, configuration
    )


def compile_saturating_multiply(
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    return _compile_saturating(
        "multiply", min_value, max_value, allow_large_lookup, configuration
    )


def make_fdim(min_value: int = 0, max_value: int = 15) -> BinaryFunction:
    """min_value: int = 0, max_value: int = 15"""
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum

    if span == 0:
        def constant_fdim(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
            return left - right

        return constant_fdim

    values = unary_values(lambda diff: max(diff, 0), -span, span)
    lookup = make_unary_lookup(values, -span)

    def fdim(left: Any, right: Any) -> Any:
    """left: Any, right: Any"""
        return lookup(left - right)

    return fdim


def compile_fdim(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    low, high = validate_bounds(min_value, max_value)
    check_lookup_domain(
        "compile_fdim",
        (0, 2 * (high - low)),
        allow_large_lookup=allow_large_lookup,
    )
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum
    if span:
        check_lookup_cost(
            "compile_fdim",
            unary_values(lambda diff: max(diff, 0), -span, span),
            allow_large_lookup=allow_large_lookup,
        )
    function = make_fdim(minimum, maximum)
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


def fma(left: Any, right: Any, addend: Any) -> Any:
    """left: Any, right: Any, addend: Any"""
    return left * right + addend


def compile_fma(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    inputset = [
        (minimum, minimum, minimum),
        (minimum, maximum, minimum),
        (maximum, minimum, maximum),
        (maximum, maximum, maximum),
        (minimum, maximum, maximum),
        (maximum, maximum, minimum),
    ]
    return compile_function(
        fma,
        {"left": "encrypted", "right": "encrypted", "addend": "encrypted"},
        inputset,
        configuration,
    )


def make_remainder(
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
) -> BinaryFunction:
    """
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
"""
    from fractions import Fraction

    zero = validate_integer("zero_result", zero_result)
    denominator_minimum, denominator_maximum = validate_bounds(
        min_denominator,
        max_denominator,
    )
    values = binary_values(
        lambda numerator, denominator: (
            zero
            if denominator == 0
            else numerator - round(Fraction(numerator, denominator)) * denominator
        ),
        min_numerator,
        max_numerator,
        denominator_minimum,
        denominator_maximum,
    )
    return make_binary_lookup(
        values,
        min_numerator,
        denominator_minimum,
        denominator_maximum - denominator_minimum + 1,
    )


def compile_remainder(
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
"""
    check_lookup_domain(
        "compile_remainder",
        (min_numerator, max_numerator),
        (min_denominator, max_denominator),
        allow_large_lookup=allow_large_lookup,
    )
    from fractions import Fraction

    zero = validate_integer("zero_result", zero_result)
    values = binary_values(
        lambda numerator, denominator: (
            zero
            if denominator == 0
            else numerator - round(Fraction(numerator, denominator)) * denominator
        ),
        min_numerator,
        max_numerator,
        min_denominator,
        max_denominator,
    )
    return compile_binary_lookup(
        "compile_remainder",
        values,
        min_numerator,
        max_numerator,
        min_denominator,
        max_denominator,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_ldexp(exponent: int) -> UnaryFunction:
    """exponent: int"""
    normalized = validate_integer("exponent", exponent)
    if normalized >= 0:
        factor = 1 << normalized

        def scale_up(value: Any) -> Any:
    """value: Any"""
            return value * factor

        return scale_up

    divisor = 1 << (-normalized)

    def scale_down(value: Any) -> Any:
    """value: Any"""
        return value // divisor

    return scale_down


def compile_ldexp(
    min_value: int,
    max_value: int,
    exponent: int,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """
    min_value: int,
    max_value: int,
    exponent: int,
    *,
    configuration: Optional[fhe.Configuration] = None,
"""
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_ldexp(exponent)
    return compile_function(
        function,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


# C-math-parity aliases.
make_scalbn = make_ldexp
compile_scalbn = compile_ldexp


def fsum(values: Any) -> Any:
    """values: Any"""
    items = list(values)
    if not items:
        return 0
    while len(items) > 1:
        next_items = []
        for index in range(0, len(items) - 1, 2):
            next_items.append(items[index] + items[index + 1])
        if len(items) % 2 == 1:
            next_items.append(items[-1])
        items = next_items
    return items[0]


def prod(values: Any, start: int = 1) -> Any:
    """values: Any, start: int = 1"""
    normalized_start = validate_integer("start", start)
    items = list(values)
    if not items:
        return normalized_start
    while len(items) > 1:
        next_items = []
        for index in range(0, len(items) - 1, 2):
            next_items.append(items[index] * items[index + 1])
        if len(items) % 2 == 1:
            next_items.append(items[-1])
        items = next_items
    return items[0] * normalized_start


def sumprod(p: Any, q: Any) -> Any:
    """p: Any, q: Any"""
    left = list(p)
    right = list(q)
    if len(left) != len(right):
        raise ValueError("p and q must have the same length")
    return fsum(item_p * item_q for item_p, item_q in zip(left, right))
