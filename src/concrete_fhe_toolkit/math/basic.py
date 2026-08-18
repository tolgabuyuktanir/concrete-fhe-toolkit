"""Basic bounded scalar operations for encrypted integers."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .._compat import fhe

from .._utils import compile_function, validate_bounds, validate_integer
from ._lookup import (
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
    """Return left + right."""
    return left + right


def subtract(left: Any, right: Any) -> Any:
    """Return left - right."""
    return left - right


def multiply(left: Any, right: Any) -> Any:
    """Return left * right."""
    return left * right


def negate(value: Any) -> Any:
    """Return -value."""
    return -value


def square(value: Any) -> Any:
    """Return value squared."""
    return value * value

def cube(value: Any) -> Any:
    """return cube of a number"""
    return value * value * value


def equal(left: Any, right: Any) -> Any:
    """Return 1 when left equals right, otherwise 0."""
    return left == right


def not_equal(left: Any, right: Any) -> Any:
    """Return 1 when left differs from right, otherwise 0."""
    return left != right


def less(left: Any, right: Any) -> Any:
    """Return 1 when left is less than right, otherwise 0."""
    return left < right


def less_equal(left: Any, right: Any) -> Any:
    """Return 1 when left is less than or equal to right."""
    return left <= right


def greater(left: Any, right: Any) -> Any:
    """Return 1 when left is greater than right, otherwise 0."""
    return left > right


def greater_equal(left: Any, right: Any) -> Any:
    """Return 1 when left is greater than or equal to right."""
    return left >= right


def is_zero(value: Any) -> Any:
    return value == 0   


def maximum(left: Any, right: Any) -> Any:
    "return the max number of a pair"
    return ((left+right)+abs(left-right))//2


def minimum(left: Any, right: Any) -> Any:
    "return the min number of a pair"
    return ((left+right)-abs(left-right))//2




def make_scalar_multiply(multiplier: int) -> UnaryFunction:
    """Create multiplication by a public integer constant."""
    normalized = validate_integer("multiplier", multiplier)

    def scalar_multiply(value: Any) -> Any:
        return value * normalized

    return scalar_multiply


def make_is_close(absolute_tolerance: int) -> BinaryFunction:
    """Create an integer closeness predicate using a public tolerance."""
    tolerance = validate_integer(
        "absolute_tolerance",
        absolute_tolerance,
        minimum=0,
    )

    def is_close(left: Any, right: Any) -> Any:
        difference = left - right
        return (difference >= -tolerance) * (difference <= tolerance)

    return is_close


def _binary_inputset(
    min_left: int,
    max_left: int,
    min_right: int,
    max_right: int,
) -> list[tuple[int, int]]:
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
    """Compile encrypted integer addition over inclusive bounds."""
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
    """Compile encrypted integer subtraction over inclusive bounds."""
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
    """Compile encrypted integer multiplication over inclusive bounds."""
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
    """Compile encrypted integer negation."""
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
    """Compile encrypted integer squaring."""
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
    """Compile encrypted integer cubing."""
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
    """Compile multiplication by a public integer constant."""
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
    """Compile encrypted equality."""
    return _compile_predicate(equal, min_value, max_value, configuration)


def compile_not_equal(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted inequality."""
    return _compile_predicate(not_equal, min_value, max_value, configuration)


def compile_less(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted less-than comparison."""
    return _compile_predicate(less, min_value, max_value, configuration)


def compile_less_equal(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted less-than-or-equal comparison."""
    return _compile_predicate(less_equal, min_value, max_value, configuration)


def compile_greater(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted greater-than comparison."""
    return _compile_predicate(greater, min_value, max_value, configuration)


def compile_greater_equal(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted greater-than-or-equal comparison."""
    return _compile_predicate(greater_equal, min_value, max_value, configuration)


def compile_is_zero(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an encrypted is-zero predicate."""
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
    """Compile the encrypted maximum of two integers over inclusive bounds."""
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
    """Compile the encrypted minimum of two integers over inclusive bounds."""
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
    """Compile encrypted integer closeness with an absolute tolerance."""
    function = make_is_close(absolute_tolerance)
    return _compile_predicate(function, min_value, max_value, configuration)


def make_absolute(min_value: int, max_value: int) -> UnaryFunction:
    """Create an absolute-value lookup over inclusive signed bounds."""
    values = unary_values(abs, min_value, max_value)
    return make_unary_lookup(values, min_value)


def compile_absolute(
    min_value: int,
    max_value: int,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile absolute value over inclusive signed bounds."""
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
    """Create a lookup that clamps input into [min_value, max_value]."""
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
    """Compile clamping of an encrypted integer into public bounds."""
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
    """Create Python-style modulo with an explicit zero-denominator result."""
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
    """Compile Python-style modulo with an explicit zero-denominator result."""
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
    """Create quotient-and-remainder lookup with explicit zero behavior."""
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
    """Compile quotient and remainder with explicit zero-denominator behavior."""
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
    """Return when_true when control is 1, otherwise when_false.

    control must be 0 or 1. Unlike `bit_select`, the branches may be any
    bounded integers, so this is the building block for oblivious branching.
    """
    return control * when_true + (1 - control) * when_false


def compile_select(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an oblivious select over two encrypted bounded branches."""
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
    """Create |left - right| for two encrypted bounded integers."""
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum

    if span == 0:
        def constant_abs_diff(left: Any, right: Any) -> Any:
            return left - right

        return constant_abs_diff

    values = unary_values(abs, -span, span)
    lookup = make_unary_lookup(values, -span)

    def abs_diff(left: Any, right: Any) -> Any:
        return lookup(left - right)

    return abs_diff


def compile_abs_diff(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile |left - right| for two encrypted bounded integers."""
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
    """Create copysign(x, y): |x| when y >= 0, otherwise -|x|.

    Follows the integer convention that y == 0 keeps the magnitude positive.
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    absolute = make_absolute(minimum, maximum)

    def copysign(x: Any, y: Any) -> Any:
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
    """Compile copysign(x, y) over inclusive signed bounds."""
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
    minimum, maximum = validate_bounds(min_value, max_value)
    low, high = _saturating_output_range(operation, minimum, maximum)
    clamp_result = make_clamp(low, high, minimum, maximum)

    if operation == "add":
        def saturating(left: Any, right: Any) -> Any:
            return clamp_result(left + right)
    elif operation == "subtract":
        def saturating(left: Any, right: Any) -> Any:
            return clamp_result(left - right)
    else:
        def saturating(left: Any, right: Any) -> Any:
            return clamp_result(left * right)

    return saturating


def make_saturating_add(min_value: int = -15, max_value: int = 15) -> BinaryFunction:
    """Create addition whose result is clamped back into [min_value, max_value]."""
    return _make_saturating("add", min_value, max_value)


def make_saturating_subtract(min_value: int = -15, max_value: int = 15) -> BinaryFunction:
    """Create subtraction whose result is clamped back into [min_value, max_value]."""
    return _make_saturating("subtract", min_value, max_value)


def make_saturating_multiply(min_value: int = -15, max_value: int = 15) -> BinaryFunction:
    """Create multiplication whose result is clamped back into [min_value, max_value]."""
    return _make_saturating("multiply", min_value, max_value)


def _compile_saturating(
    operation: str,
    min_value: int,
    max_value: int,
    allow_large_lookup: bool,
    configuration: Optional[fhe.Configuration],
) -> fhe.Circuit:
    minimum, maximum = validate_bounds(min_value, max_value)
    low, high = _saturating_output_range(operation, minimum, maximum)
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
    """Compile clamped encrypted addition."""
    return _compile_saturating("add", min_value, max_value, allow_large_lookup, configuration)


def compile_saturating_subtract(
    min_value: int = -15,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile clamped encrypted subtraction."""
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
    """Compile clamped encrypted multiplication."""
    return _compile_saturating(
        "multiply", min_value, max_value, allow_large_lookup, configuration
    )


def make_fdim(min_value: int = 0, max_value: int = 15) -> BinaryFunction:
    """Create fdim(x, y) = max(x - y, 0) for two encrypted bounded integers."""
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum

    if span == 0:
        def constant_fdim(left: Any, right: Any) -> Any:
            return left - right

        return constant_fdim

    values = unary_values(lambda diff: max(diff, 0), -span, span)
    lookup = make_unary_lookup(values, -span)

    def fdim(left: Any, right: Any) -> Any:
        return lookup(left - right)

    return fdim


def compile_fdim(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile fdim(x, y) = max(x - y, 0) over inclusive bounds."""
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
    """Return left * right + addend (fused multiply-add, exact on integers)."""
    return left * right + addend


def compile_fma(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile fused multiply-add over three encrypted bounded inputs."""
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
    """Create IEEE-style remainder(x, y) = x - round(x / y) * y (ties to even).

    Unlike modulo, the result is centered around zero. ``zero_result`` is
    returned when the denominator is zero.
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
    """Compile IEEE-style remainder with explicit zero-denominator behavior."""
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
    """Create x * 2**exponent with a public exponent.

    Negative exponents floor-divide (arithmetic shift right).
    """
    normalized = validate_integer("exponent", exponent)
    if normalized >= 0:
        factor = 1 << normalized

        def scale_up(value: Any) -> Any:
            return value * factor

        return scale_up

    divisor = 1 << (-normalized)

    def scale_down(value: Any) -> Any:
        return value // divisor

    return scale_down


def compile_ldexp(
    min_value: int,
    max_value: int,
    exponent: int,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile x * 2**exponent with a public exponent."""
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
