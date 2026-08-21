"""Bounded scalar arithmetic operations for Concrete FHE."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from ._compat import fhe

from ._utils import compile_function, validate_bounds, validate_integer

BinaryFunction = Callable[[Any, Any], Any]
TernaryFunction = Callable[[Any, Any, Any], Any]


def compare(x: Any, y: Any) -> Any:
    """Return 1 when x > y, 0 when equal, and -1 when x < y.
    
    Example:
        ```python
        from concrete_fhe_toolkit import compare
        
        print(compare(10, 5))  # 1
        print(compare(5, 5))   # 0
        ```
    """
    difference = x - y
    return 1 * (difference > 0) - 1 * (difference < 0)


def compile_compare(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a sign-comparison circuit with inclusive input bounds.
    
    Example:
        ```python
        from concrete_fhe_toolkit import compile_compare
        
        circuit = compile_compare(min_value=-10, max_value=10)
        print(circuit.encrypt_run_decrypt(5, 2))  # 1
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    inputset = [
        (minimum, minimum),
        (minimum, maximum),
        (maximum, minimum),
        (maximum, maximum),
    ]
    return compile_function(
        compare,
        {"x": "encrypted", "y": "encrypted"},
        inputset,
        configuration,
    )

def sign(x: Any) -> Any:
    """Return the sign of a number (1 if positive, -1 if negative, 0 if zero).
    
    Example:
        ```python
        from concrete_fhe_toolkit import sign
        
        print(sign(-42))  # -1
        ```
    """
    return compare(x,0)

def compile_sign(
        min_value: int = -15,
        max_value: int = 15,
        *,
        configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a circuit returning the sign (-1, 0, or 1) of one encrypted input.
    
    Example:
        ```python
        from concrete_fhe_toolkit import compile_sign
        
        circuit = compile_sign(min_value=-5, max_value=5)
        print(circuit.encrypt_run_decrypt(-3))  # -1
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    inputset = [minimum, maximum]
    if minimum <= 0 <= maximum:
        inputset.insert(1, 0)
    return compile_function(
        sign,
        {"x": "encrypted"},
        inputset,
        configuration,
    )

def make_floor_divide(*, zero_result: int = 0) -> BinaryFunction:
    """Create exact encrypted floor division using a multivariate table lookup.
    
    Example:
        ```python
        from concrete_fhe_toolkit import make_floor_divide
        
        divide = make_floor_divide(zero_result=0)
        # Use `divide(x, y)` inside a larger FHE program compilation
        ```
    """
    zero_result = validate_integer("zero_result", zero_result)

    def clear_floor_divide(numerator: Any, denominator: Any) -> Any:
        safe_denominator = np.where(denominator == 0, 1, denominator)
        quotient = np.floor_divide(numerator, safe_denominator)
        return np.asarray(
            np.where(denominator == 0, zero_result, quotient),
            dtype=np.int64,
        )

    operation = fhe.multivariate(clear_floor_divide)

    def floor_divide(numerator: Any, denominator: Any) -> Any:
        return operation(numerator, denominator)

    return floor_divide


def make_floor_divide_by_product(*, zero_result: int = 0) -> TernaryFunction:
    """Create numerator // (left * right) using a multivariate table lookup.
    
    Example:
        ```python
        from concrete_fhe_toolkit import make_floor_divide_by_product
        
        div_prod = make_floor_divide_by_product(zero_result=0)
        # Use `div_prod(num, a, b)` inside a larger FHE program compilation
        ```
    """
    floor_divide = make_floor_divide(zero_result=zero_result)

    def floor_divide_by_product(
        numerator: Any,
        left: Any,
        right: Any,
    ) -> Any:
        return floor_divide(numerator, left * right)

    return floor_divide_by_product


def compile_floor_divide(
    max_numerator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile floor division for nonnegative bounded encrypted inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit import compile_floor_divide
        
        circuit = compile_floor_divide(max_numerator=10, max_denominator=5)
        print(circuit.encrypt_run_decrypt(9, 2))  # 4
        ```
    """
    maximum_numerator = validate_integer(
        "max_numerator",
        max_numerator,
        minimum=0,
    )
    maximum_denominator = validate_integer(
        "max_denominator",
        max_denominator,
        minimum=1,
    )
    function = make_floor_divide(zero_result=zero_result)
    inputset = [
        (0, 0),
        (0, 1),
        (0, maximum_denominator),
        (maximum_numerator, 0),
        (maximum_numerator, 1),
        (maximum_numerator, maximum_denominator),
    ]
    return compile_function(
        function,
        {"numerator": "encrypted", "denominator": "encrypted"},
        inputset,
        configuration,
    )


def compile_floor_divide_by_product(
    max_numerator: int,
    max_left: int,
    max_right: int,
    *,
    zero_result: int = 0,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile numerator // (left * right) for nonnegative bounded inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit import compile_floor_divide_by_product
        
        circuit = compile_floor_divide_by_product(max_numerator=20, max_left=5, max_right=5)
        print(circuit.encrypt_run_decrypt(20, 2, 3))  # 3
        ```
    """
    maximum_numerator = validate_integer(
        "max_numerator",
        max_numerator,
        minimum=0,
    )
    maximum_left = validate_integer("max_left", max_left, minimum=1)
    maximum_right = validate_integer("max_right", max_right, minimum=1)
    function = make_floor_divide_by_product(zero_result=zero_result)
    inputset = [
        (0, 0, 0),
        (0, maximum_left, maximum_right),
        (maximum_numerator, 0, maximum_right),
        (maximum_numerator, maximum_left, 0),
        (maximum_numerator, 1, 1),
        (maximum_numerator, maximum_left, maximum_right),
    ]
    return compile_function(
        function,
        {
            "numerator": "encrypted",
            "left": "encrypted",
            "right": "encrypted",
        },
        inputset,
        configuration,
    )
