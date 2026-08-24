"""Bounded number-theory functions for encrypted integers."""

from __future__ import annotations

import math
from typing import Optional

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


def _binary_math_values(function, min_value: int, max_value: int) -> list[int]:
    minimum, maximum = validate_bounds(min_value, max_value)
    return binary_values(
        function,
        minimum,
        maximum,
        minimum,
        maximum,
    )


def _make_binary_math(function, min_value: int, max_value: int) -> BinaryFunction:
    minimum, maximum = validate_bounds(min_value, max_value)
    values = _binary_math_values(function, minimum, maximum)
    return make_binary_lookup(
        values,
        minimum,
        minimum,
        maximum - minimum + 1,
    )


def make_gcd(min_value: int = 0, max_value: int = 15) -> BinaryFunction:
    """Create math.gcd for two encrypted bounded integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_gcd
        
        gcd_fn = make_gcd(min_value=0, max_value=15)
        # Use `gcd_fn(a, b)` inside an FHE program compilation
        ```
    """
    return _make_binary_math(math.gcd, min_value, max_value)


def compile_gcd(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile math.gcd for two encrypted bounded integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_gcd
        
        circuit = compile_gcd(min_value=0, max_value=15)
        print(circuit.encrypt_run_decrypt(12, 8))  # 4
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = _binary_math_values(math.gcd, minimum, maximum)
    return compile_binary_lookup(
        "compile_gcd",
        values,
        minimum,
        maximum,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_lcm(min_value: int = 0, max_value: int = 15) -> BinaryFunction:
    """Create math.lcm for two encrypted bounded integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_lcm
        
        lcm_fn = make_lcm(min_value=0, max_value=15)
        # Use `lcm_fn(a, b)` inside an FHE program compilation
        ```
    """
    return _make_binary_math(math.lcm, min_value, max_value)


def compile_lcm(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile math.lcm for two encrypted bounded integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_lcm
        
        circuit = compile_lcm(min_value=0, max_value=15)
        print(circuit.encrypt_run_decrypt(4, 6))  # 12
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = _binary_math_values(math.lcm, minimum, maximum)
    return compile_binary_lookup(
        "compile_lcm",
        values,
        minimum,
        maximum,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_is_coprime(
    min_value: int = 0,
    max_value: int = 15,
) -> BinaryFunction:
    """Create a predicate returning 1 when gcd(left, right) == 1.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_is_coprime
        
        coprime_fn = make_is_coprime(min_value=0, max_value=15)
        # Use `coprime_fn(a, b)` inside an FHE program compilation
        ```
    """
    return _make_binary_math(
        lambda left, right: int(math.gcd(left, right) == 1),
        min_value,
        max_value,
    )


def compile_is_coprime(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a predicate returning 1 for coprime encrypted integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_is_coprime
        
        circuit = compile_is_coprime(min_value=0, max_value=15)
        print(circuit.encrypt_run_decrypt(4, 9))  # 1 (coprime)
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = _binary_math_values(
        lambda left, right: int(math.gcd(left, right) == 1),
        minimum,
        maximum,
    )
    return compile_binary_lookup(
        "compile_is_coprime",
        values,
        minimum,
        maximum,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_is_divisible(
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
) -> BinaryFunction:
    """Create divisibility testing with explicit denominator-zero behavior.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_is_divisible
        
        div_fn = make_is_divisible(0, 15, 0, 15, zero_result=0)
        # Use `div_fn(num, den)` inside an FHE program compilation
        ```
    """
    zero = validate_integer("zero_result", zero_result)
    denominator_minimum, denominator_maximum = validate_bounds(
        min_denominator,
        max_denominator,
    )
    values = binary_values(
        lambda numerator, denominator: (
            zero if denominator == 0 else int(numerator % denominator == 0)
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


def compile_is_divisible(
    min_numerator: int,
    max_numerator: int,
    min_denominator: int,
    max_denominator: int,
    *,
    zero_result: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted divisibility testing.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_is_divisible
        
        circuit = compile_is_divisible(0, 15, 0, 15, zero_result=0)
        print(circuit.encrypt_run_decrypt(10, 5))  # 1 (divisible)
        ```
    """
    zero = validate_integer("zero_result", zero_result)
    values = binary_values(
        lambda numerator, denominator: (
            zero if denominator == 0 else int(numerator % denominator == 0)
        ),
        min_numerator,
        max_numerator,
        min_denominator,
        max_denominator,
    )
    return compile_binary_lookup(
        "compile_is_divisible",
        values,
        min_numerator,
        max_numerator,
        min_denominator,
        max_denominator,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_isqrt(max_value: int) -> UnaryFunction:
    """Create math.isqrt for encrypted input in [0, max_value].
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_isqrt
        
        isqrt_fn = make_isqrt(max_value=100)
        # Use `isqrt_fn(value)` inside an FHE program compilation
        ```
    """
    maximum = validate_integer("max_value", max_value, minimum=0)
    values = unary_values(math.isqrt, 0, maximum)
    return make_unary_lookup(values, 0)


def compile_isqrt(
    max_value: int,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile math.isqrt for encrypted input in [0, max_value].
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_isqrt
        
        circuit = compile_isqrt(max_value=100)
        print(circuit.encrypt_run_decrypt(25))  # 5
        ```
    """
    maximum = validate_integer("max_value", max_value, minimum=0)
    values = unary_values(math.isqrt, 0, maximum)
    return compile_unary_lookup(
        "compile_isqrt",
        values,
        0,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_is_even(min_value: int = 0, max_value: int = 15) -> UnaryFunction:
    """Create a predicate returning 1 for even encrypted integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_is_even
        
        is_even_fn = make_is_even(min_value=0, max_value=15)
        # Use `is_even_fn(value)` inside an FHE program compilation
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(lambda value: int(value % 2 == 0), minimum, maximum)
    return make_unary_lookup(values, minimum)


def compile_is_even(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a predicate returning 1 for even encrypted integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_is_even
        
        circuit = compile_is_even(min_value=0, max_value=15)
        print(circuit.encrypt_run_decrypt(4))  # 1
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(lambda value: int(value % 2 == 0), minimum, maximum)
    return compile_unary_lookup(
        "compile_is_even",
        values,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_is_odd(min_value: int = 0, max_value: int = 15) -> UnaryFunction:
    """Create a predicate returning 1 for odd encrypted integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_is_odd
        
        is_odd_fn = make_is_odd(min_value=0, max_value=15)
        # Use `is_odd_fn(value)` inside an FHE program compilation
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(lambda value: int(value % 2 != 0), minimum, maximum)
    return make_unary_lookup(values, minimum)


def compile_is_odd(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a predicate returning 1 for odd encrypted integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_is_odd
        
        circuit = compile_is_odd(min_value=0, max_value=15)
        print(circuit.encrypt_run_decrypt(5))  # 1
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(lambda value: int(value % 2 != 0), minimum, maximum)
    return compile_unary_lookup(
        "compile_is_odd",
        values,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def _is_prime(value: int) -> int:
    if value < 2:
        return 0
    if value == 2:
        return 1
    if value % 2 == 0:
        return 0
    limit = math.isqrt(value)
    for divisor in range(3, limit + 1, 2):
        if value % divisor == 0:
            return 0
    return 1


def make_is_prime(min_value: int = 0, max_value: int = 100) -> UnaryFunction:
    """Create a predicate returning 1 for prime encrypted integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_is_prime
        
        is_prime_fn = make_is_prime(min_value=0, max_value=100)
        # Use `is_prime_fn(value)` inside an FHE program compilation
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(_is_prime, minimum, maximum)
    return make_unary_lookup(values, minimum)


def compile_is_prime(
    min_value: int = 0,
    max_value: int = 100,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a predicate returning 1 for prime encrypted integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_is_prime
        
        circuit = compile_is_prime(min_value=0, max_value=100)
        print(circuit.encrypt_run_decrypt(7))  # 1
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(_is_prime, minimum, maximum)
    return compile_unary_lookup(
        "compile_is_prime",
        values,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def _totient(value: int) -> int:
    if value <= 0:
        return 0
    result = value
    remaining = value
    divisor = 2
    while divisor * divisor <= remaining:
        if remaining % divisor == 0:
            while remaining % divisor == 0:
                remaining //= divisor
            result -= result // divisor
        divisor += 1
    if remaining > 1:
        result -= result // remaining
    return result


def make_totient(min_value: int = 0, max_value: int = 100) -> UnaryFunction:
    """Create Euler's totient for encrypted bounded integers (0 for n <= 0).
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_totient
        
        totient_fn = make_totient(min_value=0, max_value=100)
        # Use `totient_fn(value)` inside an FHE program compilation
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(_totient, minimum, maximum)
    return make_unary_lookup(values, minimum)


def compile_totient(
    min_value: int = 0,
    max_value: int = 100,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile Euler's totient for encrypted bounded integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_totient
        
        circuit = compile_totient(min_value=0, max_value=100)
        print(circuit.encrypt_run_decrypt(9))  # 6
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(_totient, minimum, maximum)
    return compile_unary_lookup(
        "compile_totient",
        values,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def _next_prime(value: int) -> int:
    candidate = max(value + 1, 2)
    while not _is_prime(candidate):
        candidate += 1
    return candidate


def make_next_prime(min_value: int = 0, max_value: int = 100) -> UnaryFunction:
    """Create the smallest prime strictly greater than an encrypted integer.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_next_prime
        
        next_prime_fn = make_next_prime(min_value=0, max_value=100)
        # Use `next_prime_fn(value)` inside an FHE program compilation
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(_next_prime, minimum, maximum)
    return make_unary_lookup(values, minimum)


def compile_next_prime(
    min_value: int = 0,
    max_value: int = 100,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile the smallest prime strictly greater than an encrypted integer.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_next_prime
        
        circuit = compile_next_prime(min_value=0, max_value=100)
        print(circuit.encrypt_run_decrypt(14))  # 17
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = unary_values(_next_prime, minimum, maximum)
    return compile_unary_lookup(
        "compile_next_prime",
        values,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def _mod_inverse_value(value: int, modulus: int, invalid_result: int) -> int:
    if modulus <= 1 or math.gcd(value, modulus) != 1:
        return invalid_result
    return pow(value, -1, modulus)


def make_mod_inverse(
    min_value: int = 0,
    max_value: int = 15,
    *,
    invalid_result: int = 0,
) -> BinaryFunction:
    """Create the modular inverse of value mod modulus for encrypted inputs.

    Returns ``invalid_result`` when the modulus is smaller than 2 or when
    the value is not coprime with the modulus.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_mod_inverse
        
        modinv_fn = make_mod_inverse(min_value=0, max_value=15, invalid_result=0)
        # Use `modinv_fn(value, modulus)` inside an FHE program compilation
        ```
    """
    invalid = validate_integer("invalid_result", invalid_result)
    return _make_binary_math(
        lambda value, modulus: _mod_inverse_value(value, modulus, invalid),
        min_value,
        max_value,
    )


def compile_mod_inverse(
    min_value: int = 0,
    max_value: int = 15,
    *,
    invalid_result: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile the modular inverse of value mod modulus for encrypted inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_mod_inverse
        
        circuit = compile_mod_inverse(min_value=0, max_value=15, invalid_result=0)
        print(circuit.encrypt_run_decrypt(3, 11))  # 4 (since 3*4 = 12 = 1 mod 11)
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    invalid = validate_integer("invalid_result", invalid_result)
    values = _binary_math_values(
        lambda value, modulus: _mod_inverse_value(value, modulus, invalid),
        minimum,
        maximum,
    )
    return compile_binary_lookup(
        "compile_mod_inverse",
        values,
        minimum,
        maximum,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_hypot(min_value: int = 0, max_value: int = 15) -> BinaryFunction:
    """Create round(hypot(x, y)) for two encrypted bounded integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_hypot
        
        hypot_fn = make_hypot(min_value=0, max_value=15)
        # Use `hypot_fn(x, y)` inside an FHE program compilation
        ```
    """
    return _make_binary_math(
        lambda left, right: round(math.hypot(left, right)),
        min_value,
        max_value,
    )


def compile_hypot(
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile round(hypot(x, y)) for two encrypted bounded integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_hypot
        
        circuit = compile_hypot(min_value=0, max_value=15)
        print(circuit.encrypt_run_decrypt(3, 4))  # 5
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    values = _binary_math_values(
        lambda left, right: round(math.hypot(left, right)),
        minimum,
        maximum,
    )
    return compile_binary_lookup(
        "compile_hypot",
        values,
        minimum,
        maximum,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_ilogb(
    min_value: int,
    max_value: int,
    *,
    invalid_result: int = 0,
) -> UnaryFunction:
    """Create floor(log2(|x|)) for encrypted integers; invalid_result handles x == 0.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_ilogb
        
        ilogb_fn = make_ilogb(min_value=0, max_value=15, invalid_result=0)
        # Use `ilogb_fn(value)` inside an FHE program compilation
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    invalid = validate_integer("invalid_result", invalid_result)
    values = unary_values(
        lambda value: invalid if value == 0 else abs(value).bit_length() - 1,
        minimum,
        maximum,
    )
    return make_unary_lookup(values, minimum)


def compile_ilogb(
    min_value: int,
    max_value: int,
    *,
    invalid_result: int = 0,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile floor(log2(|x|)) for encrypted integers; invalid_result handles x == 0.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_ilogb
        
        circuit = compile_ilogb(min_value=0, max_value=15, invalid_result=0)
        print(circuit.encrypt_run_decrypt(8))  # 3
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    invalid = validate_integer("invalid_result", invalid_result)
    values = unary_values(
        lambda value: invalid if value == 0 else abs(value).bit_length() - 1,
        minimum,
        maximum,
    )
    return compile_unary_lookup(
        "compile_ilogb",
        values,
        minimum,
        maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )


def make_dist(size: int, min_value: int = 0, max_value: int = 15) -> BinaryFunction:
    """Create round(Euclidean distance) between two encrypted coordinate lists.

    ``size`` is the (public) number of coordinates; every coordinate must
    stay in [min_value, max_value]. The square-root lookup is built over the
    worst-case squared distance for those bounds.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_dist
        
        dist_fn = make_dist(size=2, min_value=0, max_value=15)
        # Use `dist_fn(p, q)` inside an FHE program compilation
        ```
    """
    normalized_size = validate_integer("size", size, minimum=1)
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum
    max_squared = normalized_size * span * span

    if max_squared == 0:
        def zero_dist(p, q):
            total = 0
            for index in range(normalized_size):
                total = total + (p[index] - q[index])
            return total * 0

        return zero_dist

    values = unary_values(
        lambda squared: round(math.sqrt(squared)),
        0,
        max_squared,
    )
    root = make_unary_lookup(values, 0)

    def dist(p, q):
        squared = 0
        for index in range(normalized_size):
            difference = p[index] - q[index]
            squared = squared + difference * difference
        return root(squared)

    return dist


def compile_dist(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile round(Euclidean distance) between two encrypted coordinate arrays.
    
    Example:
        ```python
        import numpy as np
        from concrete_fhe_toolkit.math.number_theory import compile_dist
        
        circuit = compile_dist(size=2, min_value=0, max_value=15)
        p = np.array([3, 0])
        q = np.array([0, 4])
        print(circuit.encrypt_run_decrypt(p, q))  # 5
        ```
    """
    import numpy as np

    from .._utils import compile_function

    normalized_size = validate_integer("size", size, minimum=1)
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum
    if span:
        from ._lookup import check_lookup_cost

        check_lookup_cost(
            "compile_dist",
            unary_values(
                lambda squared: round(math.sqrt(squared)),
                0,
                normalized_size * span * span,
            ),
            allow_large_lookup=allow_large_lookup,
        )
    function = make_dist(normalized_size, minimum, maximum)
    lows = np.full(normalized_size, minimum, dtype=np.int64)
    highs = np.full(normalized_size, maximum, dtype=np.int64)
    inputset = [
        (lows, lows),
        (lows, highs),
        (highs, lows),
        (highs, highs),
    ]
    return compile_function(
        function,
        {"p": "encrypted", "q": "encrypted"},
        inputset,
        configuration,
    )


def make_pow(
    min_base: int,
    max_base: int,
    max_exponent: int,
) -> BinaryFunction:
    """Create base**exponent for an encrypted base and encrypted exponent.

    Outputs grow extremely fast; the compile-time cost guardrails will
    require ``allow_large_lookup=True`` beyond small bounds.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import make_pow
        
        pow_fn = make_pow(min_base=0, max_base=5, max_exponent=3)
        # Use `pow_fn(base, exponent)` inside an FHE program compilation
        ```
    """
    base_minimum, base_maximum = validate_bounds(min_base, max_base)
    exponent_maximum = validate_integer("max_exponent", max_exponent, minimum=0)
    values = binary_values(
        lambda base, exponent: base**exponent,
        base_minimum,
        base_maximum,
        0,
        exponent_maximum,
    )
    return make_binary_lookup(values, base_minimum, 0, exponent_maximum + 1)


def compile_pow(
    min_base: int,
    max_base: int,
    max_exponent: int,
    *,
    allow_large_lookup: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile base**exponent for an encrypted base and encrypted exponent.
    
    Example:
        ```python
        from concrete_fhe_toolkit.math.number_theory import compile_pow
        
        circuit = compile_pow(min_base=0, max_base=5, max_exponent=3)
        print(circuit.encrypt_run_decrypt(2, 3))  # 8
        ```
    """
    base_minimum, base_maximum = validate_bounds(min_base, max_base)
    exponent_maximum = validate_integer("max_exponent", max_exponent, minimum=0)
    values = binary_values(
        lambda base, exponent: base**exponent,
        base_minimum,
        base_maximum,
        0,
        exponent_maximum,
    )
    return compile_binary_lookup(
        "compile_pow",
        values,
        base_minimum,
        base_maximum,
        0,
        exponent_maximum,
        allow_large_lookup=allow_large_lookup,
        configuration=configuration,
    )
