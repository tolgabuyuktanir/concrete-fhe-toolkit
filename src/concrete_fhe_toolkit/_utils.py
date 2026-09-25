"""Internal validation and inputset helpers."""

from __future__ import annotations

from numbers import Integral
from typing import Any, List, Optional, Tuple, Callable

import numpy as np
from ._compat import fhe


def validate_integer(name: str, value: int, minimum: Optional[int] = None) -> int:
    """Validate and normalize an integer argument.
    
    Args:
        name (str): The name of the argument.
        value (int): The value to validate.
        minimum (Optional[int], optional): The minimum allowed value. Defaults to None.
        
    Returns:
        int: The normalized integer value.
        
    Raises:
        TypeError: If the value is not an integer.
        ValueError: If the value is less than the minimum.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")

    normalized = int(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def validate_bounds(min_value: int, max_value: int) -> Tuple[int, int]:
    """Validate inclusive integer bounds.
    
    Args:
        min_value (int): The minimum value bound.
        max_value (int): The maximum value bound.
        
    Returns:
        Tuple[int, int]: A tuple containing the validated minimum and maximum bounds.
        
    Raises:
        ValueError: If min_value is greater than max_value.
    """
    minimum = validate_integer("min_value", min_value)
    maximum = validate_integer("max_value", max_value)
    if minimum > maximum:
        raise ValueError("min_value must be less than or equal to max_value")
    return minimum, maximum


def validate_size(size: int, *, power_of_two: bool = False) -> int:
    """Validate a fixed circuit input size.
    
    Args:
        size (int): The size to validate.
        power_of_two (bool, optional): Whether the size must be a power of two. Defaults to False.
        
    Returns:
        int: The validated size.
        
    Raises:
        ValueError: If power_of_two is True and size is not a power of two.
    """
    normalized = validate_integer("size", size, minimum=1)
    if power_of_two and normalized & (normalized - 1):
        raise ValueError("size must be a power of two")
    return normalized


def positive_difference_lut(span: int) -> fhe.LookupTable:
    """Return a LUT for max(value, 0), where value is in [-span, span].
    
    Args:
        span (int): The maximum absolute value of the input.
        
    Returns:
        fhe.LookupTable: The lookup table for the positive difference.
    """
    span = validate_integer("span", span, minimum=1)
    required_length = 2 * span + 1
    table_length = 1 << (required_length - 1).bit_length()
    values = [
        min(max(index - span, 0), span)
        for index in range(table_length)
    ]
    return fhe.LookupTable(values)


def array_inputset(size: int, min_value: int, max_value: int) -> List[np.ndarray]:
    """Create a compact inputset that includes all important array boundaries.
    
    Args:
        size (int): The size of the arrays.
        min_value (int): The minimum value in the arrays.
        max_value (int): The maximum value in the arrays.
        
    Returns:
        List[np.ndarray]: A list of numpy arrays representing the inputset.
    """
    low = np.full(size, min_value, dtype=np.int64)
    high = np.full(size, max_value, dtype=np.int64)
    probes = []
    for index in range(size):
        high_probe = low.copy()
        high_probe[index] = max_value
        probes.append(high_probe)

        low_probe = high.copy()
        low_probe[index] = min_value
        probes.append(low_probe)

    alternating = np.array(
        [min_value if index % 2 == 0 else max_value for index in range(size)],
        dtype=np.int64,
    )
    reverse_alternating = alternating[::-1].copy()
    ramp = np.linspace(min_value, max_value, num=size, dtype=np.int64)
    reverse_ramp = ramp[::-1].copy()
    return [
        low,
        high,
        *probes,
        alternating,
        reverse_alternating,
        ramp,
        reverse_ramp,
    ]

def compile_function(
    function: Any,
    parameter_encryption: dict,
    inputset: list,
    configuration: Optional[fhe.Configuration],
) -> fhe.Circuit:
    """Compile a function while keeping configuration optional.
    
    Args:
        function (Any): The function to compile.
        parameter_encryption (dict): The encryption configuration for parameters.
        inputset (list): The inputset for compilation.
        configuration (Optional[fhe.Configuration]): Optional compilation configuration.
        
    Returns:
        fhe.Circuit: The compiled FHE circuit.
    """
    compiler = fhe.Compiler(function, parameter_encryption)
    if configuration is None:
        return compiler.compile(inputset)
    return compiler.compile(inputset, configuration=configuration)


def client_side_helper(function: Any) -> Callable:
    """Mark a function as a client-side helper.
    
    Args:
        function (Any): The function to mark.
        
    Returns:
        Callable: The marked function.
    """
    function.is_helper = True
    return function