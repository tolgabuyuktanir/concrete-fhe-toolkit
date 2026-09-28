from __future__ import annotations
from typing import Any, Callable, Literal, Optional, List, Union
from .._compat import fhe
import numpy as np
from .._utils import (
    array_inputset,
    compile_function,
    positive_difference_lut,
    validate_bounds,
    validate_size,
)

from ._utils import _ensure_tensor, _compile_array_function, TieBreak, UnaryArrayFunction

def _validate_tie_break(tie_break: str) -> TieBreak:
    if tie_break not in {"first", "last"}:
        raise ValueError("tie_break must be 'first' or 'last'")
    return tie_break  # type: ignore[return-value]


def _make_extreme(
    operation: Literal["min", "max"],
    size: int,
    min_value: int,
    max_value: int,
) -> UnaryArrayFunction:
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum

    if span == 0:
        def constant_extreme(x: Any) -> Any:
            """Return the extreme value for a constant zero-span."""
            return x[0]

        return constant_extreme

    positive_difference = positive_difference_lut(span)

    def extreme(x: Any) -> Any:
        """Find the extreme value in an array."""
        layer = [x[index] for index in range(size)]
        while len(layer) > 1:
            next_layer = []
            for index in range(0, len(layer), 2):
                if index + 1 == len(layer):
                    next_layer.append(layer[index])
                    continue

                left = layer[index]
                right = layer[index + 1]
                positive = positive_difference[left - right + span]
                if operation == "min":
                    next_layer.append(left - positive)
                else:
                    next_layer.append(right + positive)
            layer = next_layer
        return layer[0]

    return extreme


def make_array_minimum(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
) -> UnaryArrayFunction:
    """Create a tournament reduction that returns the minimum value.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.

    Returns:
        UnaryArrayFunction: A function that returns the minimum value.

    Example:
        ```python
        from concrete_fhe_toolkit import make_minimum
        
        min_fn = make_minimum(size=4, min_value=0, max_value=10)
        # Use `min_fn(array)` inside an FHE program compilation
        ```
    """
    return _make_extreme("min", size, min_value, max_value)


def compile_array_minimum(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a minimum reduction circuit.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled minimum reduction circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_minimum
        
        circuit = compile_minimum(size=4, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([4, 1, 3, 2]))  # 1
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_minimum(size, minimum, maximum)
    return _compile_array_function(function, size, minimum, maximum, configuration)


def make_array_maximum(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
) -> UnaryArrayFunction:
    """Create a tournament reduction that returns the maximum value.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.

    Returns:
        UnaryArrayFunction: A function that returns the maximum value.

    Example:
        ```python
        from concrete_fhe_toolkit import make_maximum
        
        max_fn = make_maximum(size=4, min_value=0, max_value=10)
        # Use `max_fn(array)` inside an FHE program compilation
        ```
    """
    return _make_extreme("max", size, min_value, max_value)


def compile_array_maximum(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a maximum reduction circuit.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled maximum reduction circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_maximum
        
        circuit = compile_maximum(size=4, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([4, 1, 3, 2]))  # 4
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_maximum(size, minimum, maximum)
    return _compile_array_function(function, size, minimum, maximum, configuration)


def _make_arg_extreme(
    operation: Literal["min", "max"],
    size: int,
    min_value: int,
    max_value: int,
    tie_break: str,
) -> UnaryArrayFunction:
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    normalized_tie_break = _validate_tie_break(tie_break)

    if size == 1:
        def only_index(x: Any) -> Any:
            """Return the only index when array size is 1."""
            return x[0] - x[0]

        return only_index

    value_span = maximum - minimum
    encoded_span = value_span * size + size - 1
    positive_difference = positive_difference_lut(encoded_span)

    direct_rank = (
        (operation == "min" and normalized_tie_break == "first")
        or (operation == "max" and normalized_tie_break == "last")
    )

    extraction_length = 1 << encoded_span.bit_length()
    extraction_values = []
    for encoded in range(extraction_length):
        rank = min(encoded, encoded_span) % size
        extraction_values.append(rank if direct_rank else size - 1 - rank)
    extract_index = fhe.LookupTable(extraction_values)

    def arg_extreme(x: Any) -> Any:
        """Find the index of the extreme value in an array."""
        layer = []
        for index in range(size):
            rank = index if direct_rank else size - 1 - index
            layer.append((x[index] - minimum) * size + rank)

        while len(layer) > 1:
            next_layer = []
            for index in range(0, len(layer), 2):
                if index + 1 == len(layer):
                    next_layer.append(layer[index])
                    continue

                left = layer[index]
                right = layer[index + 1]
                positive = positive_difference[left - right + encoded_span]
                if operation == "min":
                    next_layer.append(left - positive)
                else:
                    next_layer.append(right + positive)
            layer = next_layer

        return extract_index[layer[0]]

    return arg_extreme


def make_argmin(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    tie_break: TieBreak = "first",
) -> UnaryArrayFunction:
    """Create an argmin reduction with deterministic tie handling.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        tie_break (TieBreak): Which index to return if multiple elements are equal to the minimum.

    Returns:
        UnaryArrayFunction: A function that returns the index of the minimum value.

    Example:
        ```python
        from concrete_fhe_toolkit import make_argmin
        
        argmin_fn = make_argmin(size=4, min_value=0, max_value=10)
        # Use `argmin_fn(array)` inside an FHE program compilation
        ```
    """
    return _make_arg_extreme("min", size, min_value, max_value, tie_break)


def compile_argmin(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    tie_break: TieBreak = "first",
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an argmin reduction circuit.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        tie_break (TieBreak): Which index to return if multiple elements are equal to the minimum.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled argmin reduction circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_argmin
        
        circuit = compile_argmin(size=4, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([4, 1, 3, 2]))  # 1
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_argmin(
        size,
        minimum,
        maximum,
        tie_break=tie_break,
    )
    return _compile_array_function(function, size, minimum, maximum, configuration)


def make_argmax(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    tie_break: TieBreak = "first",
) -> UnaryArrayFunction:
    """Create an argmax reduction with deterministic tie handling.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        tie_break (TieBreak): Which index to return if multiple elements are equal to the maximum.

    Returns:
        UnaryArrayFunction: A function that returns the index of the maximum value.

    Example:
        ```python
        from concrete_fhe_toolkit import make_argmax
        
        argmax_fn = make_argmax(size=4, min_value=0, max_value=10)
        # Use `argmax_fn(array)` inside an FHE program compilation
        ```
    """
    return _make_arg_extreme("max", size, min_value, max_value, tie_break)


def compile_argmax(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    tie_break: TieBreak = "first",
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an argmax reduction circuit.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        tie_break (TieBreak): Which index to return if multiple elements are equal to the maximum.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled argmax reduction circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_argmax
        
        circuit = compile_argmax(size=4, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([4, 1, 3, 2]))  # 0
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_argmax(
        size,
        minimum,
        maximum,
        tie_break=tie_break,
    )
    return _compile_array_function(function, size, minimum, maximum, configuration)


