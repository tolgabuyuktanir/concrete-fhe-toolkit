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

from ._utils import _ensure_tensor, _compile_array_function

def make_array_index_of(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
    *,
    missing_result: Optional[int] = None,
) -> Callable:
    """Create a first-index-of search function for bounded encrypted arrays.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        missing_result (Optional[int]): The value to return if the target is not found (default: size).

    Returns:
        Callable: A function that returns the index of a specified value.

    Example:
        ```python
        from concrete_fhe_toolkit import make_array_index_of

        index_of_fn = make_array_index_of(size=3, min_value=0, max_value=10)
        print(index_of_fn([10, 20, 30], 20))  # 1
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    missing = size if missing_result is None else int(missing_result)

    def array_index_of(array: Union[np.ndarray, List[Any]], value: Any) -> Any:
        """Return the first index holding value, or missing_result (default size).
        Args:
            array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.
            value (Any): The encrypted value to process or validate.

        Returns:
            Any: The encrypted index of the target value.

        """
        tensor = _ensure_tensor(array)
        tensor_size = len(tensor)
        if tensor_size == 0:
            raise ValueError("array must contain at least one element")
        
        # We subtract value first, then compare to 0.
        # This prevents the Zama compiler from creating a multi-input subgraph 
        # when 'value' is a cleartext runtime argument.
        diff = tensor - value
        found_array = (diff == 0)
        
        argmax_func = make_argmax(tensor_size, 0, 1)
        index = argmax_func(found_array)
    
        # Check if value exists using a single TLU over the sum.
        # This forces a PBS so it doesn't fuse with the select multiplication.
        sum_val = array_sum(found_array)
        any_found = fhe.LookupTable([0] + [1] * tensor_size)[sum_val]

        @fhe.multivariate
        def select_index(a: Any, i: Any) -> Any:
            """Select element at the specified index from array."""
            return i if a else missing
            
        return select_index(any_found, index)

    return array_index_of


def compile_array_index_of(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
    *,
    missing_result: Optional[int] = None,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a first-index-of search circuit.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        missing_result (Optional[int]): The value to return if the target is not found (default: size).
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled search circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_array_index_of

        circuit = compile_array_index_of(size=3, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([10, 20, 30], 20))  # 1
        ```
    """
    func = make_array_index_of(size, min_value, max_value, missing_result=missing_result)
    base_arrays = array_inputset(size, min_value, max_value)
    inputset = []
    for arr in base_arrays:
        inputset.append((arr, min_value))
        inputset.append((arr, max_value))
    return compile_function(
        func, {"array": "encrypted", "value": "encrypted"}, inputset, configuration
    )


def make_array_count(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
) -> Callable:
    """Create a value-counting function for bounded encrypted arrays.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.

    Returns:
        Callable: A function that counts occurrences of a value.

    Example:
        ```python
        from concrete_fhe_toolkit import make_array_count

        count_fn = make_array_count(size=4, min_value=0, max_value=10)
        print(count_fn([1, 2, 2, 3], 2))  # 2
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)

    def array_count(array: Union[np.ndarray, List[Any]], value: Any) -> Any:
        """Count occurrences of a specific value in an encrypted array.
        Args:
            array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.
            value (Any): The encrypted value to process or validate.

        Returns:
            Any: The encrypted count of occurrences.

        """
        count = np.sum(_ensure_tensor(array) == value)
        return count

    return array_count


def compile_array_count(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a value-counting circuit.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled counting circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_array_count

        circuit = compile_array_count(size=4, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([1, 2, 2, 3], 2))  # 2
        ```
    """
    func = make_array_count(size, min_value, max_value)
    base_arrays = array_inputset(size, min_value, max_value)
    inputset = []
    for arr in base_arrays:
        inputset.append((arr, min_value))
        inputset.append((arr, max_value))
    return compile_function(
        func, {"array": "encrypted", "value": "encrypted"}, inputset, configuration
    )


def make_array_contains(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
) -> Callable:
    """Create a membership-test function for bounded encrypted arrays.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.

    Returns:
        Callable: A function that tests whether a value exists in the array.

    Example:
        ```python
        from concrete_fhe_toolkit import make_array_contains

        contains_fn = make_array_contains(size=3, min_value=0, max_value=10)
        print(contains_fn([1, 3, 5], 3))  # 1
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)

    def array_contains(array: Union[np.ndarray, List[Any]], value: Any) -> Any:
        """Check if an encrypted array contains a specific target value (returns 1 or 0).
        Args:
            array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.
            value (Any): The encrypted value to process or validate.

        Returns:
            Any: 1 if the value exists in the array, 0 otherwise.

        """
        contains = np.max(_ensure_tensor(array) == value)
        return contains
    
    return array_contains


def compile_array_contains(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a membership-test circuit.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled membership-test circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_array_contains

        circuit = compile_array_contains(size=3, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([1, 3, 5], 3))  # 1
        ```
    """
    func = make_array_contains(size, min_value, max_value)
    base_arrays = array_inputset(size, min_value, max_value)
    inputset = []
    for arr in base_arrays:
        inputset.append((arr, min_value))
        inputset.append((arr, max_value))
    return compile_function(
        func, {"array": "encrypted", "value": "encrypted"}, inputset, configuration
    )


