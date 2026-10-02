from __future__ import annotations
from typing import Any, Callable, Optional, List, Union
from .._compat import fhe
import numpy as np
from .._utils import (
    array_inputset,
    compile_function,
    validate_bounds,
    validate_size,
)

from ._utils import _ensure_tensor

def array_slice(array: Union[np.ndarray, List[Any]], begin_index: Any, end_index: Any) -> Any:
    """Slice an encrypted array (return elements from begin_index to end_index - 1).
    
    Args:
        array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.
        begin_index (Any): The starting index (inclusive).
        end_index (Any): The ending index (exclusive).

    Returns:
        Any: The sliced array.

    Example:
        ```python
        from concrete_fhe_toolkit import array_slice
        
        print(array_slice([10, 20, 30, 40], 1, 3))  # [20, 30]
        ```
    """
    raw_list = list(array)
    list_length = len(raw_list)
    if(list_length < begin_index or list_length < end_index or end_index < begin_index):
        raise ValueError(
            "indexes must be within the array size and begin_index cannot be greater than end_index"
        )

    return fhe.array(raw_list[begin_index:end_index])


def array_all_equal(array1: Union[np.ndarray, List[Any]], array2: Union[np.ndarray, List[Any]]) -> Any:
    """Check if two encrypted arrays are identical (returns 1 or 0).
    
    Args:
        array1 (Union[np.ndarray, List[Any]]): The first encrypted array.
        array2 (Union[np.ndarray, List[Any]]): The second encrypted array.

    Returns:
        Any: 1 if all elements are equal, 0 otherwise.

    Example:
        ```python
        from concrete_fhe_toolkit import array_all_equal
        
        print(array_all_equal([1, 2], [1, 2]))  # 1
        ```
    """
    tensor1 = _ensure_tensor(array1)
    tensor2 = _ensure_tensor(array2)

    return np.sum(tensor1 == tensor2) == len(tensor1)


def array_cumsum(array: Union[np.ndarray, List[Any]]) -> Union[np.ndarray, List[Any]]:
    """Return the running prefix sums of an encrypted array.
    
    Args:
        array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.

    Returns:
        Union[np.ndarray, List[Any]]: The array of prefix sums.

    Example:
        ```python
        from concrete_fhe_toolkit import array_cumsum
        
        print(array_cumsum([1, 2, 3]))  # [1, 3, 6]
        ```
    """
    sums: List[Any] = []
    running: Any = 0
    for item in array:
        running = running + item
        sums.append(running)
    return fhe.array(sums)


def array_reverse(array: Union[np.ndarray, List[Any]]) -> Union[np.ndarray, List[Any]]:
    """Return the array with its (public) element order reversed.
    
    Args:
        array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.

    Returns:
        Union[np.ndarray, List[Any]]: The reversed array.

    Example:
        ```python
        from concrete_fhe_toolkit import array_reverse
        
        print(array_reverse([1, 2, 3]))  # [3, 2, 1]
        ```
    """
    return _ensure_tensor(array)[::-1]


def array_concat(*arrays: Union[np.ndarray, List[Any]]) -> Union[np.ndarray, List[Any]]:
    """Concatenate encrypted arrays along their public length.
    
    Args:
        *arrays (Union[np.ndarray, List[Any]]): A variable number of arrays to concatenate.

    Returns:
        Union[np.ndarray, List[Any]]: The concatenated array.

    Example:
        ```python
        from concrete_fhe_toolkit import array_concat
        
        print(array_concat([1, 2], [3, 4]))  # [1, 2, 3, 4]
        ```
    """
    combined: List[Any] = []
    for array in arrays:
        combined.extend(list(array))
    return fhe.array(combined)


def make_array_pad(
    size: int,
    target_size: int,
    min_value: int = -15,
    max_value: int = 15,
) -> Callable:
    """Create a fixed-size array padding function.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        target_size (int): The desired size of the array after padding.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.

    Returns:
        Callable: A function that pads the input array with zeros.

    Example:
        ```python
        from concrete_fhe_toolkit import make_array_pad

        pad_fn = make_array_pad(size=2, target_size=4, min_value=0, max_value=10)
        print(pad_fn([1, 2]))  # [1, 2, 0, 0]
        ```
    """
    size = validate_size(size)
    target_size = validate_size(target_size)
    minimum, maximum = validate_bounds(min_value, max_value)
    if size > target_size:
        raise ValueError("target_size must be at least the array size")

    def array_pad(array: Union[np.ndarray, List[Any]]) -> Any:
        """Pad an encrypted array with zeros up to the specified target size.
        Args:
            array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.

        Returns:
            Any: The zero-padded array.

        """
        raw_list = list(array)
        padded_list = raw_list + [0] * (target_size - len(raw_list))
        return fhe.array(padded_list)

    return array_pad


def compile_array_pad(
    size: int,
    target_size: int,
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a fixed-size array padding circuit.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        target_size (int): The desired size of the array after padding.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled padding circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_array_pad

        circuit = compile_array_pad(size=2, target_size=4, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([1, 2]))  # [1, 2, 0, 0]
        ```
    """
    func = make_array_pad(size, target_size, min_value, max_value)
    inputset = array_inputset(size, min_value, max_value)
    return compile_function(func, {"array": "encrypted"}, inputset, configuration)


