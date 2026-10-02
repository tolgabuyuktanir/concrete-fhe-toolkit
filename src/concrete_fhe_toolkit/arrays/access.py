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

def make_array_index(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
) -> Callable:
    """Create an oblivious-read function for bounded encrypted arrays.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.

    Returns:
        Callable: A function that obliviously reads an element from the array at a given index.

    Example:
        ```python
        from concrete_fhe_toolkit import make_array_index

        index_fn = make_array_index(size=3, min_value=0, max_value=30)
        print(index_fn([10, 20, 30], 1))  # 20
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)

    def array_index(array: Union[np.ndarray, List[Any]], index: Any) -> Any:
        """Oblivious read: return array[index] without revealing the encrypted index.
        Args:
            array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.
            index (Any): The encrypted index to read.

        Returns:
            Any: The encrypted value at the given index.

        """
        tensor = _ensure_tensor(array)
        if len(tensor) == 0:
            raise ValueError("array must contain at least one element")
        
        positions = np.arange(len(tensor))
        mask = positions == index
        return np.sum(tensor * mask)

    return array_index


def compile_array_index(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an oblivious-read circuit.

    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled oblivious-read circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_array_index

        circuit = compile_array_index(size=3, min_value=0, max_value=30)
        print(circuit.encrypt_run_decrypt([10, 20, 30], 1))  # 20
        ```
    """
    func = make_array_index(size, min_value, max_value)
    base_arrays = array_inputset(size, min_value, max_value)
    inputset = []
    for arr in base_arrays:
        inputset.append((arr, 0))
        inputset.append((arr, size - 1))
    return compile_function(
        func, {"array": "encrypted", "index": "encrypted"}, inputset, configuration
    )


def make_array_set(
    size : int,
    min_value: int = -15,
    max_value: int = 15,
) -> Callable:
    """Create a function for oblivious array writing.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.

    Returns:
        Callable: A function that updates an element in the encrypted array obliviously.

    Example:
        ```python
        from concrete_fhe_toolkit.arrays import make_array_set
        
        array_set = make_array_set(size=5)
        # new_arr = array_set(arr, index, value)
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value,max_value) 

    def array_set(array: Union[np.ndarray, List[Any]], index: Any, value: Any) -> Union[np.ndarray, List[Any]]:
        """Oblivious write: return a copy with array[index] replaced by value.
        
        Args:
            array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.
            index (Any): The encrypted index to update.
            value (Any): The encrypted value to process or validate.

        Returns:
            Union[np.ndarray, List[Any]]: The updated array.

        Example:
            ```python
            from concrete_fhe_toolkit import array_set
            
            print(array_set([10, 20, 30], index=1, value=99))  # [10, 99, 30]
            ```
        """
        arr_tensor = fhe.array(array)
        tensor_size = len(arr_tensor)
        if tensor_size == 0:
            raise ValueError("array must contain at least one element")
        
        positions = np.arange(tensor_size)
        # Use tensor broadcasting to avoid FHE scalar boolean op failures
        mask = (positions == index) * 1
        return mask * value + (1 - mask) * arr_tensor

    return array_set    


def compile_array_set(
    size: int,
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an FHE circuit for oblivious array writing.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled circuit for oblivious array writing.

    Example:
        ```python
        from concrete_fhe_toolkit.arrays import compile_array_set
        
        circuit = compile_array_set(size=5)
        # circuit.encrypt_run_decrypt(arr, index, value)
        ```
    """
    array_set_func = make_array_set(size, min_value, max_value)
    base_arrays = array_inputset(size, min_value, max_value)
    inputset = []
    for arr in base_arrays:
        inputset.append((arr, 0, min_value))
        inputset.append((arr, size - 1, max_value))
    return compile_function(
        array_set_func,
        {"array": "encrypted", "index": "encrypted", "value": "encrypted"},
        inputset,
        configuration,
    )


