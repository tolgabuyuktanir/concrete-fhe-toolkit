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

from ._utils import _ensure_tensor

def array_sum(elements: Union[np.ndarray, List[Any]]) -> Any:
    """Calculate the sum of all elements in an encrypted array using a tournament reduction.
    
    Args:
        elements (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.

    Returns:
        Any: The computed encrypted sum of all elements in the array.

    Example:
        ```python
        from concrete_fhe_toolkit import array_sum
        
        print(array_sum([1, 2, 3, 4]))  # 10
        ```
    """
    total = np.sum(_ensure_tensor(elements))
    return total


def array_scale(array: Union[np.ndarray, List[Any]],factor: int) -> Union[np.ndarray, List[Any]]:
    """Multiply every element of an encrypted array by a scalar constant.
    
    Args:
        array (Union[np.ndarray, List[Any]]): The encrypted array or list of elements.
        factor (int): The scalar constant to multiply by.

    Returns:
        Union[np.ndarray, List[Any]]: The scaled array.

    Example:
        ```python
        from concrete_fhe_toolkit import array_scale
        
        print(array_scale([1, 2, 3], factor=3))  # [3, 6, 9]
        ```
    """
    return _ensure_tensor(array) * factor


def array_add(array1: Union[np.ndarray, List[Any]],array2: Union[np.ndarray, List[Any]]) -> Any:
    """Perform element-wise addition of two encrypted arrays.
    
    Args:
        array1 (Union[np.ndarray, List[Any]]): The first encrypted array.
        array2 (Union[np.ndarray, List[Any]]): The second encrypted array.

    Returns:
        Any: The element-wise sum of the two arrays.

    Example:
        ```python
        from concrete_fhe_toolkit import array_add
        
        print(array_add([1, 2], [3, 4]))  # [4, 6]
        ```
    """
    return _ensure_tensor(array1) + _ensure_tensor(array2)


def array_sub(array1: Union[np.ndarray, List[Any]],array2: Union[np.ndarray, List[Any]]) -> Any:
    """Perform element-wise subtraction of two encrypted arrays.
    
    Args:
        array1 (Union[np.ndarray, List[Any]]): The first encrypted array.
        array2 (Union[np.ndarray, List[Any]]): The second encrypted array.

    Returns:
        Any: The element-wise difference between the two arrays.

    Example:
        ```python
        from concrete_fhe_toolkit import array_sub
        
        print(array_sub([5, 5], [2, 1]))  # [3, 4]
        ```
    """
    return _ensure_tensor(array1) - _ensure_tensor(array2)


def array_multiply(array1: Union[np.ndarray, List[Any]],array2: Union[np.ndarray, List[Any]]) -> Any:
    """Perform element-wise multiplication of two encrypted arrays.
    
    Args:
        array1 (Union[np.ndarray, List[Any]]): The first encrypted array.
        array2 (Union[np.ndarray, List[Any]]): The second encrypted array.

    Returns:
        Any: The element-wise product of the two arrays.

    Example:
        ```python
        from concrete_fhe_toolkit import array_multiply
        
        print(array_multiply([1, 2], [3, 4]))  # [3, 8]
        ```
    """
    return _ensure_tensor(array1) * _ensure_tensor(array2)


