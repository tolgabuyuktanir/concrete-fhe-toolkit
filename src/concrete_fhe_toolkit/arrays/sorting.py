from __future__ import annotations
from typing import Any, Optional
from .._compat import fhe
from .._utils import (
    compile_function,
    positive_difference_lut,
    validate_bounds,
    validate_size,
)
from ._utils import _compile_array_function, UnaryArrayFunction, BinaryScalarFunction

def make_compare_swap(
    min_value: int = 0,
    max_value: int = 15,
) -> BinaryScalarFunction:
    """Create an ascending compare-swap function for bounded encrypted integers.
    
    Args:
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.

    Returns:
        BinaryScalarFunction: A function that sorts two encrypted values.

    Example:
        ```python
        from concrete_fhe_toolkit import make_compare_swap
        
        swap = make_compare_swap(min_value=0, max_value=10)
        # Use `swap(x, y)` inside an FHE program compilation
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum

    if span == 0:
        def compare_equal(x: Any, y: Any) -> Any:
            """Compare two equal elements."""
            return x, y

        return compare_equal

    positive_difference = positive_difference_lut(span)

    def compare_swap(x: Any, y: Any) -> Any:
        """Compare and swap two elements."""
        positive = positive_difference[x - y + span]
        return x - positive, y + positive

    return compare_swap


def compile_compare_swap(
    min_value: int = 0,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an ascending compare-swap circuit.
    
    Args:
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled compare-swap circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_compare_swap
        
        circuit = compile_compare_swap(min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt(5, 3))  # (3, 5)
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_compare_swap(minimum, maximum)
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


def make_sort(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    descending: bool = False,
) -> UnaryArrayFunction:
    """Create a fixed-size bitonic sorting network.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        descending (bool): Whether to sort in descending order instead of ascending.

    Returns:
        UnaryArrayFunction: A function that sorts the input array.

    Example:
        ```python
        from concrete_fhe_toolkit import make_sort
        
        sort_fn = make_sort(size=4, min_value=0, max_value=10)
        # Use `sort_fn(array)` inside an FHE program compilation
        ```
    """
    size = validate_size(size, power_of_two=True)
    minimum, maximum = validate_bounds(min_value, max_value)
    compare_swap = make_compare_swap(minimum, maximum)

    def sort_values(x: Any) -> Any:
        """Sort an array of elements."""
        values = [x[index] for index in range(size)]
        width = 2

        while width <= size:
            distance = width // 2
            while distance > 0:
                for left in range(size):
                    right = left ^ distance
                    if right <= left:
                        continue

                    low, high = compare_swap(values[left], values[right])
                    if left & width:
                        values[left], values[right] = high, low
                    else:
                        values[left], values[right] = low, high
                distance //= 2
            width *= 2

        if descending:
            values.reverse()
        return fhe.array(values)

    return sort_values


def compile_sort(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    descending: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a fixed-size bitonic sorting circuit.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        descending (bool): Whether to sort in descending order.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled sorting circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_sort
        
        circuit = compile_sort(size=4, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([4, 1, 3, 2]))  # [1, 2, 3, 4]
        ```
    """
    size = validate_size(size, power_of_two=True)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_sort(
        size,
        minimum,
        maximum,
        descending=descending,
    )
    return _compile_array_function(function, size, minimum, maximum, configuration)


def make_top_k(
    size: int,
    k: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    largest: bool = True,
) -> UnaryArrayFunction:
    """Create a reduction returning the k largest (or smallest) values in order.

    Uses a sorting network (Bitonic Sort) and slices the top k elements,
    which is significantly more efficient in FHE than iterative greedy selection.
    
    Note: Requires `size` to be a power of two (2, 4, 8...) due to the underlying sort.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        k (int): The number of top elements to return.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        largest (bool): Whether to return the largest values (True) or smallest values (False).

    Returns:
        UnaryArrayFunction: A function that returns the top k values of the input array.

    Example:
        ```python
        from concrete_fhe_toolkit import make_top_k
        
        top_k_fn = make_top_k(size=4, k=2, min_value=0, max_value=10)
        # Use `top_k_fn(array)` inside an FHE program compilation
        ```
    """
    size = validate_size(size, power_of_two=True)
    minimum, maximum = validate_bounds(min_value, max_value)
    k = validate_size(k)
    if k > size:
        raise ValueError("k cannot exceed size")

    # Use bitonic sort
    sort_fn = make_sort(size, minimum, maximum, descending=largest)

    def top_k(x: Any) -> Any:
        """Find the top k elements in an array."""
        sorted_arr = sort_fn(x)
        return sorted_arr[:k]

    return top_k


def compile_top_k(
    size: int,
    k: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    largest: bool = True,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a top-k reduction circuit.
    
    Args:
        size (int): The fixed size of the input array, must be known at compile time.
        k (int): The number of top elements to return.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        largest (bool): Whether to return the largest values (True) or smallest values (False).
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Returns:
        fhe.Circuit: The compiled top-k reduction circuit.

    Example:
        ```python
        from concrete_fhe_toolkit import compile_top_k
        
        circuit = compile_top_k(size=4, k=2, min_value=0, max_value=10)
        print(circuit.encrypt_run_decrypt([4, 1, 3, 2]))  # [4, 3]
        ```
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_top_k(size, k, minimum, maximum, largest=largest)
    return _compile_array_function(function, size, minimum, maximum, configuration)


