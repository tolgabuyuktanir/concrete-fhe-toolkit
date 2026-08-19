"""Fixed-size bounded array operations for Concrete FHE."""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional,List

from ._compat import fhe

from concrete_fhe_toolkit.math import (
    bit_and,
    bit_and_many,
    bit_not,
    bit_or,
    bit_or_many,
    equal,
    select,
)

from ._utils import (
    array_inputset,
    compile_function,
    positive_difference_lut,
    validate_bounds,
    validate_size,
)

TieBreak = Literal["first", "last"]
UnaryArrayFunction = Callable[[Any], Any]
BinaryScalarFunction = Callable[[Any, Any], Any]
    
def array_sum(elements: List[Any]) -> Any:
    """Calculate the sum of all elements in an encrypted array using a tournament reduction."""
    current_round = list(elements)
    if(len(current_round) == 0):
        return 0

    sum_result = 0
    while len(current_round) > 1:
        next_round = []
        for i in range(0,len(current_round)-1,2):
            sum_result = current_round[i] + current_round[i+1]
            next_round.append(sum_result)

        if(len(current_round)%2 == 1): #odd number of element check
            next_round.append(current_round[-1])

        current_round = next_round    
    
    return current_round[0]

def array_scale(array: List[Any],factor: int):
    """Multiply every element of an encrypted array by a scalar constant."""
    return [factor * value for value in array]

def array_add(array1: List[Any],array2: List[Any]) -> List[Any]:
    """Perform element-wise addition of two encrypted arrays."""
    return [x+y for x,y in zip(array1,array2)]

def array_sub(array1: List[Any],array2: List[Any]) -> List[Any]:
    """Perform element-wise subtraction of two encrypted arrays."""
    return [x-y for x,y in zip(array1,array2)]

def array_multiply(array1: List[Any],array2: List[Any]) -> List[Any]:
    """Perform element-wise multiplication of two encrypted arrays."""
    return [x*y for x,y in zip(array1,array2)]

def array_pad(array: List[Any], target_size: Any) -> List[Any]:
    """Pad an encrypted array with zeros up to the specified target size."""
    raw_list = list(array)
    if(len(raw_list) > target_size):
        raise ValueError("target_size must be at least the array size")
    padded_list = raw_list + [0] * (target_size - len(raw_list))
    return padded_list

def array_slice(array: List[Any], begin_index: Any, end_index: Any) -> List[Any]:
    """Slice an encrypted array (return elements from begin_index to end_index - 1)."""
    raw_list = list(array)
    list_length = len(raw_list)
    if(list_length < begin_index or list_length < end_index or end_index < begin_index):
        raise ValueError(
            "indexes must be within the array size and begin_index cannot be greater than end_index"
        )

    return raw_list[begin_index:end_index]

def array_contains(array: List[Any], value: Any) -> Any:
    """Check if an encrypted array contains a specific target value (returns 1 or 0)."""
    contain_list = [equal(item,value) for item in array]
    return bit_or_many(contain_list)

def array_count(array: List[Any], value: Any) -> Any:
    """Count occurrences of a specific value in an encrypted array."""
    count_list = [equal(item,value) for item in array]
    return array_sum(count_list)

def array_all_equal(array1: List[Any], array2: List[Any]) -> Any:
    """Check if two encrypted arrays are identical (returns 1 or 0)."""
    list1 = list(array1)
    list2 = list(array2)
    if(len(list1) != len(list2)):
        raise ValueError("The array sizes must be equal")
    
    equal_list = [equal(item1,item2) for item1,item2 in zip(list1,list2)]
    return bit_and_many(equal_list)

def make_compare_swap(
    min_value: int = 0,
    max_value: int = 15,
) -> BinaryScalarFunction:
    """Create an ascending compare-swap function for bounded encrypted integers."""
    minimum, maximum = validate_bounds(min_value, max_value)
    span = maximum - minimum

    if span == 0:
        def compare_equal(x: Any, y: Any) -> Any:
            return x, y

        return compare_equal

    positive_difference = positive_difference_lut(span)

    def compare_swap(x: Any, y: Any) -> Any:
        positive = positive_difference[x - y + span]
        return x - positive, y + positive

    return compare_swap


def make_sort(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    descending: bool = False,
) -> UnaryArrayFunction:
    """Create a fixed-size bitonic sorting network."""
    size = validate_size(size, power_of_two=True)
    minimum, maximum = validate_bounds(min_value, max_value)
    compare_swap = make_compare_swap(minimum, maximum)

    def sort_values(x: Any) -> Any:
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
            return x[0]

        return constant_extreme

    positive_difference = positive_difference_lut(span)

    def extreme(x: Any) -> Any:
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


def make_minimum(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
) -> UnaryArrayFunction:
    """Create a tournament reduction that returns the minimum value."""
    return _make_extreme("min", size, min_value, max_value)


def make_maximum(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
) -> UnaryArrayFunction:
    """Create a tournament reduction that returns the maximum value."""
    return _make_extreme("max", size, min_value, max_value)


def _validate_tie_break(tie_break: str) -> TieBreak:
    if tie_break not in {"first", "last"}:
        raise ValueError("tie_break must be 'first' or 'last'")
    return tie_break  # type: ignore[return-value]


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
    """Create an argmin reduction with deterministic tie handling."""
    return _make_arg_extreme("min", size, min_value, max_value, tie_break)


def make_argmax(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    tie_break: TieBreak = "first",
) -> UnaryArrayFunction:
    """Create an argmax reduction with deterministic tie handling."""
    return _make_arg_extreme("max", size, min_value, max_value, tie_break)


def compile_compare_swap(
    min_value: int = 0,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an ascending compare-swap circuit."""
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


def _compile_array_function(
    function: UnaryArrayFunction,
    size: int,
    min_value: int,
    max_value: int,
    configuration: Optional[fhe.Configuration],
) -> fhe.Circuit:
    inputset = array_inputset(size, min_value, max_value)
    return compile_function(function, {"x": "encrypted"}, inputset, configuration)


def compile_sort(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    descending: bool = False,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a fixed-size bitonic sorting circuit."""
    size = validate_size(size, power_of_two=True)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_sort(
        size,
        minimum,
        maximum,
        descending=descending,
    )
    return _compile_array_function(function, size, minimum, maximum, configuration)


def compile_minimum(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a minimum reduction circuit."""
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_minimum(size, minimum, maximum)
    return _compile_array_function(function, size, minimum, maximum, configuration)


def compile_maximum(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a maximum reduction circuit."""
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_maximum(size, minimum, maximum)
    return _compile_array_function(function, size, minimum, maximum, configuration)


def compile_argmin(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    tie_break: TieBreak = "first",
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an argmin reduction circuit."""
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_argmin(
        size,
        minimum,
        maximum,
        tie_break=tie_break,
    )
    return _compile_array_function(function, size, minimum, maximum, configuration)


def compile_argmax(
    size: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    tie_break: TieBreak = "first",
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an argmax reduction circuit."""
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_argmax(
        size,
        minimum,
        maximum,
        tie_break=tie_break,
    )
    return _compile_array_function(function, size, minimum, maximum, configuration)


def array_index(array: List[Any], index: Any) -> Any:
    """Oblivious read: return array[index] without revealing the encrypted index."""
    items = list(array)
    if not items:
        raise ValueError("array must contain at least one element")
    result: Any = 0
    for position, value in enumerate(items):
        result = result + equal(position, index) * value
    return result


def array_set(array: List[Any], index: Any, value: Any) -> List[Any]:
    """Oblivious write: return a copy with array[index] replaced by value."""
    items = list(array)
    if not items:
        raise ValueError("array must contain at least one element")
    return [
        select(equal(position, index), value, item)
        for position, item in enumerate(items)
    ]


def array_index_of(array: List[Any], value: Any, *, missing_result: Optional[int] = None) -> Any:
    """Return the first index holding value, or missing_result (default len(array))."""
    items = list(array)
    if not items:
        raise ValueError("array must contain at least one element")
    missing = len(items) if missing_result is None else int(missing_result)

    found: Any = 0
    result: Any = 0
    for position, item in enumerate(items):
        flag = equal(item, value)
        is_first = bit_and(flag, bit_not(found))
        result = result + position * is_first
        found = bit_or(found, flag)
    return result + missing * bit_not(found)


def array_cumsum(array: List[Any]) -> List[Any]:
    """Return the running prefix sums of an encrypted array."""
    sums: List[Any] = []
    running: Any = 0
    for item in array:
        running = running + item
        sums.append(running)
    return sums


def array_reverse(array: List[Any]) -> List[Any]:
    """Return the array with its (public) element order reversed."""
    return list(reversed(list(array)))


def array_concat(*arrays: List[Any]) -> List[Any]:
    """Concatenate encrypted arrays along their public length."""
    combined: List[Any] = []
    for array in arrays:
        combined.extend(list(array))
    return combined


def make_top_k(
    size: int,
    k: int,
    min_value: int = 0,
    max_value: int = 15,
    *,
    largest: bool = True,
) -> UnaryArrayFunction:
    """Create a reduction returning the k largest (or smallest) values in order.

    Runs k arg-extreme rounds, masking each selected element with a penalty,
    so circuit cost grows linearly with k.
    """
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    k = validate_size(k)
    if k > size:
        raise ValueError("k cannot exceed size")
    span = maximum - minimum
    penalty = span + 1

    if largest:
        arg_extreme = make_argmax(size, minimum - penalty, maximum)
    else:
        arg_extreme = make_argmin(size, minimum, maximum + penalty)

    def top_k(x: Any) -> Any:
        current = [x[index] for index in range(size)]
        results = []
        for _ in range(k):
            extreme_index = arg_extreme(current)
            selected_flags = [
                equal(position, extreme_index) for position in range(size)
            ]
            value: Any = 0
            next_values = []
            for flag, item in zip(selected_flags, current):
                value = value + flag * item
                if largest:
                    next_values.append(item - flag * penalty)
                else:
                    next_values.append(item + flag * penalty)
            results.append(value)
            current = next_values
        return fhe.array(results)

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
    """Compile a top-k reduction circuit."""
    size = validate_size(size)
    minimum, maximum = validate_bounds(min_value, max_value)
    function = make_top_k(size, k, minimum, maximum, largest=largest)
    return _compile_array_function(function, size, minimum, maximum, configuration)
