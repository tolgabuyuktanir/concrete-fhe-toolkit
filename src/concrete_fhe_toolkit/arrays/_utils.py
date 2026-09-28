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

TieBreak = Literal['first', 'last']
UnaryArrayFunction = Callable[[Any], Any]
BinaryScalarFunction = Callable[[Any, Any], Any]

def _ensure_tensor(arr: Any) -> Any:
    if isinstance(arr, (list, tuple)):
        return np.array(arr)
    return arr


def _compile_array_function(
    function: UnaryArrayFunction,
    size: int,
    min_value: int,
    max_value: int,
    configuration: Optional[fhe.Configuration],
) -> fhe.Circuit:
    inputset = array_inputset(size, min_value, max_value)
    return compile_function(function, {"x": "encrypted"}, inputset, configuration)


