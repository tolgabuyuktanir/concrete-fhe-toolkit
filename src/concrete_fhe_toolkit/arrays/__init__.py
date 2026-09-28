"""Fixed-size bounded array operations for Concrete FHE."""

from .arithmetic import array_add, array_multiply, array_scale, array_sub, array_sum
from .manipulation import array_all_equal, array_concat, array_cumsum, array_reverse, array_slice, make_array_pad, compile_array_pad
from .access import compile_array_index, compile_array_set, make_array_index, make_array_set
from .extremes import compile_argmax, compile_argmin, compile_array_maximum, compile_array_minimum, make_argmax, make_argmin, make_array_maximum, make_array_minimum
from .search import compile_array_contains, compile_array_count, compile_array_index_of, make_array_contains, make_array_count, make_array_index_of
from .sorting import compile_compare_swap, compile_sort, compile_top_k, make_compare_swap, make_sort, make_top_k

__all__ = [
    "array_sum",
    "array_scale",
    "array_add",
    "array_sub",
    "array_multiply",
    "array_slice",
    "array_all_equal",
    "array_cumsum",
    "array_reverse",
    "array_concat",
    "make_array_pad",
    "compile_array_pad",
    "compile_array_index",
    "compile_array_set",
    "make_array_index",
    "make_array_set",
    "compile_argmax",
    "compile_argmin",
    "compile_array_maximum",
    "compile_array_minimum",
    "make_argmax",
    "make_argmin",
    "make_array_maximum",
    "make_array_minimum",
    "compile_array_contains",
    "compile_array_count",
    "compile_array_index_of",
    "make_array_contains",
    "make_array_count",
    "make_array_index_of",
    "compile_compare_swap",
    "compile_sort",
    "compile_top_k",
    "make_compare_swap",
    "make_sort",
    "make_top_k",
]

from ._utils import _ensure_tensor
__all__.append('_ensure_tensor')

