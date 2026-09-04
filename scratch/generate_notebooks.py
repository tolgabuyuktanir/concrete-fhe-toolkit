import json
import os

def create_cell(cell_type, source, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    }
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip('\n')
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs or []
    return cell

def save_notebook(filename, cells):
    notebook = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

def generate_notebook(filename, module_name, tests):
    cells = []
    for fn_name, desc, test_code in tests:
        cells.append(create_cell("markdown", f"### Testing `{fn_name}`\n\n{desc}"))
        
        # Build imports
        imports = f"import numpy as np\nfrom concrete import fhe\nfrom {module_name} import {fn_name}\n\n"
        
        cells.append(create_cell("code", f"{imports}{test_code}\nprint(\"{fn_name} tests passed!\")"))
        
    save_notebook(filename, cells)


def get_arithmetic_tests():
    return [
        (
            "compare",
            "This function compares two values and returns 1 if x > y, 0 if x == y, and -1 if x < y. The parameters `x` and `y` are typed as `Any`, so they are compiled as `encrypted`.",
            """def test_compare(x, y):
    return compare(x, y)

compiler = fhe.Compiler(test_compare, {"x": "encrypted", "y": "encrypted"})
inputset = [(10, 5), (5, 5), (-5, 5), (0, 0)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt(10, 5) == 1
assert circuit.encrypt_run_decrypt(5, 5) == 0
assert circuit.encrypt_run_decrypt(-5, 5) == -1
assert circuit.encrypt_run_decrypt(0, 0) == 0"""
        ),
        (
            "compile_compare",
            "This function returns an already compiled FHE circuit to compare two encrypted values.",
            """circuit = compile_compare(min_value=-50, max_value=50)

assert circuit.encrypt_run_decrypt(30, 10) == 1
assert circuit.encrypt_run_decrypt(20, 20) == 0
assert circuit.encrypt_run_decrypt(-20, 40) == -1"""
        ),
        (
            "sign",
            "This function returns the sign of a number. Parameter `x` is `Any`, compiled as `encrypted`.",
            """def test_sign(x):
    return sign(x)

compiler = fhe.Compiler(test_sign, {"x": "encrypted"})
inputset = [[42], [-42], [0]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt(42) == 1
assert circuit.encrypt_run_decrypt(-42) == -1
assert circuit.encrypt_run_decrypt(0) == 0"""
        ),
        (
            "compile_sign",
            "This function returns a compiled FHE circuit to return the sign of a number.",
            """circuit = compile_sign(min_value=-50, max_value=50)

assert circuit.encrypt_run_decrypt(40) == 1
assert circuit.encrypt_run_decrypt(-30) == -1
assert circuit.encrypt_run_decrypt(0) == 0"""
        ),
        (
            "make_floor_divide",
            "Creates an exact encrypted floor division. The returned function takes `numerator` and `denominator` as `Any`, both compiled as `encrypted`.",
            """floor_divide = make_floor_divide(zero_result=99)

def test_floor_divide(num, den):
    return floor_divide(num, den)

compiler = fhe.Compiler(test_floor_divide, {"num": "encrypted", "den": "encrypted"})
inputset = [(10, 3), (10, 5), (5, 0)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt(10, 3) == 3
assert circuit.encrypt_run_decrypt(10, 5) == 2
assert circuit.encrypt_run_decrypt(5, 0) == 99"""
        ),
        (
            "make_floor_divide_by_product",
            "Creates a division by a product (numerator // (left * right)). Returned function takes three `Any` params, all compiled as `encrypted`.",
            """div_prod = make_floor_divide_by_product(zero_result=0)

def test_div_prod(num, left, right):
    return div_prod(num, left, right)

compiler = fhe.Compiler(test_div_prod, {"num": "encrypted", "left": "encrypted", "right": "encrypted"})
inputset = [(20, 2, 3), (10, 5, 2), (15, 0, 5)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt(20, 2, 3) == 3
assert circuit.encrypt_run_decrypt(10, 5, 2) == 1
assert circuit.encrypt_run_decrypt(15, 0, 5) == 0"""
        ),
        (
            "compile_floor_divide",
            "Returns a compiled floor division circuit.",
            """circuit = compile_floor_divide(max_numerator=50, max_denominator=20, zero_result=0)

assert circuit.encrypt_run_decrypt(45, 10) == 4
assert circuit.encrypt_run_decrypt(20, 5) == 4
assert circuit.encrypt_run_decrypt(8, 0) == 0"""
        ),
        (
            "compile_floor_divide_by_product",
            "Returns a compiled circuit for division by a product.",
            """circuit = compile_floor_divide_by_product(max_numerator=50, max_left=10, max_right=10, zero_result=0)

assert circuit.encrypt_run_decrypt(50, 2, 5) == 5
assert circuit.encrypt_run_decrypt(15, 5, 1) == 3
assert circuit.encrypt_run_decrypt(10, 0, 2) == 0"""
        )
    ]

def get_arrays_tests():
    return [
        (
            "array_sum",
            "Calculates the sum of an encrypted array. `elements` is `List[Any]`, compiled as `encrypted`.",
            """def test_array_sum(arr):
    return array_sum(arr)

compiler = fhe.Compiler(test_array_sum, {"arr": "encrypted"})
inputset = [[1, 2, 3, 4], [0, 0, 0, 0], [-1, 1, -1, 1]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2, 3, 4]) == 10
assert circuit.encrypt_run_decrypt([0, 0, 0, 0]) == 0
assert circuit.encrypt_run_decrypt([-1, 1, -1, 1]) == 0"""
        ),
        (
            "array_scale",
            "Multiplies an array by a scalar. `array` is `List[Any]` (`encrypted`), `factor` is `int` (`clear`).",
            """def test_array_scale(arr, factor):
    return array_scale(arr, factor)

compiler = fhe.Compiler(test_array_scale, {"arr": "encrypted", "factor": "clear"})
inputset = [([1, 2, 3], 3), ([0, 0, 0], 5), ([-1, 2, -3], -2)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 3], 3), [3, 6, 9])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0], 5), [0, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, 2, -3], -2), [2, -4, 6])"""
        ),
        (
            "array_add",
            "Element-wise addition. Both arrays are `List[Any]`, so both `encrypted`.",
            """def test_array_add(a1, a2):
    return array_add(a1, a2)

compiler = fhe.Compiler(test_array_add, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([1, 2], [3, 4]), ([-1, -2], [1, 2]), ([0, 0], [0, 0])]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2], [3, 4]), [4, 6])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, -2], [1, 2]), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0], [0, 0]), [0, 0])"""
        ),
        (
            "array_sub",
            "Element-wise subtraction. Both arrays `encrypted`.",
            """def test_array_sub(a1, a2):
    return array_sub(a1, a2)

compiler = fhe.Compiler(test_array_sub, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([5, 5], [2, 1]), ([1, 2], [3, 4]), ([0, 0], [0, 0])]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, 5], [2, 1]), [3, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2], [3, 4]), [-2, -2])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0], [0, 0]), [0, 0])"""
        ),
        (
            "array_multiply",
            "Element-wise multiplication. Both arrays `encrypted`.",
            """def test_array_multiply(a1, a2):
    return array_multiply(a1, a2)

compiler = fhe.Compiler(test_array_multiply, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([1, 2], [3, 4]), ([0, 5], [10, 0]), ([-1, -2], [3, 4])]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2], [3, 4]), [3, 8])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 5], [10, 0]), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, -2], [3, 4]), [-3, -8])"""
        ),
        (
            "array_pad",
            "Pads an array. `array` is `encrypted`, `target_size` defines list length so it must be `clear` during tracing.",
            """def test_array_pad(arr, size):
    return array_pad(arr, size)

compiler = fhe.Compiler(test_array_pad, {"arr": "encrypted", "size": "clear"})
inputset = [([1, 2], 4), ([-1, -1], 3)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2], 4), [1, 2, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, -1], 3), [-1, -1, 0])"""
        ),
        (
            "array_slice",
            "Slices an array. `array` is `encrypted`. Indices are used in python slices so they are `clear`.",
            """def test_array_slice(arr, start, end):
    return array_slice(arr, start, end)

compiler = fhe.Compiler(test_array_slice, {"arr": "encrypted", "start": "clear", "end": "clear"})
inputset = [([10, 20, 30, 40], 1, 3), ([0, 0, 0], 0, 2)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 20, 30, 40], 1, 3), [20, 30])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0], 0, 2), [0, 0])"""
        ),
        (
            "array_contains",
            "Checks if array contains value. `array` and `value` are `Any`, both `encrypted`.",
            """def test_array_contains(arr, val):
    return array_contains(arr, val)

compiler = fhe.Compiler(test_array_contains, {"arr": "encrypted", "val": "encrypted"})
inputset = [([1, 3, 5], 3), ([1, 3, 5], 2), ([0, 0, 0], 0)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 3, 5], 3) == 1
assert circuit.encrypt_run_decrypt([1, 3, 5], 2) == 0
assert circuit.encrypt_run_decrypt([0, 0, 0], 0) == 1"""
        ),
        (
            "array_count",
            "Counts occurrences of value. Both `encrypted`.",
            """def test_array_count(arr, val):
    return array_count(arr, val)

compiler = fhe.Compiler(test_array_count, {"arr": "encrypted", "val": "encrypted"})
inputset = [([1, 2, 2, 3], 2), ([1, 2, 2, 3], 5), ([0, 0, 0], 0)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2, 2, 3], 2) == 2
assert circuit.encrypt_run_decrypt([1, 2, 2, 3], 5) == 0
assert circuit.encrypt_run_decrypt([0, 0, 0], 0) == 3"""
        ),
        (
            "array_all_equal",
            "Checks if two arrays are identical. Both `encrypted`.",
            """def test_array_all_equal(a1, a2):
    return array_all_equal(a1, a2)

compiler = fhe.Compiler(test_array_all_equal, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([1, 2], [1, 2]), ([1, 2], [1, 3]), ([0, 0], [0, 0])]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2], [1, 2]) == 1
assert circuit.encrypt_run_decrypt([1, 2], [1, 3]) == 0
assert circuit.encrypt_run_decrypt([0, 0], [0, 0]) == 1"""
        ),
        (
            "make_compare_swap",
            "Creates ascending compare-swap. The returned function takes `x` and `y` (`Any`), both `encrypted`.",
            """swap = make_compare_swap(min_value=-50, max_value=50)

def test_swap(x, y):
    return swap(x, y)

compiler = fhe.Compiler(test_swap, {"x": "encrypted", "y": "encrypted"})
inputset = [(5, 3), (2, 4), (7, 7)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt(5, 3), [3, 5])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(2, 4), [2, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(7, 7), [7, 7])"""
        ),
        (
            "make_sort",
            "Creates a bitonic sorting network. The returned function takes `x` (`Any`), `encrypted`.",
            """sort_fn = make_sort(size=4, min_value=-50, max_value=50)

def test_sort(x):
    return sort_fn(x)

compiler = fhe.Compiler(test_sort, {"x": "encrypted"})
inputset = [[4, 1, 3, 2], [0, 0, 0, 0]]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([4, 1, 3, 2]), [1, 2, 3, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0, 0]), [0, 0, 0, 0])"""
        ),
        (
            "make_minimum",
            "Creates a minimum reduction function. The array `x` is `encrypted`.",
            """min_fn = make_minimum(size=4, min_value=-50, max_value=50)

def test_min(x):
    return min_fn(x)

compiler = fhe.Compiler(test_min, {"x": "encrypted"})
inputset = [[4, 1, 3, 2], [5, 5, 5, 5], [0, 8, 3, 9]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 1
assert circuit.encrypt_run_decrypt([5, 5, 5, 5]) == 5
assert circuit.encrypt_run_decrypt([0, 8, 3, 9]) == 0"""
        ),
        (
            "make_maximum",
            "Creates a maximum reduction function. `x` is `encrypted`.",
            """max_fn = make_maximum(size=4, min_value=-50, max_value=50)

def test_max(x):
    return max_fn(x)

compiler = fhe.Compiler(test_max, {"x": "encrypted"})
inputset = [[4, 1, 3, 2], [5, 5, 5, 5], [0, 8, 3, 9]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 4
assert circuit.encrypt_run_decrypt([5, 5, 5, 5]) == 5
assert circuit.encrypt_run_decrypt([0, 8, 3, 9]) == 9"""
        ),
        (
            "make_argmin",
            "Creates an argmin reduction. `x` is `encrypted`.",
            """argmin_fn = make_argmin(size=4, min_value=-50, max_value=50)

def test_argmin(x):
    return argmin_fn(x)

compiler = fhe.Compiler(test_argmin, {"x": "encrypted"})
inputset = [[4, 1, 3, 2], [4, 1, 1, 2]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 1
assert circuit.encrypt_run_decrypt([4, 1, 1, 2]) == 1"""
        ),
        (
            "make_argmax",
            "Creates an argmax reduction. `x` is `encrypted`.",
            """argmax_fn = make_argmax(size=4, min_value=-50, max_value=50)

def test_argmax(x):
    return argmax_fn(x)

compiler = fhe.Compiler(test_argmax, {"x": "encrypted"})
inputset = [[4, 1, 5, 2], [4, 5, 5, 2]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([4, 1, 5, 2]) == 2
assert circuit.encrypt_run_decrypt([4, 5, 5, 2]) == 1"""
        ),
        (
            "compile_compare_swap",
            "Returns a compiled compare-swap circuit.",
            """circuit = compile_compare_swap(min_value=-50, max_value=50)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt(5, 3), [3, 5])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(1, 4), [1, 4])"""
        ),
        (
            "compile_sort",
            "Returns a compiled bitonic sort circuit.",
            """circuit = compile_sort(size=4, min_value=-50, max_value=50)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([4, 1, 3, 2]), [1, 2, 3, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-5, 0, 5, -10]), [-10, -5, 0, 5])"""
        ),
        (
            "compile_minimum",
            "Returns a compiled minimum reduction circuit.",
            """circuit = compile_minimum(size=4, min_value=-50, max_value=50)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 1
assert circuit.encrypt_run_decrypt([5, 8, 9, 6]) == 5"""
        ),
        (
            "compile_maximum",
            "Returns a compiled maximum reduction circuit.",
            """circuit = compile_maximum(size=4, min_value=-50, max_value=50)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 4
assert circuit.encrypt_run_decrypt([1, 2, 0, 0]) == 2"""
        ),
        (
            "compile_argmin",
            "Returns a compiled argmin circuit.",
            """circuit = compile_argmin(size=4, min_value=-50, max_value=50)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 1
assert circuit.encrypt_run_decrypt([5, 3, 3, 6]) == 1"""
        ),
        (
            "compile_argmax",
            "Returns a compiled argmax circuit.",
            """circuit = compile_argmax(size=4, min_value=-50, max_value=50)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 0
assert circuit.encrypt_run_decrypt([1, 5, 5, 2]) == 1"""
        ),
        (
            "array_index",
            "Oblivious read. `array` and `index` are both `Any`, compiled as `encrypted`.",
            """def test_array_index(arr, idx):
    return array_index(arr, idx)

compiler = fhe.Compiler(test_array_index, {"arr": "encrypted", "idx": "encrypted"})
inputset = [([10, 20, 30], 1), ([10, 20, 30], 0), ([5, 5, 5], 2)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([10, 20, 30], 1) == 20
assert circuit.encrypt_run_decrypt([10, 20, 30], 0) == 10
assert circuit.encrypt_run_decrypt([5, 5, 5], 2) == 5"""
        ),
        (
            "array_set",
            "Oblivious write. `array`, `index`, `value` are all `encrypted`.",
            """def test_array_set(arr, idx, val):
    return array_set(arr, idx, val)

compiler = fhe.Compiler(test_array_set, {"arr": "encrypted", "idx": "encrypted", "val": "encrypted"})
inputset = [([10, 20, 30], 1, 99), ([10, 20, 30], 0, 50)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 20, 30], 1, 99), [10, 99, 30])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 20, 30], 0, 50), [50, 20, 30])"""
        ),
        (
            "array_index_of",
            "Returns index of value. `array` and `value` are `encrypted`, `missing_result` is `clear`.",
            """def test_array_index_of(arr, val, missing):
    return array_index_of(arr, val, missing_result=missing)

compiler = fhe.Compiler(test_array_index_of, {"arr": "encrypted", "val": "encrypted", "missing": "clear"})
inputset = [([10, 20, 30], 20, 99), ([10, 20, 30], 40, 99)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([10, 20, 30], 20, 99) == 1
assert circuit.encrypt_run_decrypt([10, 20, 30], 40, 99) == 99"""
        ),
        (
            "array_cumsum",
            "Prefix sums of array. `array` is `encrypted`.",
            """def test_array_cumsum(arr):
    return array_cumsum(arr)

compiler = fhe.Compiler(test_array_cumsum, {"arr": "encrypted"})
inputset = [[1, 2, 3], [0, 0, 0], [-1, 1, -1, 1]]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 3]), [1, 3, 6])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0]), [0, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, 1, -1, 1]), [-1, 0, -1, 0])"""
        ),
        (
            "array_reverse",
            "Reverses element order. `array` is `encrypted`.",
            """def test_array_reverse(arr):
    return array_reverse(arr)

compiler = fhe.Compiler(test_array_reverse, {"arr": "encrypted"})
inputset = [[1, 2, 3], [0, 0], [5, -2, 3]]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 3]), [3, 2, 1])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0]), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, -2, 3]), [3, -2, 5])"""
        ),
        (
            "array_concat",
            "Concatenates arrays. Since it takes `*arrays`, we wrap two arguments. Both are `encrypted`.",
            """def test_array_concat(a1, a2):
    return array_concat(a1, a2)

compiler = fhe.Compiler(test_array_concat, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([1, 2], [3, 4]), ([0], [0])]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2], [3, 4]), [1, 2, 3, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0], [0]), [0, 0])"""
        ),
        (
            "make_top_k",
            "Creates a top_k reduction. `x` is `encrypted`.",
            """top_k_fn = make_top_k(size=4, k=2, min_value=-50, max_value=50)

def test_top_k(x):
    return top_k_fn(x)

compiler = fhe.Compiler(test_top_k, {"x": "encrypted"})
inputset = [[4, 1, 3, 2], [10, 10, 0, -10]]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([4, 1, 3, 2]), [4, 3])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 10, 0, -10]), [10, 10])"""
        ),
        (
            "compile_top_k",
            "Returns a compiled top-k circuit.",
            """circuit = compile_top_k(size=4, k=2, min_value=-50, max_value=50)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([4, 1, 3, 2]), [4, 3])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 10, 0, -10]), [10, 10])"""
        )
    ]

def get_stats_tests():
    return [
        (
            "array_mean",
            "Computes floor mean. `array` is `encrypted`.",
            """def test_array_mean(arr):
    return array_mean(arr)

compiler = fhe.Compiler(test_array_mean, {"arr": "encrypted"})
inputset = [[2, 4, 6], [1, 1, 1, 1], [0, 5, 10]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([2, 4, 6]) == 4
assert circuit.encrypt_run_decrypt([1, 1, 1, 1]) == 1
assert circuit.encrypt_run_decrypt([0, 5, 10]) == 5"""
        ),
        (
            "array_variance",
            "Computes floor variance. `array` is `encrypted`.",
            """def test_array_variance(arr):
    return array_variance(arr)

compiler = fhe.Compiler(test_array_variance, {"arr": "encrypted"})
inputset = [[2, 4, 6], [5, 5, 5]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([2, 4, 6]) == 2
assert circuit.encrypt_run_decrypt([5, 5, 5]) == 0"""
        ),
        (
            "array_std",
            "Computes integer standard deviation. `array` is `encrypted`, bounds are `clear`.",
            """def test_array_std(arr, min_val, max_val):
    return array_std(arr, min_val, max_val)

compiler = fhe.Compiler(test_array_std, {"arr": "encrypted", "min_val": "clear", "max_val": "clear"})
inputset = [([2, 4, 6], -50, 50), ([5, 5, 5], -50, 50)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([2, 4, 6], -50, 50) == 1
assert circuit.encrypt_run_decrypt([5, 5, 5], -50, 50) == 0"""
        ),
        (
            "array_covariance",
            "Computes covariance of two arrays. Both `encrypted`.",
            """def test_array_covariance(a1, a2):
    return array_covariance(a1, a2)

compiler = fhe.Compiler(test_array_covariance, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([1, 2, 3], [1, 2, 3]), ([2, 4, 6], [2, 4, 6])]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2, 3], [1, 2, 3]) == 0
assert circuit.encrypt_run_decrypt([2, 4, 6], [2, 4, 6]) == 2"""
        ),
        (
            "array_max",
            "Finds maximum value. `elements` is `encrypted`.",
            """def test_array_max(arr):
    return array_max(arr)

compiler = fhe.Compiler(test_array_max, {"arr": "encrypted"})
inputset = [[1, 5, 3], [7, 7, 7], [0, -2, 2]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3]) == 5
assert circuit.encrypt_run_decrypt([7, 7, 7]) == 7
assert circuit.encrypt_run_decrypt([0, -2, 2]) == 2"""
        ),
        (
            "array_min",
            "Finds minimum value. `elements` is `encrypted`.",
            """def test_array_min(arr):
    return array_min(arr)

compiler = fhe.Compiler(test_array_min, {"arr": "encrypted"})
inputset = [[1, 5, 3], [7, 7, 7], [0, -2, 2]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3]) == 1
assert circuit.encrypt_run_decrypt([7, 7, 7]) == 7
assert circuit.encrypt_run_decrypt([0, -2, 2]) == -2"""
        ),
        (
            "array_range",
            "Calculates the range (max - min). `array` is `encrypted`.",
            """def test_array_range(arr):
    return array_range(arr)

compiler = fhe.Compiler(test_array_range, {"arr": "encrypted"})
inputset = [[1, 5, 3], [7, 7, 7], [-5, 0, 5]]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3]) == 4
assert circuit.encrypt_run_decrypt([7, 7, 7]) == 0
assert circuit.encrypt_run_decrypt([-5, 0, 5]) == 10"""
        ),
        (
            "array_count_greater",
            "Counts elements strictly greater than threshold. `array` and `threshold` are `encrypted`.",
            """def test_array_count_greater(arr, thresh):
    return array_count_greater(arr, thresh)

compiler = fhe.Compiler(test_array_count_greater, {"arr": "encrypted", "thresh": "encrypted"})
inputset = [([1, 5, 3], 2), ([1, 5, 3], 5), ([1, 5, 3], 0)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3], 2) == 2
assert circuit.encrypt_run_decrypt([1, 5, 3], 5) == 0
assert circuit.encrypt_run_decrypt([1, 5, 3], 0) == 3"""
        ),
        (
            "array_median",
            "Computes median. `array` is `encrypted`, bounds are `clear`.",
            """def test_array_median(arr, min_val, max_val):
    return array_median(arr, min_val, max_val)

compiler = fhe.Compiler(test_array_median, {"arr": "encrypted", "min_val": "clear", "max_val": "clear"})
inputset = [([1, 5, 3, 4], -50, 50), ([0, 0, 2, 2], -50, 50)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3, 4], -50, 50) == 3
assert circuit.encrypt_run_decrypt([0, 0, 2, 2], -50, 50) == 1"""
        ),
        (
            "array_percentile",
            "Calculates q-th percentile. `array` is `encrypted`, other params are `clear`.",
            """def test_array_percentile(arr, q, min_val, max_val):
    return array_percentile(arr, q, min_val, max_val)

compiler = fhe.Compiler(test_array_percentile, {"arr": "encrypted", "q": "clear", "min_val": "clear", "max_val": "clear"})
inputset = [([1, 5, 3, 4], 50, -50, 50), ([1, 5, 3, 4], 100, -50, 50)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3, 4], 50, -50, 50) == 4
assert circuit.encrypt_run_decrypt([1, 5, 3, 4], 100, -50, 50) == 5"""
        ),
        (
            "array_histogram",
            "Counts occurrences in range. `array` is `encrypted`, bounds are `clear`.",
            """def test_array_histogram(arr, min_val, max_val):
    return array_histogram(arr, min_val, max_val)

compiler = fhe.Compiler(test_array_histogram, {"arr": "encrypted", "min_val": "clear", "max_val": "clear"})
inputset = [([1, 2, 1], 1, 3), ([2, 2, 2], 1, 3)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 1], 1, 3), [2, 1, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([2, 2, 2], 1, 3), [0, 3, 0])"""
        ),
        (
            "array_mode",
            "Returns most frequent value. `array` is `encrypted`, bounds are `clear`.",
            """def test_array_mode(arr, min_val, max_val):
    return array_mode(arr, min_val, max_val)

compiler = fhe.Compiler(test_array_mode, {"arr": "encrypted", "min_val": "clear", "max_val": "clear"})
inputset = [([1, 2, 2, 3], 1, 5), ([1, 1, 3, 3], 1, 5)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2, 2, 3], 1, 5) == 2
assert circuit.encrypt_run_decrypt([1, 1, 3, 3], 1, 5) == 1"""
        ),
        (
            "array_normalize",
            "Normalizes array elements. `array` and `mean` are `encrypted`, `scale` is `clear`.",
            """def test_array_normalize(arr, mean, scale):
    return array_normalize(arr, mean, scale)

compiler = fhe.Compiler(test_array_normalize, {"arr": "encrypted", "mean": "encrypted", "scale": "clear"})
inputset = [([2, 4, 6], 4, 10), ([0, 0, 0], 0, 5)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([2, 4, 6], 4, 10), [-20, 0, 20])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0], 0, 5), [0, 0, 0])"""
        )
    ]

if __name__ == "__main__":
    generate_notebook("docs/tutorials/01_arithmetic.ipynb", "concrete_fhe_toolkit.arithmetic", get_arithmetic_tests())
    generate_notebook("docs/tutorials/02_arrays.ipynb", "concrete_fhe_toolkit.arrays", get_arrays_tests())
    generate_notebook("docs/tutorials/03_stats.ipynb", "concrete_fhe_toolkit.stats", get_stats_tests())
