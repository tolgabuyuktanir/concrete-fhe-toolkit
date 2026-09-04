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
        imports = f"from concrete import fhe\nfrom {module_name} import {fn_name}\n\n"
        if "np." in test_code:
            imports = "import numpy as np\n" + imports
        
        cells.append(create_cell("code", f"{imports}{test_code}\nprint(\"{fn_name} tests passed!\")"))
        
    save_notebook(filename, cells)

def get_arithmetic_tests():
    return [
        (
            "compare",
            "This function compares two values and returns 1 if x > y, 0 if x == y, and -1 if x < y. Includes edge cases like negatives and zero.",
            """def test_compare(x, y):
    return compare(x, y)

compiler = fhe.Compiler(test_compare, {"x": "encrypted", "y": "encrypted"})
inputset = [(15, 10), (10, 10), (-15, 10), (0, 0), (-5, -10), (-10, -5)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt(15, 10) == 1
assert circuit.encrypt_run_decrypt(10, 10) == 0
assert circuit.encrypt_run_decrypt(-15, 10) == -1
assert circuit.encrypt_run_decrypt(0, 0) == 0
assert circuit.encrypt_run_decrypt(-5, -10) == 1
assert circuit.encrypt_run_decrypt(-10, -5) == -1"""
        ),
        (
            "compile_compare",
            "Returns an already compiled FHE circuit to compare two encrypted values. Tests positives, negatives, and zeros.",
            """circuit = compile_compare(min_value=-15, max_value=15)

assert circuit.encrypt_run_decrypt(10, 5) == 1
assert circuit.encrypt_run_decrypt(12, 12) == 0
assert circuit.encrypt_run_decrypt(-10, 15) == -1
assert circuit.encrypt_run_decrypt(0, 0) == 0
assert circuit.encrypt_run_decrypt(-15, -15) == 0"""
        ),
        (
            "sign",
            "Returns the sign of a number (1, -1, or 0). Tests positives, negatives, and zero.",
            """def test_sign(x):
    return sign(x)

compiler = fhe.Compiler(test_sign, {"x": "encrypted"})
inputset = [(15,), (-15,), (0,), (1,), (-1,)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt(15) == 1
assert circuit.encrypt_run_decrypt(-15) == -1
assert circuit.encrypt_run_decrypt(0) == 0
assert circuit.encrypt_run_decrypt(1) == 1
assert circuit.encrypt_run_decrypt(-1) == -1"""
        ),
        (
            "compile_sign",
            "Returns a compiled FHE circuit to return the sign of a number.",
            """circuit = compile_sign(min_value=-15, max_value=15)

assert circuit.encrypt_run_decrypt(10) == 1
assert circuit.encrypt_run_decrypt(-10) == -1
assert circuit.encrypt_run_decrypt(0) == 0"""
        ),
        (
            "make_floor_divide",
            "Creates an exact encrypted floor division. Tests exact division, floor division, negatives, and division by zero. Bounds are kept very small to prevent FHE bivariate table lookup from exploding RAM.",
            """floor_divide = make_floor_divide(zero_result=7)

def test_floor_divide(num, den):
    return floor_divide(num, den)

compiler = fhe.Compiler(test_floor_divide, {"num": "encrypted", "den": "encrypted"})
inputset = [(4, 2), (4, -2), (-4, 2), (-4, -2), (3, 0), (0, 3)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt(4, 2) == 2
assert circuit.encrypt_run_decrypt(4, -2) == -2
assert circuit.encrypt_run_decrypt(-4, 2) == -2
assert circuit.encrypt_run_decrypt(-4, -2) == 2
assert circuit.encrypt_run_decrypt(3, 0) == 7
assert circuit.encrypt_run_decrypt(0, 3) == 0"""
        ),
        (
            "make_floor_divide_by_product",
            "Creates a division by a product (numerator // (left * right)). Tests positives, negatives, zeroes.",
            """div_prod = make_floor_divide_by_product(zero_result=7)

def test_div_prod(num, left, right):
    return div_prod(num, left, right)

compiler = fhe.Compiler(test_div_prod, {"num": "encrypted", "left": "encrypted", "right": "encrypted"})
inputset = [(4, 2, 1), (-4, 2, -1), (4, -2, 1), (4, 0, 1), (0, 2, 1)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt(4, 2, 1) == 2
assert circuit.encrypt_run_decrypt(-4, 2, -1) == 2
assert circuit.encrypt_run_decrypt(4, -2, 1) == -2
assert circuit.encrypt_run_decrypt(4, 0, 1) == 7
assert circuit.encrypt_run_decrypt(0, 2, 1) == 0"""
        ),
        (
            "compile_floor_divide",
            "Returns a compiled floor division circuit. Note: This function only supports nonnegative inputs by design.",
            """circuit = compile_floor_divide(max_numerator=5, max_denominator=3, zero_result=7)

assert circuit.encrypt_run_decrypt(5, 2) == 2
assert circuit.encrypt_run_decrypt(4, 3) == 1
assert circuit.encrypt_run_decrypt(5, 0) == 7
assert circuit.encrypt_run_decrypt(0, 2) == 0"""
        ),
        (
            "compile_floor_divide_by_product",
            "Returns a compiled circuit for division by a product. Note: This function only supports nonnegative inputs by design.",
            """circuit = compile_floor_divide_by_product(max_numerator=5, max_left=2, max_right=2, zero_result=7)

assert circuit.encrypt_run_decrypt(4, 2, 1) == 2
assert circuit.encrypt_run_decrypt(5, 0, 2) == 7
assert circuit.encrypt_run_decrypt(0, 2, 2) == 0"""
        )
    ]

def get_arrays_tests():
    return [
        (
            "array_sum",
            "Calculates the sum of an encrypted array. Tests positive, negative, zero, and alternating arrays.",
            """def test_array_sum(arr):
    return array_sum(arr)

compiler = fhe.Compiler(test_array_sum, {"arr": "encrypted"})
inputset = [([5, 5, 5, 5],), ([0, 0, 0, 0],), ([-5, 5, -5, 5],), ([-2, -3, -4, -1],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([5, 5, 5, 5]) == 20
assert circuit.encrypt_run_decrypt([0, 0, 0, 0]) == 0
assert circuit.encrypt_run_decrypt([-5, 5, -5, 5]) == 0
assert circuit.encrypt_run_decrypt([-2, -3, -4, -1]) == -10"""
        ),
        (
            "array_scale",
            "Multiplies an array by a scalar. Tests zero scale, negative scale, and positive scale.",
            """def test_array_scale(arr, factor):
    return array_scale(arr, factor)

compiler = fhe.Compiler(test_array_scale, {"arr": "encrypted", "factor": "clear"})
inputset = [([1, 2, 3], 3), ([0, 0, 0], 5), ([-1, 2, -3], -2), ([5, 10, 15], 0)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 3], 3), [3, 6, 9])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0], 5), [0, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, 2, -3], -2), [2, -4, 6])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, 10, 15], 0), [0, 0, 0])"""
        ),
        (
            "array_add",
            "Element-wise addition. Tests normal addition, negative addition, and adding zeros.",
            """def test_array_add(a1, a2):
    return array_add(a1, a2)

compiler = fhe.Compiler(test_array_add, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([5, 10], [10, 5]), ([-5, -10], [5, 10]), ([0, 0], [0, 0]), ([5, 5], [-5, -5])]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, 10], [10, 5]), [15, 15])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-5, -10], [5, 10]), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0], [0, 0]), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, 5], [-5, -5]), [0, 0])"""
        ),
        (
            "array_sub",
            "Element-wise subtraction. Tests identical arrays, negative results, and zero arrays.",
            """def test_array_sub(a1, a2):
    return array_sub(a1, a2)

compiler = fhe.Compiler(test_array_sub, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([15, 15], [5, 10]), ([5, 10], [15, 20]), ([0, 0], [0, 0]), ([5, 10], [5, 10]), ([-5, -5], [5, 5])]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([15, 15], [5, 10]), [10, 5])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, 10], [15, 20]), [-10, -10])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0], [0, 0]), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, 10], [5, 10]), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-5, -5], [5, 5]), [-10, -10])"""
        ),
        (
            "array_multiply",
            "Element-wise multiplication. Tests zeros, negatives, and normal positive cases.",
            """def test_array_multiply(a1, a2):
    return array_multiply(a1, a2)

compiler = fhe.Compiler(test_array_multiply, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([2, 3], [4, 5]), ([0, 5], [10, 0]), ([-2, -3], [4, -5])]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([2, 3], [4, 5]), [8, 15])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 5], [10, 0]), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-2, -3], [4, -5]), [-8, 15])"""
        ),
        (
            "array_pad",
            "Pads or truncates an array to exactly `target_size`. The wrapper strictly enforces size 4. We test arrays that are smaller, exact size, and larger (truncation).",
            """def test_pad_from_2_to_4(arr):
    return array_pad(arr, 4)
compiler2 = fhe.Compiler(test_pad_from_2_to_4, {"arr": "encrypted"})
circuit2 = compiler2.compile([([1, 2],), ([-1, -1],)])

def test_pad_from_5_to_4(arr):
    return array_pad(arr, 4)
compiler5 = fhe.Compiler(test_pad_from_5_to_4, {"arr": "encrypted"})
circuit5 = compiler5.compile([([1, 2, 3, 4, 5],), ([0, 0, 0, 0, 0],)])

def test_pad_from_4_to_4(arr):
    return array_pad(arr, 4)
compiler4 = fhe.Compiler(test_pad_from_4_to_4, {"arr": "encrypted"})
circuit4 = compiler4.compile([([1, 2, 3, 4],), ([0, 0, 0, 0],)])

np.testing.assert_array_equal(circuit2.encrypt_run_decrypt([1, 2]), [1, 2, 0, 0])
np.testing.assert_array_equal(circuit5.encrypt_run_decrypt([1, 2, 3, 4, 5]), [1, 2, 3, 4])
np.testing.assert_array_equal(circuit4.encrypt_run_decrypt([1, 2, 3, 4]), [1, 2, 3, 4])"""
        ),
        (
            "array_slice",
            "Slices an array. Tests standard slice, full slice, and negative index slice (safely constrained by python slice semantics).",
            """def test_array_slice_1_3(arr):
    return array_slice(arr, 1, 3)

def test_array_slice_all(arr):
    return array_slice(arr, 0, 4)

def test_array_slice_negative(arr):
    return array_slice(arr, -3, -1)

compiler1 = fhe.Compiler(test_array_slice_1_3, {"arr": "encrypted"})
circuit1 = compiler1.compile([([10, 20, 30, 40],), ([0, 0, 0, 0],)])

compiler2 = fhe.Compiler(test_array_slice_all, {"arr": "encrypted"})
circuit2 = compiler2.compile([([10, 20, 30, 40],), ([0, 0, 0, 0],)])

compiler3 = fhe.Compiler(test_array_slice_negative, {"arr": "encrypted"})
circuit3 = compiler3.compile([([10, 20, 30, 40],), ([0, 0, 0, 0],)])

np.testing.assert_array_equal(circuit1.encrypt_run_decrypt([10, 20, 30, 40]), [20, 30])
np.testing.assert_array_equal(circuit2.encrypt_run_decrypt([10, 20, 30, 40]), [10, 20, 30, 40])
np.testing.assert_array_equal(circuit3.encrypt_run_decrypt([10, 20, 30, 40]), [20, 30])"""
        ),
        (
            "array_contains",
            "Checks if array contains value. Tests exists, does not exist, exists multiple times, negatives, zeroes.",
            """def test_array_contains(arr, val):
    return array_contains(arr, val)

compiler = fhe.Compiler(test_array_contains, {"arr": "encrypted", "val": "encrypted"})
inputset = [([1, 3, 5], 3), ([1, 3, 5], 2), ([0, 0, 0], 0), ([-2, -2, -2], -2), ([-1, 1, 2], -1)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 3, 5], 3) == 1
assert circuit.encrypt_run_decrypt([1, 3, 5], 2) == 0
assert circuit.encrypt_run_decrypt([0, 0, 0], 0) == 1
assert circuit.encrypt_run_decrypt([-2, -2, -2], -2) == 1
assert circuit.encrypt_run_decrypt([-1, 1, 2], -1) == 1"""
        ),
        (
            "array_count",
            "Counts occurrences of value. Tests zero count, single count, multiple count, negatives and zeroes.",
            """def test_array_count(arr, val):
    return array_count(arr, val)

compiler = fhe.Compiler(test_array_count, {"arr": "encrypted", "val": "encrypted"})
inputset = [([1, 2, 2, 3], 2), ([1, 2, 2, 3], 5), ([0, 0, 0], 0), ([-5, -6, -5, -8], -5)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2, 2, 3], 2) == 2
assert circuit.encrypt_run_decrypt([1, 2, 2, 3], 5) == 0
assert circuit.encrypt_run_decrypt([0, 0, 0], 0) == 3
assert circuit.encrypt_run_decrypt([-5, -6, -5, -8], -5) == 2"""
        ),
        (
            "array_all_equal",
            "Checks if two arrays are identical. Tests identical, mismatch at start, mismatch at end, completely different, negatives.",
            """def test_array_all_equal(a1, a2):
    return array_all_equal(a1, a2)

compiler = fhe.Compiler(test_array_all_equal, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [9, 2, 3]), ([1, 2, 3], [1, 2, 9]), ([0, 0, 0], [1, 1, 1]), ([-1, -2], [-1, -2])]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2, 3], [1, 2, 3]) == 1
assert circuit.encrypt_run_decrypt([1, 2, 3], [9, 2, 3]) == 0
assert circuit.encrypt_run_decrypt([1, 2, 3], [1, 2, 9]) == 0
assert circuit.encrypt_run_decrypt([0, 0, 0], [1, 1, 1]) == 0
assert circuit.encrypt_run_decrypt([-1, -2], [-1, -2]) == 1"""
        ),
        (
            "make_compare_swap",
            "Creates ascending compare-swap. Tests x > y, x < y, x == y, zeroes and negative inputs.",
            """swap = make_compare_swap(min_value=-15, max_value=15)

def test_swap(x, y):
    return swap(x, y)

compiler = fhe.Compiler(test_swap, {"x": "encrypted", "y": "encrypted"})
inputset = [(5, 3), (2, 4), (7, 7), (-5, -10), (0, 0), (-2, 5)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt(5, 3), [3, 5])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(2, 4), [2, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(7, 7), [7, 7])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(-5, -10), [-10, -5])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(0, 0), [0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(-2, 5), [-2, 5])"""
        ),
        (
            "make_sort",
            "Creates a bitonic sorting network. Tests reverse sorted, already sorted, duplicates, zeroes and negatives.",
            """sort_fn = make_sort(size=4, min_value=-15, max_value=15)

def test_sort(x):
    return sort_fn(x)

compiler = fhe.Compiler(test_sort, {"x": "encrypted"})
inputset = [([4, 3, 2, 1],), ([1, 2, 3, 4],), ([5, 5, 5, 5],), ([0, -5, -2, 3],), ([0, 0, 0, 0],)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([4, 3, 2, 1]), [1, 2, 3, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 3, 4]), [1, 2, 3, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, 5, 5, 5]), [5, 5, 5, 5])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, -5, -2, 3]), [-5, -2, 0, 3])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0, 0]), [0, 0, 0, 0])"""
        ),
        (
            "make_minimum",
            "Creates a minimum reduction function. Tests min at start, min at end, duplicates, zeroes and negatives.",
            """min_fn = make_minimum(size=4, min_value=-15, max_value=15)

def test_min(x):
    return min_fn(x)

compiler = fhe.Compiler(test_min, {"x": "encrypted"})
inputset = [([1, 4, 3, 2],), ([5, 5, 5, 5],), ([8, 9, 3, 0],), ([-1, -5, -2, -3],), ([0, 0, 0, 0],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 4, 3, 2]) == 1
assert circuit.encrypt_run_decrypt([5, 5, 5, 5]) == 5
assert circuit.encrypt_run_decrypt([8, 9, 3, 0]) == 0
assert circuit.encrypt_run_decrypt([-1, -5, -2, -3]) == -5
assert circuit.encrypt_run_decrypt([0, 0, 0, 0]) == 0"""
        ),
        (
            "make_maximum",
            "Creates a maximum reduction function. Tests max at start, max at end, duplicates, zeroes and negatives.",
            """max_fn = make_maximum(size=4, min_value=-15, max_value=15)

def test_max(x):
    return max_fn(x)

compiler = fhe.Compiler(test_max, {"x": "encrypted"})
inputset = [([4, 1, 3, 2],), ([5, 5, 5, 5],), ([0, 8, 3, 9],), ([-1, -5, -2, -3],), ([0, 0, 0, 0],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 4
assert circuit.encrypt_run_decrypt([5, 5, 5, 5]) == 5
assert circuit.encrypt_run_decrypt([0, 8, 3, 9]) == 9
assert circuit.encrypt_run_decrypt([-1, -5, -2, -3]) == -1
assert circuit.encrypt_run_decrypt([0, 0, 0, 0]) == 0"""
        ),
        (
            "make_argmin",
            "Creates an argmin reduction. Tests argmin at start, at end, with duplicates (returns first index) and negatives.",
            """argmin_fn = make_argmin(size=4, min_value=-15, max_value=15)

def test_argmin(x):
    return argmin_fn(x)

compiler = fhe.Compiler(test_argmin, {"x": "encrypted"})
inputset = [([1, 4, 3, 2],), ([4, 1, 1, 2],), ([9, 8, 7, 0],), ([-5, -1, -2, -3],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 4, 3, 2]) == 0
assert circuit.encrypt_run_decrypt([4, 1, 1, 2]) == 1
assert circuit.encrypt_run_decrypt([9, 8, 7, 0]) == 3
assert circuit.encrypt_run_decrypt([-5, -1, -2, -3]) == 0"""
        ),
        (
            "make_argmax",
            "Creates an argmax reduction. Tests argmax at start, at end, with duplicates (returns first index) and negatives.",
            """argmax_fn = make_argmax(size=4, min_value=-15, max_value=15)

def test_argmax(x):
    return argmax_fn(x)

compiler = fhe.Compiler(test_argmax, {"x": "encrypted"})
inputset = [([9, 1, 5, 2],), ([4, 9, 9, 2],), ([0, 1, 2, 9],), ([-5, -1, -2, -3],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([9, 1, 5, 2]) == 0
assert circuit.encrypt_run_decrypt([4, 9, 9, 2]) == 1
assert circuit.encrypt_run_decrypt([0, 1, 2, 9]) == 3
assert circuit.encrypt_run_decrypt([-5, -1, -2, -3]) == 1"""
        ),
        (
            "compile_compare_swap",
            "Returns a compiled compare-swap circuit.",
            """circuit = compile_compare_swap(min_value=-15, max_value=15)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt(5, 3), [3, 5])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt(1, 4), [1, 4])"""
        ),
        (
            "compile_sort",
            "Returns a compiled bitonic sort circuit.",
            """circuit = compile_sort(size=4, min_value=-15, max_value=15)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([4, 1, 3, 2]), [1, 2, 3, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-5, 0, 5, -10]), [-10, -5, 0, 5])"""
        ),
        (
            "compile_minimum",
            "Returns a compiled minimum reduction circuit.",
            """circuit = compile_minimum(size=4, min_value=-15, max_value=15)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 1
assert circuit.encrypt_run_decrypt([5, 8, 9, 6]) == 5"""
        ),
        (
            "compile_maximum",
            "Returns a compiled maximum reduction circuit.",
            """circuit = compile_maximum(size=4, min_value=-15, max_value=15)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 4
assert circuit.encrypt_run_decrypt([1, 2, 0, 0]) == 2"""
        ),
        (
            "compile_argmin",
            "Returns a compiled argmin circuit.",
            """circuit = compile_argmin(size=4, min_value=-15, max_value=15)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 1
assert circuit.encrypt_run_decrypt([5, 3, 3, 6]) == 1"""
        ),
        (
            "compile_argmax",
            "Returns a compiled argmax circuit.",
            """circuit = compile_argmax(size=4, min_value=-15, max_value=15)

assert circuit.encrypt_run_decrypt([4, 1, 3, 2]) == 0
assert circuit.encrypt_run_decrypt([1, 5, 5, 2]) == 1"""
        ),
        (
            "array_index",
            "Oblivious read. Tests reading from start, middle, and end of the array. Out of bounds triggers Python IndexError, so not FHE-compatible in clear.",
            """def test_array_index(arr, idx):
    return array_index(arr, idx)

compiler = fhe.Compiler(test_array_index, {"arr": "encrypted", "idx": "encrypted"})
inputset = [([10, 20, 30], 1), ([10, 20, 30], 0), ([5, 5, 5], 2)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([10, 20, 30], 1) == 20
assert circuit.encrypt_run_decrypt([10, 20, 30], 0) == 10
assert circuit.encrypt_run_decrypt([10, 20, 30], 2) == 30"""
        ),
        (
            "array_set",
            "Oblivious write. Tests writing to start, middle, and end of the array with positive, zeroes, negatives.",
            """def test_array_set(arr, idx, val):
    return array_set(arr, idx, val)

compiler = fhe.Compiler(test_array_set, {"arr": "encrypted", "idx": "encrypted", "val": "encrypted"})
inputset = [([10, 20, 30], 1, 15), ([10, 20, 30], 0, -5), ([10, 20, 30], 2, 0)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 20, 30], 1, 15), [10, 15, 30])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 20, 30], 0, -5), [-5, 20, 30])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 20, 30], 2, 0), [10, 20, 0])"""
        ),
        (
            "array_index_of",
            "Returns index of value. Tests existing element, missing element (returns `missing_result`), first occurrence of duplicates, and zeroes.",
            """def test_array_index_of(arr, val, missing):
    return array_index_of(arr, val, missing_result=missing)

compiler = fhe.Compiler(test_array_index_of, {"arr": "encrypted", "val": "encrypted", "missing": "clear"})
inputset = [([10, 20, 30, 20], 20, 99), ([10, 20, 30, 20], 40, 99), ([10, 20, 30, 20], 30, 99), ([0, 0, 0, 0], 0, 99)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([10, 20, 30, 20], 20, 99) == 1
assert circuit.encrypt_run_decrypt([10, 20, 30, 20], 40, 99) == 99
assert circuit.encrypt_run_decrypt([10, 20, 30, 20], 30, 99) == 2
assert circuit.encrypt_run_decrypt([0, 0, 0, 0], 0, 99) == 0"""
        ),
        (
            "array_cumsum",
            "Prefix sums of array. Tests positive arrays, zero arrays, negative arrays, and mixed arrays.",
            """def test_array_cumsum(arr):
    return array_cumsum(arr)

compiler = fhe.Compiler(test_array_cumsum, {"arr": "encrypted"})
inputset = [([1, 2, 3],), ([0, 0, 0],), ([-1, 1, -1, 1],), ([5, -2, -3],), ([-2, -3, -4],)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 3]), [1, 3, 6])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0]), [0, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, 1, -1, 1]), [-1, 0, -1, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, -2, -3]), [5, 3, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-2, -3, -4]), [-2, -5, -9])"""
        ),
        (
            "array_reverse",
            "Reverses element order. Tests even length, odd length (implicitly by padding in FHE, but FHE arrays are fixed shape, so we test multiple static shapes).",
            """def test_array_reverse(arr):
    return array_reverse(arr)

compiler = fhe.Compiler(test_array_reverse, {"arr": "encrypted"})
inputset = [([1, 2, 3, 4],), ([0, 0, 0, 0],), ([5, -2, 3, 1],)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 3, 4]), [4, 3, 2, 1])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0, 0]), [0, 0, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([5, -2, 3, 1]), [1, 3, -2, 5])

# Test odd length
def test_array_reverse_odd(arr):
    return array_reverse(arr)
compiler_odd = fhe.Compiler(test_array_reverse_odd, {"arr": "encrypted"})
circuit_odd = compiler_odd.compile([([1, 2, 3],), ([-1, -2, -3],)])
np.testing.assert_array_equal(circuit_odd.encrypt_run_decrypt([1, 2, 3]), [3, 2, 1])"""
        ),
        (
            "array_concat",
            "Concatenates arrays. Tests normal concatenation, zeroes, negatives.",
            """def test_array_concat(a1, a2):
    return array_concat(a1, a2)

compiler = fhe.Compiler(test_array_concat, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([1, 2], [3, 4]), ([0, 0], [0, 0]), ([-1, 1], [2, -2])]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2], [3, 4]), [1, 2, 3, 4])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0], [0, 0]), [0, 0, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, 1], [2, -2]), [-1, 1, 2, -2])"""
        ),
        (
            "make_top_k",
            "Creates a top_k reduction. Tests sorted, unsorted, zeroes, negatives, and duplicates.",
            """top_k_fn = make_top_k(size=4, k=2, min_value=-15, max_value=15)

def test_top_k(x):
    return top_k_fn(x)

compiler = fhe.Compiler(test_top_k, {"x": "encrypted"})
inputset = [([4, 1, 3, 2],), ([10, 10, 0, -10],), ([1, 2, 3, 4],), ([-1, -2, -3, -4],), ([0, 0, 0, 0],)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([4, 1, 3, 2]), [4, 3])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 10, 0, -10]), [10, 10])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 3, 4]), [4, 3])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-1, -2, -3, -4]), [-1, -2])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0, 0]), [0, 0])"""
        ),
        (
            "compile_top_k",
            "Returns a compiled top-k circuit.",
            """circuit = compile_top_k(size=4, k=2, min_value=-15, max_value=15)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([4, 1, 3, 2]), [4, 3])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 10, 0, -10]), [10, 10])"""
        )
    ]

def get_stats_tests():
    return [
        (
            "array_mean",
            "Computes floor mean. Tests exact divisibility, flooring, zeroes, and negatives.",
            """def test_array_mean(arr):
    return array_mean(arr)

compiler = fhe.Compiler(test_array_mean, {"arr": "encrypted"})
inputset = [([2, 4, 6],), ([1, 1, 1],), ([0, 5, 10],), ([0, 0, 0],), ([-2, -4, -6],), ([-1, 1, 0],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([2, 4, 6]) == 4
assert circuit.encrypt_run_decrypt([1, 1, 1]) == 1
assert circuit.encrypt_run_decrypt([0, 5, 10]) == 5
assert circuit.encrypt_run_decrypt([0, 0, 0]) == 0
assert circuit.encrypt_run_decrypt([-2, -4, -6]) == -4
assert circuit.encrypt_run_decrypt([-1, 1, 0]) == 0"""
        ),
        (
            "array_variance",
            "Computes floor variance. Tests zero variance (all same elements), positive variance, zeroes, and negatives.",
            """def test_array_variance(arr):
    return array_variance(arr)

compiler = fhe.Compiler(test_array_variance, {"arr": "encrypted"})
inputset = [([2, 4, 6],), ([5, 5, 5],), ([1, 5, 9],), ([0, 0, 0],), ([-2, -4, -6],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([2, 4, 6]) == 2
assert circuit.encrypt_run_decrypt([5, 5, 5]) == 0
assert circuit.encrypt_run_decrypt([1, 5, 9]) == 10
assert circuit.encrypt_run_decrypt([0, 0, 0]) == 0
assert circuit.encrypt_run_decrypt([-2, -4, -6]) == 2"""
        ),
        (
            "array_std",
            "Computes integer standard deviation. Tests zero std, positive std, zeroes and negatives.",
            """def test_array_std(arr, min_val, max_val):
    return array_std(arr, min_val, max_val)

compiler = fhe.Compiler(test_array_std, {"arr": "encrypted", "min_val": "clear", "max_val": "clear"})
inputset = [([2, 4, 6], -15, 15), ([5, 5, 5], -15, 15), ([1, 5, 9], -15, 15), ([0, 0, 0], -15, 15), ([-2, -4, -6], -15, 15)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([2, 4, 6], -15, 15) == 1
assert circuit.encrypt_run_decrypt([5, 5, 5], -15, 15) == 0
assert circuit.encrypt_run_decrypt([1, 5, 9], -15, 15) == 3
assert circuit.encrypt_run_decrypt([0, 0, 0], -15, 15) == 0
assert circuit.encrypt_run_decrypt([-2, -4, -6], -15, 15) == 1"""
        ),
        (
            "array_covariance",
            "Computes covariance of two arrays. Tests identical (positive cov), inverse (negative cov), zeroes, and constants (zero cov).",
            """def test_array_covariance(a1, a2):
    return array_covariance(a1, a2)

compiler = fhe.Compiler(test_array_covariance, {"a1": "encrypted", "a2": "encrypted"})
inputset = [([1, 2, 3], [1, 2, 3]), ([2, 4, 6], [2, 4, 6]), ([1, 2, 3], [3, 2, 1]), ([5, 5, 5], [1, 2, 3]), ([0, 0, 0], [0, 0, 0]), ([-1, -2, -3], [-1, -2, -3])]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2, 3], [1, 2, 3]) == 0
assert circuit.encrypt_run_decrypt([2, 4, 6], [2, 4, 6]) == 2
assert circuit.encrypt_run_decrypt([1, 2, 3], [3, 2, 1]) == 0
assert circuit.encrypt_run_decrypt([5, 5, 5], [1, 2, 3]) == 0
assert circuit.encrypt_run_decrypt([0, 0, 0], [0, 0, 0]) == 0
assert circuit.encrypt_run_decrypt([-1, -2, -3], [-1, -2, -3]) == 0"""
        ),
        (
            "array_max",
            "Finds maximum value. Tests max at different positions, duplicates, zeroes and negatives.",
            """def test_array_max(arr):
    return array_max(arr)

compiler = fhe.Compiler(test_array_max, {"arr": "encrypted"})
inputset = [([1, 5, 3],), ([7, 7, 7],), ([0, -2, 2],), ([9, 2, 1],), ([0, 0, 0],), ([-1, -5, -3],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3]) == 5
assert circuit.encrypt_run_decrypt([7, 7, 7]) == 7
assert circuit.encrypt_run_decrypt([0, -2, 2]) == 2
assert circuit.encrypt_run_decrypt([9, 2, 1]) == 9
assert circuit.encrypt_run_decrypt([0, 0, 0]) == 0
assert circuit.encrypt_run_decrypt([-1, -5, -3]) == -1"""
        ),
        (
            "array_min",
            "Finds minimum value. Tests min at different positions, duplicates, zeroes and negatives.",
            """def test_array_min(arr):
    return array_min(arr)

compiler = fhe.Compiler(test_array_min, {"arr": "encrypted"})
inputset = [([1, 5, 3],), ([7, 7, 7],), ([0, -2, 2],), ([3, 2, -5],), ([0, 0, 0],), ([-1, -5, -3],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3]) == 1
assert circuit.encrypt_run_decrypt([7, 7, 7]) == 7
assert circuit.encrypt_run_decrypt([0, -2, 2]) == -2
assert circuit.encrypt_run_decrypt([3, 2, -5]) == -5
assert circuit.encrypt_run_decrypt([0, 0, 0]) == 0
assert circuit.encrypt_run_decrypt([-1, -5, -3]) == -5"""
        ),
        (
            "array_range",
            "Calculates the range (max - min). Tests zero range (all elements equal), positive ranges, zeroes, and negatives.",
            """def test_array_range(arr):
    return array_range(arr)

compiler = fhe.Compiler(test_array_range, {"arr": "encrypted"})
inputset = [([1, 5, 3],), ([7, 7, 7],), ([-5, 0, 5],), ([0, 0, 0],), ([-10, -5, -2],)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3]) == 4
assert circuit.encrypt_run_decrypt([7, 7, 7]) == 0
assert circuit.encrypt_run_decrypt([-5, 0, 5]) == 10
assert circuit.encrypt_run_decrypt([0, 0, 0]) == 0
assert circuit.encrypt_run_decrypt([-10, -5, -2]) == 8"""
        ),
        (
            "array_count_greater",
            "Counts elements strictly greater than threshold. Tests all greater, none greater, some greater, zeroes and negatives.",
            """def test_array_count_greater(arr, thresh):
    return array_count_greater(arr, thresh)

compiler = fhe.Compiler(test_array_count_greater, {"arr": "encrypted", "thresh": "encrypted"})
inputset = [([1, 5, 3], 2), ([1, 5, 3], 5), ([1, 5, 3], 0), ([2, 2, 2], 2), ([0, 0, 0], 0), ([-1, -5, -3], -4)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3], 2) == 2
assert circuit.encrypt_run_decrypt([1, 5, 3], 5) == 0
assert circuit.encrypt_run_decrypt([1, 5, 3], 0) == 3
assert circuit.encrypt_run_decrypt([2, 2, 2], 2) == 0
assert circuit.encrypt_run_decrypt([0, 0, 0], 0) == 0
assert circuit.encrypt_run_decrypt([-1, -5, -3], -4) == 2"""
        ),
        (
            "array_median",
            "Computes median. Tests sorted, unsorted, arrays with duplicates, zeroes, and negatives.",
            """def test_array_median(arr, min_val, max_val):
    return array_median(arr, min_val, max_val)

compiler = fhe.Compiler(test_array_median, {"arr": "encrypted", "min_val": "clear", "max_val": "clear"})
inputset = [([1, 5, 3, 4], -15, 15), ([0, 0, 2, 2], -15, 15), ([4, 3, 2, 1], -15, 15), ([0, 0, 0, 0], -15, 15), ([-1, -5, -3, -4], -15, 15)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3, 4], -15, 15) == 3
assert circuit.encrypt_run_decrypt([0, 0, 2, 2], -15, 15) == 1
assert circuit.encrypt_run_decrypt([4, 3, 2, 1], -15, 15) == 2
assert circuit.encrypt_run_decrypt([0, 0, 0, 0], -15, 15) == 0
assert circuit.encrypt_run_decrypt([-1, -5, -3, -4], -15, 15) == -3"""
        ),
        (
            "array_percentile",
            "Calculates q-th percentile. Tests 0th (min), 50th (median), 100th (max) percentiles, zeroes and negatives.",
            """def test_array_percentile(arr, q, min_val, max_val):
    return array_percentile(arr, q, min_val, max_val)

compiler = fhe.Compiler(test_array_percentile, {"arr": "encrypted", "q": "clear", "min_val": "clear", "max_val": "clear"})
inputset = [([1, 5, 3, 4], 50, -15, 15), ([1, 5, 3, 4], 100, -15, 15), ([1, 5, 3, 4], 0, -15, 15), ([-1, -5, -3, -4], 50, -15, 15), ([0, 0, 0, 0], 50, -15, 15)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 5, 3, 4], 50, -15, 15) == 4
assert circuit.encrypt_run_decrypt([1, 5, 3, 4], 100, -15, 15) == 5
assert circuit.encrypt_run_decrypt([1, 5, 3, 4], 0, -15, 15) == 1
assert circuit.encrypt_run_decrypt([-1, -5, -3, -4], 50, -15, 15) == -3
assert circuit.encrypt_run_decrypt([0, 0, 0, 0], 50, -15, 15) == 0"""
        ),
        (
            "array_histogram",
            "Counts occurrences in range. Tests elements within range, out of range, zeroes and negatives.",
            """def test_array_histogram(arr, min_val, max_val):
    return array_histogram(arr, min_val, max_val)

compiler = fhe.Compiler(test_array_histogram, {"arr": "encrypted", "min_val": "clear", "max_val": "clear"})
inputset = [([1, 2, 1], 1, 3), ([2, 2, 2], 1, 3), ([0, 4, 5], 1, 3), ([-2, -1, 0], -2, 0)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([1, 2, 1], 1, 3), [2, 1, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([2, 2, 2], 1, 3), [0, 3, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 4, 5], 1, 3), [0, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-2, -1, 0], -2, 0), [1, 1, 1])"""
        ),
        (
            "array_mode",
            "Returns most frequent value. Tests single clear mode, tie cases (returns first match), zeroes and negatives.",
            """def test_array_mode(arr, min_val, max_val):
    return array_mode(arr, min_val, max_val)

compiler = fhe.Compiler(test_array_mode, {"arr": "encrypted", "min_val": "clear", "max_val": "clear"})
inputset = [([1, 2, 2, 3], 1, 5), ([1, 1, 3, 3], 1, 5), ([4, 4, 4, 4], 1, 5), ([-2, -2, -1, -1], -3, 0), ([0, 0, 0, 0], -1, 1)]
circuit = compiler.compile(inputset)

assert circuit.encrypt_run_decrypt([1, 2, 2, 3], 1, 5) == 2
assert circuit.encrypt_run_decrypt([1, 1, 3, 3], 1, 5) == 1
assert circuit.encrypt_run_decrypt([4, 4, 4, 4], 1, 5) == 4
assert circuit.encrypt_run_decrypt([-2, -2, -1, -1], -3, 0) == -2
assert circuit.encrypt_run_decrypt([0, 0, 0, 0], -1, 1) == 0"""
        ),
        (
            "array_normalize",
            "Normalizes array elements. Tests centering arrays around specific means, zeroes, and negatives.",
            """def test_array_normalize(arr, mean, scale):
    return array_normalize(arr, mean, scale)

compiler = fhe.Compiler(test_array_normalize, {"arr": "encrypted", "mean": "encrypted", "scale": "clear"})
inputset = [([2, 4, 6], 4, 10), ([0, 0, 0], 0, 5), ([10, 20, 30], 20, 1), ([-2, -4, -6], -4, 10)]
circuit = compiler.compile(inputset)

np.testing.assert_array_equal(circuit.encrypt_run_decrypt([2, 4, 6], 4, 10), [-20, 0, 20])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([0, 0, 0], 0, 5), [0, 0, 0])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([10, 20, 30], 20, 1), [-10, 0, 10])
np.testing.assert_array_equal(circuit.encrypt_run_decrypt([-2, -4, -6], -4, 10), [20, 0, -20])"""
        )
    ]

if __name__ == "__main__":
    generate_notebook("docs/tutorials/01_arithmetic.ipynb", "concrete_fhe_toolkit.arithmetic", get_arithmetic_tests())
    generate_notebook("docs/tutorials/02_arrays.ipynb", "concrete_fhe_toolkit.arrays", get_arrays_tests())
    generate_notebook("docs/tutorials/03_stats.ipynb", "concrete_fhe_toolkit.stats", get_stats_tests())
