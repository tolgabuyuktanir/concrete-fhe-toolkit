import numpy as np

from concrete_fhe_toolkit.arrays import (
    compile_array_pad,
    compile_array_count,
    compile_array_contains,
    compile_array_index_of,
    compile_array_index,
    compile_array_set,
)


def test_array_pad_compiles_and_simulates():
    circuit = compile_array_pad(min_value=-5, max_value=5, target_length=5, pad_value=0)
    sample = np.array([1, 2], dtype=np.int64)
    expected = np.array([1, 2, 0, 0, 0], dtype=np.int64)
    
    assert np.array_equal(circuit.simulate(sample), expected)


def test_array_count_compiles_and_simulates():
    circuit = compile_array_count(5, -5, 5, target=-2)
    sample = np.array([1, -2, 3, -2, -2], dtype=np.int64)
    
    assert int(circuit.simulate(sample)) == 3


def test_array_contains_compiles_and_simulates():
    circuit = compile_array_contains(5, -5, 5, target=3)
    sample = np.array([1, -2, 3, 0, 0], dtype=np.int64)
    
    assert int(circuit.simulate(sample)) == 1


def test_array_index_of_compiles_and_simulates():
    circuit = compile_array_index_of(5, -5, 5, target=3, missing_index=-1)
    sample = np.array([1, -2, 3, 0, 0], dtype=np.int64)
    
    assert int(circuit.simulate(sample)) == 2


def test_array_index_compiles_and_simulates():
    # array_index requires size and bounds of the array elements
    circuit = compile_array_index(5, -50, 50)
    sample = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    
    assert int(circuit.simulate(sample, 2)) == 30


def test_array_set_compiles_and_simulates():
    # array_set requires size and bounds of the array elements
    circuit = compile_array_set(5, -50, 99)
    sample = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    expected = np.array([10, 20, 99, 40, 50], dtype=np.int64)
    
    assert np.array_equal(circuit.simulate(sample, 2, 99), expected)
