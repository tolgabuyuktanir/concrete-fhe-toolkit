import numpy as np

from concrete_fhe_toolkit.math.special import (
    compile_exp,
    compile_expm1,
    compile_log10,
    compile_log1p,
    compile_erfc,
    compile_sinh,
    compile_cosh,
    compile_asin,
    compile_acos,
    compile_cbrt,
)

# Note: We test compilation and simulation independently to ensure each 
# special mathematical FHE lookup circuit compiles correctly.

def test_compile_exp_simulates():
    circuit = compile_exp(-3, 3)
    assert int(circuit.simulate(1)) == int(np.exp(1))

def test_compile_expm1_simulates():
    circuit = compile_expm1(-3, 3)
    assert int(circuit.simulate(1)) == int(np.expm1(1))

def test_compile_log10_simulates():
    circuit = compile_log10(1, 10)
    assert int(circuit.simulate(10)) == int(np.log10(10))

def test_compile_log1p_simulates():
    circuit = compile_log1p(0, 10)
    assert int(circuit.simulate(9)) == int(np.log1p(9))

def test_compile_erfc_simulates():
    circuit = compile_erfc(-3, 3)
    from scipy.special import erfc
    assert int(circuit.simulate(1)) == int(erfc(1))

def test_compile_sinh_simulates():
    circuit = compile_sinh(-3, 3)
    assert int(circuit.simulate(2)) == int(np.sinh(2))

def test_compile_cosh_simulates():
    circuit = compile_cosh(-3, 3)
    assert int(circuit.simulate(2)) == int(np.cosh(2))

def test_compile_asin_simulates():
    # asin expects inputs in [-1, 1], we map bounds appropriately 
    # (assuming the toolkit handles internal scaling or integers properly, testing compilation limits)
    circuit = compile_asin(-1, 1)
    assert int(circuit.simulate(0)) == int(np.arcsin(0))

def test_compile_acos_simulates():
    circuit = compile_acos(-1, 1)
    assert int(circuit.simulate(0)) == int(np.arccos(0))

def test_compile_cbrt_simulates():
    circuit = compile_cbrt(-27, 27)
    assert int(circuit.simulate(8)) == 2
