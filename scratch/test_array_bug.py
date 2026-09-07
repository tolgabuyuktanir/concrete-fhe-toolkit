import numpy as np
from concrete import fhe

def test(x, y):
    return fhe.array([x, y])

compiler = fhe.Compiler(test, {"x": "encrypted", "y": "encrypted"})
inputset = [(0, -5), (1, 1), (5, 5)]

try:
    circuit = compiler.compile(inputset)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
