import numpy as np

try:
    assert np.array([True])
    print("np.array([True]) PASSES assert")
except Exception as e:
    print(f"np.array([True]) FAILS assert: {type(e).__name__}: {e}")

try:
    assert np.array(4) == 4
    print("np.array(4) == 4 PASSES assert")
except Exception as e:
    print(f"np.array(4) == 4 FAILS assert: {type(e).__name__}: {e}")

try:
    assert np.array([4]) == 4
    print("np.array([4]) == 4 PASSES assert")
except Exception as e:
    print(f"np.array([4]) == 4 FAILS assert: {type(e).__name__}: {e}")
