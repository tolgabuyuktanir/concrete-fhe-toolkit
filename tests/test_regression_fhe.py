"""Encrypted round trips for corrected arithmetic and model contracts."""

import os
from itertools import product

import numpy as np
import pytest
from concrete import fhe

from concrete_fhe_toolkit import finance, ml
from concrete_fhe_toolkit import math as fm
from concrete_fhe_toolkit.ml.matrix import covariance_matrix

pytestmark = [
    pytest.mark.fhe,
    pytest.mark.skipif(
        os.environ.get("RUN_FHE_TESTS") != "1",
        reason="set RUN_FHE_TESTS=1 to run encrypted regression tests",
    ),
]


def test_encrypted_wide_denominator():
    config = fhe.Configuration(p_error=2**-40)
    divide = fm.compile_unsigned_floor_divide(2, 4, configuration=config)
    modulo = fm.compile_unsigned_mod(2, 4, configuration=config)
    assert int(divide.encrypt_run_decrypt(3, 8)) == 0
    assert int(modulo.encrypt_run_decrypt(3, 8)) == 3


def test_encrypted_negative_transfer():
    circuit = fhe.Compiler(
        finance.transfer,
        {"sender_balance": "encrypted", "receiver_balance": "encrypted", "amount": "encrypted"},
    ).compile(
        list(product([0, 3], [0, 3], [-3, 0, 3, 4])),
        configuration=fhe.Configuration(p_error=2**-40),
    )
    assert tuple(circuit.encrypt_run_decrypt(3, 2, -1)) == (3, 2)
    assert tuple(circuit.encrypt_run_decrypt(3, 2, 2)) == (1, 4)


def test_encrypted_tensor_batch_and_regression():
    class TensorSum(ml.FHEModel):
        def _circuit_logic(self, features):
            return np.sum(features)

    model = TensorSum()
    model.compile(
        [np.ones((2, 2), dtype=np.int64), np.full((2, 2), 3)],
        batch_size=2,
        inputset_is_batched=False,
        configuration=fhe.Configuration(p_error=2**-40),
    )
    assert [int(v) for v in model.predict_many([[[1, 2], [2, 3]]])] == [8]

    tree = {"feature": 0, "threshold": 2, "left": 5, "right": -3}
    regressor = ml.FHEXGBoostRegressor([tree, tree])
    regressor.compile([[0], [3]], configuration=fhe.Configuration(p_error=2**-40))
    assert int(regressor.predict([0])) == -6
    assert int(regressor.predict([3])) == 10


def test_encrypted_covariance():
    def covariance(data):
        return fhe.array(covariance_matrix(data))

    samples = [np.array(v).reshape(2, 2) for v in product(range(-1, 2), repeat=4)]
    circuit = fhe.Compiler(covariance, {"data": "encrypted"}).compile(
        samples, configuration=fhe.Configuration(p_error=2**-40)
    )
    np.testing.assert_array_equal(
        circuit.encrypt_run_decrypt(np.array([[0, 0], [1, -1]])), [[0, -1], [-1, 0]]
    )
