"""Batch interface and tensor-shape regression tests."""

import numpy as np
import pytest
from concrete import fhe

from concrete_fhe_toolkit.ml import FHECNN, FHELinearRegression, FHEMLP, FHEModel, FHEPCA
from concrete_fhe_toolkit.ml import classes


class ImageSum(FHEModel):
    def _circuit_logic(self, features):
        return sum(sum(row) for row in features)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "model,samples,sample,expected",
    [
        (FHEPCA([0, 0], [[1, 0], [0, 1]]), [[0, 0], [3, 3]], [1, 2], [1, 2]),
        (FHEMLP([([[1, 0], [0, 1]], [0, 0])]), [[0, 0], [3, 3]], [1, 2], [1, 2]),
        (
            FHECNN([[[1]]], [0]),
            [[[[0, 0], [0, 0]]], [[[3, 3], [3, 3]]]],
            [[[1, 2], [2, 3]]],
            [1, 2, 2, 3],
        ),
    ],
)
def test_vector_output_models_compile(model, samples, sample, expected, batched):
    kwargs = {"batch_size": 2, "inputset_is_batched": False} if batched else {}
    model.compile(samples, configuration=fhe.Configuration(p_error=2**-40), **kwargs)
    np.testing.assert_array_equal(model.simulate(sample), expected)
    np.testing.assert_array_equal(model.simulate_many([sample]), [expected])


def test_default_single_sample_compiles():
    model = FHELinearRegression([2], 1)
    model.compile([[0], [4]], configuration=fhe.Configuration(p_error=2**-40))
    assert int(model.simulate([3])) == 7
    assert [int(v) for v in model.simulate_many([[1], [4]])] == [3, 9]
    # Deployment uses the same single-sample input shape.
    assert int(model.circuit.simulate(np.array([3]))) == 7


@pytest.mark.parametrize("prebatched", [False, True, None])
def test_batch_compilation_and_partial_prediction(prebatched):
    model = FHELinearRegression([2], 1)
    kwargs = {} if prebatched is None else {"inputset_is_batched": prebatched}
    inputset = [[1], [4]] if prebatched is False else [[[1], [1]], [[4], [4]]]
    model.compile(inputset, batch_size=2, configuration=fhe.Configuration(p_error=2**-40), **kwargs)
    assert [int(v) for v in model.simulate_many([[1], [3], [4]])] == [3, 7, 9]
    assert model.simulate_many([]) == []


def test_tensor_padding_repeats_an_in_domain_sample(monkeypatch):
    calls = []
    calibration = []

    class Compiler:
        def __init__(self, function, encryption):
            self.function = function

        def compile(self, inputset):
            calibration.extend(inputset)
            return self

        def simulate(self, value):
            calls.append(value)
            return np.array([np.sum(sample) for sample in value])

        encrypt_run_decrypt = simulate

    monkeypatch.setattr(classes.fhe, "Compiler", Compiler)
    model = ImageSum()
    first = [[2, 3], [4, 5]]
    second = [[3, 4], [5, 6]]
    model.compile([first, second], batch_size=2, inputset_is_batched=False)
    assert model.predict(first) == 14
    assert calls[0].shape == (2, 2, 2)
    np.testing.assert_array_equal(calls[0][0], calls[0][1])
    for batch, sample in zip(calibration, [first, second]):
        np.testing.assert_array_equal(batch, [sample, sample])


def test_infer_prebatched_size():
    model = FHELinearRegression([1], 0)
    model.compile([[[1], [1]], [[3], [3]]], inputset_is_batched=True)
    assert model.batch_size == 2
    assert int(model.simulate([2])) == 2


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_batch_size(size):
    with pytest.raises((TypeError, ValueError)):
        FHELinearRegression([1], 0).compile([[1]], batch_size=size)


def test_input_shapes_and_uncompiled_calls():
    model = FHELinearRegression([1], 0)
    for method in [model.predict_many, model.simulate_many]:
        with pytest.raises(ValueError, match="compiled"):
            method([])
    for inputset in [[], [[1], [1, 2]], [[1.5]]]:
        with pytest.raises((TypeError, ValueError)):
            model.compile(inputset)
    with pytest.raises(ValueError, match="batch dimension"):
        model.compile([[[1], [2]]], batch_size=3)
    model.compile([[0], [3]])
    with pytest.raises(ValueError, match="shape"):
        model.simulate([1, 2])
