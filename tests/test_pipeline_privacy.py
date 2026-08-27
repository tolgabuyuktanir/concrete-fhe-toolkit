"""Tests for FHEPipeline, preprocessing transformers, DP release, batching."""

import numpy as np
import pytest
from concrete import fhe

from concrete_fhe_toolkit import ml, privacy
from concrete_fhe_toolkit.ml.pipeline import FHEPipeline
from concrete_fhe_toolkit.ml.preprocessing import (
    FHEBinner,
    FHEMinMaxScaler,
    FHEStandardScaler,
    bin_feature,
)


def test_bin_feature_clear():
    edges = [18, 30, 45, 65]
    assert int(bin_feature(10, edges)) == 0
    assert int(bin_feature(18, edges)) == 1
    assert int(bin_feature(35, edges)) == 2
    assert int(bin_feature(70, edges)) == 4
    with pytest.raises(ValueError):
        bin_feature(1, [])


def test_transformers_clear():
    binner = FHEBinner([[18, 30, 45, 65], [1000, 5000]])
    assert [int(v) for v in binner._transform_logic([35, 7000])] == [2, 2]

    scaler = FHEStandardScaler(means=[50, 100], stds=[10, 20], scale=10)
    assert [int(v) for v in scaler._transform_logic([65, 60])] == [15, -20]

    minmax = FHEMinMaxScaler([0, 10], [200, 20], scale=100)
    assert [int(v) for v in minmax._transform_logic([50, 15])] == [25, 50]

    with pytest.raises(ValueError):
        FHEStandardScaler(means=[1], stds=[1, 2])
    with pytest.raises(ValueError):
        FHEMinMaxScaler([5], [5])


def test_pipeline_clear_and_simulated():
    # Ages in years, incomes in thousands to keep circuit bit widths small.
    pipeline = FHEPipeline(
        [
            FHEBinner([[18, 30, 45, 65], [1, 5, 20]]),
            ml.FHELogisticRegression(weights=[3, 2], bias=-7),
        ]
    )

    # Clear path: binned [2, 2] -> score 3*2 + 2*2 - 7 = 3 -> class 1.
    assert int(pipeline._circuit_logic([35, 7])) == 1
    assert int(pipeline._circuit_logic([20, 0])) == 0  # bins [1, 0] -> -2

    pipeline.compile(
        [
            np.array([16, 0], dtype=np.int64),
            np.array([70, 30], dtype=np.int64),
            np.array([40, 4], dtype=np.int64),
        ]
    )
    assert int(pipeline.simulate(np.array([35, 7], dtype=np.int64))) == 1
    assert int(pipeline.simulate(np.array([20, 0], dtype=np.int64))) == 0

    results = pipeline.simulate_many(
        [
            np.array([35, 7], dtype=np.int64),
            np.array([20, 0], dtype=np.int64),
        ]
    )
    assert [int(v) for v in results] == [1, 0]


def test_pipeline_validation():
    with pytest.raises(ValueError):
        FHEPipeline([])
    with pytest.raises(ValueError):
        FHEPipeline([object()])
    with pytest.raises(ValueError):
        # A transformer cannot be the final step.
        FHEPipeline([FHEBinner([[1]]), FHEBinner([[1]])])


def test_predict_many_uses_one_circuit():
    model = ml.FHELogisticRegression(weights=[1], bias=0)

    class _StubCircuit:
        def __init__(self):
            self.calls = 0

        def encrypt_run_decrypt(self, features):
            self.calls += 1
            return features[0] >= 0

    stub = _StubCircuit()
    model.circuit = stub
    assert model.predict_many([[1], [-2], [3]]) == [True, False, True]
    assert stub.calls == 3

    unfitted = ml.FHELogisticRegression(weights=[1], bias=0)
    with pytest.raises(ValueError):
        unfitted.predict([1])
    with pytest.raises(ValueError):
        unfitted.simulate([1])


def test_dp_mechanisms_are_calibrated_and_reproducible():
    rng = np.random.default_rng(42)
    first = privacy.laplace_mechanism(842, sensitivity=1, epsilon=1.0, rng=rng)
    assert isinstance(first, int)
    again = privacy.laplace_mechanism(
        842, sensitivity=1, epsilon=1.0, rng=np.random.default_rng(42)
    )
    assert first == again  # seeded noise is reproducible

    # High epsilon => tiny noise; low epsilon => visibly larger spread.
    tight = [
        privacy.laplace_mechanism(100, sensitivity=1, epsilon=100.0,
                                  rng=np.random.default_rng(seed))
        for seed in range(50)
    ]
    loose = [
        privacy.laplace_mechanism(100, sensitivity=1, epsilon=0.05,
                                  rng=np.random.default_rng(seed))
        for seed in range(50)
    ]
    assert max(abs(value - 100) for value in tight) <= 1
    assert max(abs(value - 100) for value in loose) > 5

    noisy = privacy.gaussian_mechanism(
        842, sensitivity=1, epsilon=0.5, delta=1e-5, rng=np.random.default_rng(7)
    )
    assert isinstance(noisy, int)

    with pytest.raises(ValueError):
        privacy.laplace_mechanism(1, sensitivity=1, epsilon=0)
    with pytest.raises(ValueError):
        privacy.gaussian_mechanism(1, sensitivity=1, epsilon=2.0, delta=1e-5)


def test_dp_release_splits_budget():
    counts = [412, 430]
    noisy = privacy.dp_release(
        counts, sensitivity=1, epsilon=1.0, rng=np.random.default_rng(0)
    )
    assert len(noisy) == 2 and all(isinstance(value, int) for value in noisy)
    assert privacy.dp_release([], sensitivity=1, epsilon=1.0) == []

    with pytest.raises(ValueError):
        privacy.dp_release([1], sensitivity=1, epsilon=1.0, mechanism="gaussian")
    with pytest.raises(ValueError):
        privacy.dp_release([1], sensitivity=1, epsilon=1.0, mechanism="uniform")


def test_trainer_aggregates_compose_with_dp():
    # End-to-end: encrypted k-means aggregates -> DP-noised centroid update.
    X_train = [[0, 1], [1, 0], [2, 2], [1, 1], [8, 8], [9, 7], [7, 9], [8, 7]]
    trainer = ml.FHEKMeansTrainer(
        initial_centroids=[[2, 2], [7, 7]],
        min_value=0,
        max_value=9,
        n_iterations=2,
        simulate=True,
        configuration=fhe.Configuration(p_error=2**-40),
    )
    model = trainer.fit_encrypted(X_train)
    noised = privacy.dp_release(
        [value for centroid in model.centroids for value in centroid],
        sensitivity=1,
        epsilon=50.0,
        rng=np.random.default_rng(3),
    )
    assert len(noised) == 4


def test_model_serialization_round_trip(tmp_path):
    from concrete_fhe_toolkit.ml.serialization import load_model, save_model

    tree = {"feature": 0, "threshold": 5, "left": 1, "right": 0}
    originals = [
        ml.FHELogisticRegression(weights=[3, 2], bias=-7),
        ml.FHEDecisionTree(tree),
        ml.FHERandomForest([tree, tree, tree]),
        ml.FHEKNN([[0, 0], [5, 5]], [0, 1], 1),
        ml.FHEMLP([([[1, 0], [0, 1]], [0, 0]), ([[1, 1]], [1])]),
        ml.FHEKMeans([[1, 1], [8, 8]], max_distance=200),
    ]

    for index, model in enumerate(originals):
        path = tmp_path / f"model_{index}.json"
        save_model(model, str(path))
        restored = load_model(str(path))
        assert type(restored) is type(model)

    # Behavior survives the round trip (clear evaluation of the tree).
    path = tmp_path / "tree.json"
    save_model(ml.FHEDecisionTree(tree), str(path))
    restored = load_model(str(path))
    assert int(ml.decision_tree_inference([7, 0], restored.tree)) == 1
    assert int(ml.decision_tree_inference([2, 0], restored.tree)) == 0

    # Trainer extras (output_scale) survive too.
    trained = ml.FHELinearRegression([200, 300], 100)
    trained.output_scale = 100
    path = tmp_path / "linreg.json"
    save_model(trained, str(path))
    assert load_model(str(path)).output_scale == 100

    with pytest.raises(ValueError):
        save_model(object(), str(tmp_path / "bad.json"))
