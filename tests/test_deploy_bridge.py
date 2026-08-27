"""Tests for deployment helpers and the sklearn bridge."""

import os

import numpy as np
import pytest

from concrete_fhe_toolkit import deploy, ml

sklearn = pytest.importorskip("sklearn")
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402


def _grid(low, high):
    return [[a, b] for a in range(low, high) for b in range(low, high)]


def test_from_sklearn_linear_logistic_matches():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 10, size=(80, 2))
    y = (X[:, 0] + 2 * X[:, 1] >= 12).astype(int)
    clf = LogisticRegression().fit(X, y)

    fhe_model = ml.from_sklearn_linear(clf, scale=100)
    assert isinstance(fhe_model, ml.FHELogisticRegression)

    agree = sum(
        int(fhe_model._circuit_logic(list(sample))) == int(clf.predict([sample])[0])
        for sample in _grid(0, 10)
    )
    assert agree >= 97  # only boundary rounding may differ


def test_from_sklearn_linear_regression():
    X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 5], [0, 1]])
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1
    reg = LinearRegression().fit(X, y)

    fhe_model = ml.from_sklearn_linear(reg, scale=100)
    assert isinstance(fhe_model, ml.FHELinearRegression)
    assert fhe_model.weights == [200, 300]
    assert fhe_model.bias == 100
    assert fhe_model.output_scale == 100

    with pytest.raises(ValueError):
        ml.from_sklearn_linear(object())


def test_from_sklearn_tree_matches_exactly():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 10, size=(120, 2))
    y = ((X[:, 0] >= 5) & (X[:, 1] >= 3)).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)

    fhe_tree = ml.from_sklearn_tree(clf)
    for sample in _grid(0, 10):
        expected = int(clf.predict([sample])[0])
        assert int(ml.decision_tree_inference(sample, fhe_tree.tree)) == expected


def test_from_sklearn_forest_matches_majority():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 10, size=(150, 2))
    y = (X[:, 0] + X[:, 1] >= 10).astype(int)
    clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0).fit(X, y)

    fhe_forest = ml.from_sklearn_forest(clf)
    assert len(fhe_forest.trees) == 5
    agree = sum(
        int(ml.random_forest_inference(list(sample), fhe_forest.trees))
        == int(clf.predict([sample])[0])
        for sample in _grid(0, 10)
    )
    assert agree >= 95  # forest majority may differ from sklearn's proba vote on ties


def test_deployment_artifacts_save_and_load(tmp_path):
    model = ml.FHELogisticRegression(weights=[2, 1], bias=-4)
    model.compile([np.array([0, 0], dtype=np.int64), np.array([5, 5], dtype=np.int64)])

    directory = str(tmp_path / "deployment")
    deploy.save_deployment(model.circuit, directory)
    assert os.path.exists(os.path.join(directory, deploy.SERVER_FILENAME))
    assert os.path.exists(os.path.join(directory, deploy.CLIENT_FILENAME))

    from concrete import fhe

    server = deploy.load_server(directory)
    client = deploy.load_client(directory)
    assert isinstance(server, fhe.Server)
    assert isinstance(client, fhe.Client)

    with pytest.raises(ValueError):
        deploy.save_deployment(None, directory)
    with pytest.raises(FileNotFoundError):
        deploy.load_server(str(tmp_path / "missing"))


@pytest.mark.fhe
@pytest.mark.skipif(
    os.environ.get("RUN_FHE_TESTS") != "1",
    reason="set RUN_FHE_TESTS=1 to run the real client/server round trip",
)
def test_deployment_round_trip_real(tmp_path):
    model = ml.FHELogisticRegression(weights=[2, 1], bias=-4)
    model.compile([np.array([0, 0], dtype=np.int64), np.array([5, 5], dtype=np.int64)])

    directory = str(tmp_path / "deployment")
    deploy.save_deployment(model.circuit, directory)

    server = deploy.load_server(directory)
    client = deploy.load_client(directory)
    client.keys.generate()

    args = client.encrypt(np.array([4, 1], dtype=np.int64))
    result = server.run(args, evaluation_keys=client.evaluation_keys)
    assert int(client.decrypt(result)) == 1
