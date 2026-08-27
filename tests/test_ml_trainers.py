"""Tests for encrypted trainers and the task namespaces (simulation mode)."""

import pytest
from concrete import fhe

from concrete_fhe_toolkit import ml
from concrete_fhe_toolkit.ml import classification, clustering, regression

TIGHT = fhe.Configuration(p_error=2**-40)


def test_linear_regression_trainer_recovers_weights():
    X_train = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5], [0, 1]]
    y_train = [2 * a + 3 * b + 1 for a, b in X_train]

    trainer = regression.FHELinearRegressionTrainer(
        weight_scale=100, simulate=True, configuration=TIGHT
    )
    model = trainer.fit_encrypted(X_train, y_train)

    assert model.weights == [200, 300]
    assert model.bias == 100
    assert model.output_scale == 100

    with pytest.raises(ValueError):
        trainer.fit_encrypted([[1, 1], [2, 2], [3, 3]], [1, 2, 3])  # singular


def test_decision_tree_trainer_learns_split():
    X_train = [[1, 1], [2, 4], [6, 1], [7, 4], [4, 2], [9, 3], [3, 3], [8, 0]]
    y_train = [0, 0, 1, 1, 0, 1, 0, 1]  # class = x0 >= 5

    trainer = classification.FHEDecisionTreeTrainer(
        candidate_thresholds=[[3, 5, 7], [2, 4]],
        max_depth=2,
        num_classes=2,
        simulate=True,
        configuration=TIGHT,
    )
    model = trainer.fit_encrypted(X_train, y_train)

    tree = model.tree
    assert tree["feature"] == 0 and tree["threshold"] == 5

    for features, label in zip(X_train, y_train):
        assert int(ml.decision_tree_inference(features, tree)) == label


def test_kmeans_trainer_finds_clusters():
    X_train = [[0, 1], [1, 0], [2, 2], [1, 1], [8, 8], [9, 7], [7, 9], [8, 7]]

    trainer = clustering.FHEKMeansTrainer(
        initial_centroids=[[2, 2], [7, 7]],
        min_value=0,
        max_value=9,
        n_iterations=3,
        simulate=True,
        configuration=TIGHT,
    )
    model = trainer.fit_encrypted(X_train)

    assert model.centroids == [[1, 1], [8, 8]]

    assert int(ml.nearest_centroid_inference([1, 2], model.centroids, max_distance=model.max_distance)) == 0
    assert int(ml.nearest_centroid_inference([9, 8], model.centroids, max_distance=model.max_distance)) == 1

    trained_inertia = int(
        clustering.inertia(X_train, model.centroids, max_distance=model.max_distance)
    )
    initial_inertia = int(
        clustering.inertia(X_train, [[2, 2], [7, 7]], max_distance=model.max_distance)
    )
    assert trained_inertia <= initial_inertia


def test_r2_score_clear():
    assert int(ml.r2_score([1, 2, 3, 4], [1, 2, 3, 4])) == 100
    assert int(ml.r2_score([2, 3, 4, 5], [1, 2, 3, 4])) == 34
    assert int(ml.r2_score([3, 3], [3, 3])) == 0  # constant targets edge case


def test_task_namespaces():
    assert classification.FHEDecisionTreeClassifier is ml.FHEDecisionTree
    assert classification.FHERandomForestClassifier is ml.FHERandomForest
    assert regression.FHEDecisionTreeRegressor is ml.FHEDecisionTree
    assert regression.FHELinearRegression is ml.FHELinearRegression
    assert clustering.FHEKMeans is ml.FHEKMeans
    assert issubclass(ml.FHELinearRegressionTrainer, ml.FHETrainer)
    assert issubclass(ml.FHEDecisionTreeTrainer, ml.FHETrainer)
    assert issubclass(ml.FHEKMeansTrainer, ml.FHETrainer)
