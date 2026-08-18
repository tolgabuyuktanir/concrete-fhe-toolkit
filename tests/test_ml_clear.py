"""Cleartext behavior tests for the ml subpackage."""

import numpy as np
import pytest

from concrete_fhe_toolkit import ml


def test_distances_clear():
    assert int(ml.manhattan_distance([1, 2, 3], [4, 0, 3])) == 5
    assert int(ml.hamming_distance([1, 2, 3], [1, 0, 3])) == 1
    assert int(ml.euclidean_distance_squared([1, 2], [4, 6])) == 25

    with pytest.raises(ValueError):
        ml.manhattan_distance([1, 2], [1])
    with pytest.raises(ValueError):
        ml.hamming_distance([1, 2], [1])
    with pytest.raises(ValueError):
        ml.euclidean_distance_squared([1, 2], [1])


def test_regression_metrics_clear():
    assert int(ml.mean_squared_error([1, 2, 3], [1, 4, 6])) == (0 + 4 + 9) // 3
    assert int(ml.mean_absolute_error([1, 2, 3], [1, 4, 6])) == (0 + 2 + 3) // 3


def test_classification_metrics_clear():
    y_preds = [1, 1, 0, 0]
    y_trues = [1, 0, 1, 0]

    assert int(ml.accuracy_score(y_preds, y_trues)) == 50
    assert int(ml.true_positives(y_preds, y_trues)) == 1
    assert int(ml.true_negatives(y_preds, y_trues)) == 1
    assert int(ml.false_positives(y_preds, y_trues)) == 1
    assert int(ml.false_negatives(y_preds, y_trues)) == 1

    matrix = ml.confusion_matrix(y_preds, y_trues)
    assert [[int(cell) for cell in row] for row in matrix] == [[1, 1], [1, 1]]


def test_hinge_loss_clear():
    for y_pred in range(-3, 4):
        for y_true in (-1, 1):
            expected = max(0, 1 - y_true * y_pred)
            assert int(ml.hinge_loss(y_pred, y_true)) == expected


def test_matrix_operations_clear():
    matrix1 = [[1, 2], [3, 4]]
    matrix2 = [[5, 6], [7, 8]]

    assert int(ml.dot_product([1, 2, 3], [4, 5, 6])) == 32
    assert ml.matrix_transpose(matrix1) == [[1, 3], [2, 4]]
    assert ml.matrix_add(matrix1, matrix2) == [[6, 8], [10, 12]]
    assert ml.matrix_subtract(matrix2, matrix1) == [[4, 4], [4, 4]]

    product = ml.matrix_multiply(matrix1, matrix2)
    expected = np.array(matrix1) @ np.array(matrix2)
    assert [[int(cell) for cell in row] for row in product] == expected.tolist()

    vector = ml.matrix_vector_multiply(matrix1, [1, 1])
    assert [int(item) for item in vector] == [3, 7]

    with pytest.raises(ValueError):
        ml.dot_product([1, 2], [1])
    with pytest.raises(ValueError):
        ml.matrix_multiply([[1, 2, 3]], [[1, 2], [3, 4]])


def test_activations_clear():
    for value in range(-30, 31):
        assert int(ml.relu(value)) == max(0, value)
        assert int(ml.unit_step(value)) == int(value >= 0)
        assert int(ml.threshold_activation(value, 5)) == int(value >= 5)

        # leaky_relu truncates the scaled negative branch toward zero.
        expected_leaky = value if value >= 0 else -((-value) // 10)
        assert int(ml.leaky_relu(value, alpha=0.1)) == expected_leaky

    with pytest.raises(ValueError):
        ml.leaky_relu(1, alpha=0.3)
    with pytest.raises(ValueError):
        ml.leaky_relu(1, alpha=0)


def test_models_clear():
    assert int(ml.linear_regression_inference([2, 3], 1, [4, 5])) == 24

    assert int(ml.decision_tree_node(5, 3, 10, 20)) == 10
    assert int(ml.decision_tree_node(2, 3, 10, 20)) == 20
    assert int(ml.decision_tree_node(2, 3, -7, 9)) == 9

    assert int(ml.majority_votes([1, 1, 0])) == 1
    assert int(ml.majority_votes([1, 0, 0])) == 0


def test_knn_inference_clear():
    train_samples = [[0, 0], [5, 5], [9, 1]]
    train_labels = [7, 9, 4]

    assert int(ml.knn_inference([1, 1], train_samples, train_labels, max_distance=200)) == 7
    assert int(ml.knn_inference([5, 4], train_samples, train_labels, max_distance=200)) == 9
    assert int(ml.knn_inference([8, 0], train_samples, train_labels, max_distance=200)) == 4


def test_stats_clear():
    values = [1, 2, 3, 4]

    assert int(ml.array_mean(values)) == 2
    assert int(ml.array_variance(values)) == 1
    assert int(ml.array_min([3, -2, 7, -2, 4])) == -2
    assert int(ml.array_max([3, -2, 7, -2, 4])) == 7
    assert int(ml.array_range([3, -2, 7, -2, 4])) == 9
    assert int(ml.array_count_greater([1, 5, 3, 8], 2)) == 3


def test_utils_clear():
    one_hot = ml.one_hot_encode(2, 4)
    assert [int(bit) for bit in one_hot] == [0, 0, 1, 0]

    binarized = ml.binarize([1, 5, 3], 3)
    assert [int(bit) for bit in binarized] == [0, 1, 1]

    clipped = ml.clip_array([-5, 0, 9], -2, 4)
    assert [int(item) for item in clipped] == [-2, 0, 4]


def test_logistic_regression_inference_clear():
    weights = [2, -1]
    bias = -3

    # score = 2*x - y - 3; class = score >= 0
    assert int(ml.logistic_regression_inference(weights, bias, [3, 1])) == 1
    assert int(ml.logistic_regression_inference(weights, bias, [1, 1])) == 0
    assert int(ml.logistic_regression_inference(weights, bias, [2, 1], threshold=1)) == 0
    assert int(ml.logistic_regression_inference(weights, bias, [3, 0], threshold=1)) == 1


def test_knn_inference_k_greater_than_one_clear():
    train_samples = [[0, 0], [1, 1], [5, 5], [6, 6], [9, 9]]
    train_labels = [0, 0, 1, 1, 1]

    # Nearest to [1, 0]: [0,0] and [1,1] (labels 0), then [5,5] (label 1).
    assert int(ml.knn_inference([1, 0], train_samples, train_labels, k=3, max_distance=200)) == 0
    # Nearest to [6, 5]: [5,5], [6,6], then [9,9] (all label 1).
    assert int(ml.knn_inference([6, 5], train_samples, train_labels, k=3, max_distance=200)) == 1
    # k=5 uses every sample: labels sum to 3 of 5 -> majority 1.
    assert int(ml.knn_inference([1, 0], train_samples, train_labels, k=5, max_distance=200)) == 1

    with pytest.raises(ValueError):
        ml.knn_inference([1, 0], train_samples, train_labels, k=6, max_distance=200)
    with pytest.raises(ValueError):
        ml.knn_inference([1, 0], train_samples, [0, 1], max_distance=200)


def test_decision_tree_inference_clear():
    tree = {
        "feature": 0,
        "threshold": 5,
        "left": {"feature": 1, "threshold": 3, "left": 30, "right": 20},
        "right": 10,
    }

    def reference(x, y):
        if x >= 5:
            return 30 if y >= 3 else 20
        return 10

    for x in range(0, 10):
        for y in range(0, 6):
            assert int(ml.decision_tree_inference([x, y], tree)) == reference(x, y)

    with pytest.raises(ValueError):
        ml.decision_tree_inference([1, 2], {"feature": 0, "threshold": 5, "left": 1})
    with pytest.raises(ValueError):
        ml.decision_tree_inference([1, 2], {"feature": 7, "threshold": 5, "left": 1, "right": 0})
