"""Encrypted inference helpers for simple ML models."""

from typing import Any, List, Optional

from .._compat import fhe

from concrete_fhe_toolkit._utils import compile_function, validate_bounds, validate_integer
from concrete_fhe_toolkit.arrays import array_sum, make_argmin
from concrete_fhe_toolkit.math import equal, greater, greater_equal

from .activations import threshold_activation
from .core import euclidean_distance_squared
from .matrix import dot_product


def linear_regression_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    """Evaluate a linear regression model (dot product of weights and features plus bias)."""
    product = dot_product(weights, features)
    return bias + product


def logistic_regression_inference(
    weights: List[Any],
    bias: Any,
    features: List[Any],
    *,
    threshold: int = 0,
) -> Any:
    """Evaluate a logistic regression classifier on encrypted features.

    Computes the linear score ``weights . features + bias`` and returns the
    binary class ``score >= threshold``. Because the sigmoid is monotonic,
    thresholding the raw score at 0 is equivalent to thresholding the
    probability at 0.5, so no sigmoid lookup is needed for classification.
    Scale weights, bias, and features to integers with the same factor
    before calling.
    """
    score = linear_regression_inference(weights, bias, features)
    return threshold_activation(score, threshold)


def decision_tree_node(feature_val: Any, threshold: Any, left_branch: Any, right_branch: Any) -> Any:
    """Evaluate a single decision tree node.

    Returns ``left_branch`` when ``feature_val >= threshold``, otherwise
    ``right_branch``. Branch values may be arbitrary bounded integers.
    """
    control = greater_equal(feature_val, threshold)
    return control * left_branch + (1 - control) * right_branch


def majority_votes(predictions: List[Any]) -> Any:
    """Perform majority voting for an ensemble of binary predictions."""
    sum_predictions = array_sum(predictions)
    return greater(sum_predictions, len(predictions) // 2)


def knn_inference(
    test_sample: List[Any],
    train_samples: List[List[Any]],
    train_labels: List[Any],
    *,
    k: int = 1,
    max_distance: int = 15,
) -> Any:
    """Evaluate a k-nearest-neighbor model on an encrypted test sample.

    ``max_distance`` must be an upper bound on the squared Euclidean distance
    between the test sample and any training sample. The argmin reductions
    are built for that range, so distances above the bound silently corrupt
    the result.

    With ``k=1`` the nearest label is returned directly and labels may be
    arbitrary bounded integers. With ``k > 1`` the labels must be binary
    (0/1) and the majority vote of the k nearest labels is returned; use an
    odd ``k`` to avoid ties (a tie resolves to 0).

    The k > 1 path runs k argmin rounds, masking each selected neighbor with
    a distance penalty, so circuit cost grows linearly with ``k``.
    """
    maximum_distance = validate_integer("max_distance", max_distance, minimum=1)
    normalized_k = validate_integer("k", k, minimum=1)
    if len(train_samples) != len(train_labels):
        raise ValueError("train_samples and train_labels must have the same length")
    if normalized_k > len(train_samples):
        raise ValueError("k cannot exceed the number of training samples")

    distances = [
        euclidean_distance_squared(row, test_sample)
        for row in train_samples
    ]

    if normalized_k == 1:
        argmin_func = make_argmin(len(distances), 0, maximum_distance)
        min_index = argmin_func(distances)

        prediction = 0
        for index, label in enumerate(train_labels):
            prediction += equal(index, min_index) * label

        return prediction

    # k > 1: iterated argmin. Each round the selected neighbor's distance is
    # pushed above max_distance so it cannot be selected again.
    penalty = maximum_distance + 1
    argmin_func = make_argmin(len(distances), 0, maximum_distance + penalty)
    current_distances = list(distances)
    votes = []

    for _ in range(normalized_k):
        min_index = argmin_func(current_distances)

        vote = 0
        next_distances = []
        for index, (distance, label) in enumerate(zip(current_distances, train_labels)):
            selected = equal(index, min_index)
            vote += selected * label
            next_distances.append(distance + selected * penalty)

        current_distances = next_distances
        votes.append(vote)

    return majority_votes(votes)


def decision_tree_inference(features: List[Any], tree: Any) -> Any:
    """Evaluate a full decision tree on encrypted features.

    ``tree`` is a public structure. An internal node is a dict with keys
    ``"feature"`` (index into ``features``), ``"threshold"`` (public
    integer), ``"left"``, and ``"right"`` (subtrees); a leaf is a plain
    integer label or value. The left branch is taken when
    ``features[feature] >= threshold``.

    Every path of the tree is evaluated obliviously, so the visited path is
    never revealed — circuit cost grows with the total number of nodes.
    """
    if not isinstance(tree, dict):
        return validate_integer("leaf value", tree)

    missing = {"feature", "threshold", "left", "right"} - tree.keys()
    if missing:
        raise ValueError(
            f"tree node is missing keys: {', '.join(sorted(missing))}"
        )

    feature_index = validate_integer("feature", tree["feature"], minimum=0)
    if feature_index >= len(features):
        raise ValueError("tree feature index is out of range for the feature vector")
    threshold = validate_integer("threshold", tree["threshold"])

    control = greater_equal(features[feature_index], threshold)
    left_value = decision_tree_inference(features, tree["left"])
    right_value = decision_tree_inference(features, tree["right"])
    return control * left_value + (1 - control) * right_value


def compile_decision_tree_node(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a single encrypted decision tree node."""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        decision_tree_node,
        {
            "feature_val": "encrypted",
            "threshold": "encrypted",
            "left_branch": "encrypted",
            "right_branch": "encrypted",
        },
        [
            (minimum, minimum, minimum, minimum),
            (minimum, maximum, minimum, maximum),
            (maximum, minimum, maximum, minimum),
            (maximum, maximum, maximum, maximum),
        ],
        configuration,
    )
