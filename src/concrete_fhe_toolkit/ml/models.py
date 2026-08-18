"""Encrypted inference helpers for simple ML models."""

from typing import Any, List, Optional

from .._compat import fhe

from concrete_fhe_toolkit._utils import compile_function, validate_bounds, validate_integer
from concrete_fhe_toolkit.arrays import array_sum, make_argmin
from concrete_fhe_toolkit.math import equal, greater, greater_equal

from .core import euclidean_distance_squared
from .matrix import dot_product


def linear_regression_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    """Evaluate a linear regression model (dot product of weights and features plus bias)."""
    product = dot_product(weights, features)
    return bias + product


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
    max_distance: int = 15,
) -> Any:
    """Evaluate a 1-nearest-neighbor model on an encrypted test sample.

    ``max_distance`` must be an upper bound on the squared Euclidean distance
    between the test sample and any training sample. The argmin reduction is
    built for the ``[0, max_distance]`` range, so distances above the bound
    silently corrupt the result.
    """
    maximum_distance = validate_integer("max_distance", max_distance, minimum=1)
    distances = [
        euclidean_distance_squared(row, test_sample)
        for row in train_samples
    ]

    argmin_func = make_argmin(len(distances), 0, maximum_distance)
    min_index = argmin_func(distances)

    prediction = 0
    for index, label in enumerate(train_labels):
        prediction += equal(index, min_index) * label

    return prediction


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
