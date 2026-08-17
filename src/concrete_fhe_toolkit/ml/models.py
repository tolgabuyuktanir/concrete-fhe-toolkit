from typing import Any, List,Optional
from concrete import fhe
from concrete_fhe_toolkit.ml import dot_product, euclidean_distance_squared
from concrete_fhe_toolkit.math import bit_select, greater_equal, equal, greater
from concrete_fhe_toolkit._utils import validate_bounds, compile_function
from concrete_fhe_toolkit.arrays import array_sum, make_argmin

def linear_regression_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    """Evaluate a linear regression model (dot product of weights and features plus bias)."""
    product = dot_product(weights,features)
    return bias + product

def decision_tree_node(feature_val: Any, threshold: Any, left_branch: Any, right_branch: Any) -> Any:
    """Evaluate a single decision tree node, selecting left or right branch based on threshold."""
    control = greater_equal(feature_val,threshold)
    return bit_select(control,left_branch,right_branch)

def majority_votes(predictions: List[Any]) -> Any:
    """Perform majority voting for an ensemble of binary predictions."""
    sum_predictions = array_sum(predictions)
    return greater(sum_predictions, len(predictions) // 2)

def knn_inference(test_sample: List[Any], train_samples: List[List[Any]], train_labels: List[Any]) -> Any:
    """Evaluate a 1-Nearest Neighbor (KNN) model on an encrypted test sample."""
    distances = []
    for row in train_samples:
        distances.append(euclidean_distance_squared(row,test_sample))

    argmin_func = make_argmin(len(distances))
    min_index = argmin_func(distances)

    sum = 0
    for i in range(len(train_labels)):
        sum += equal(i,min_index) * train_labels[i]

    return sum    

def compile_decision_tree_node(
    min_value: int = -15,
    max_value: int = 15,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted decision tree node"""
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        decision_tree_node,
        {
            "feature_val": "encrypted",
            "threshold": "encrypted",
            "left_branch": "encrypted",
            "right_branch": "encrypted"
        },
        [
            (minimum, maximum, minimum, maximum),
            (maximum, minimum, maximum, minimum),
        ],
        configuration,
    )
    