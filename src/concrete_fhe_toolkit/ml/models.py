from typing import Any, List,Optional
from concrete import fhe
from concrete_fhe_toolkit.ml import dot_product
from concrete_fhe_toolkit.math import bit_select,greater_equal
from concrete_fhe_toolkit._utils import validate_bounds,compile_function

def linear_regression_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    product = dot_product(weights,features)
    return bias + product

def decision_tree_node(feature_val: Any, threshold: Any, left_branch: Any, right_branch: Any) -> Any:
    control = greater_equal(feature_val,threshold)
    return bit_select(control,left_branch,right_branch)


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
    