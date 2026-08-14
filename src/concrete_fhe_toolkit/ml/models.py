from typing import Any, List
from concrete_fhe_toolkit.ml import dot_product
from concrete_fhe_toolkit.math import bit_select,greater_equal

def linear_regression_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    product = dot_product(weights,features)
    return bias + product

def decision_tree_node(feature_val: Any, threshold: Any, left_branch: Any, right_branch: Any) -> Any:
    control = greater_equal(feature_val,threshold)
    return bit_select(control,left_branch,right_branch)