from typing import Any, List
from concrete_fhe_toolkit.ml import dot_product

def linear_regression_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    product = dot_product(weights,features)
    return bias + product
    