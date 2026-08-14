"""Machine Learning specific operations for Concrete FHE."""

from .activations import compile_leaky_relu, compile_relu, compile_unit_step, leaky_relu, relu, sigmoid, sign, tanh, unit_step
from .core import accuracy_score, compile_hinge_loss, euclidean_distance_squarred, hinge_loss, manhattan_distance, mean_absolute_error, mean_squared_error
from .matrix import dot_product, matrix_add, matrix_transpose
from .models import decision_tree_node, linear_regression_inference
from .stats import array_max, array_mean, array_min, array_variance
from .utils import one_hot_encode

__all__ = [
    "array_max",
    "array_mean",
    "array_min",
    "array_variance",
    "accuracy_score",
    "compile_hinge_loss",
    "compile_leaky_relu",
    "compile_relu",
    "compile_unit_step",
    "dot_product",
    "euclidean_distance_squarred",
    "hinge_loss",
    "leaky_relu",
    "linear_regression_inference",
    "decision_tree_node",
    "manhattan_distance",
    "matrix_add",
    "matrix_transpose",
    "mean_absolute_error",
    "mean_squared_error",
    "relu",
    "sigmoid",
    "sign",
    "tanh",
    "unit_step",
    "one_hot_encode",
]
