"""Machine Learning specific operations for Concrete FHE."""

# sign, sigmoid, and tanh are re-exported from their real homes for
# backwards compatibility; sigmoid/tanh are BoundedOperation compile
# factories from the math subpackage, not traceable activations.
from ..arithmetic import sign
from ..math import sigmoid, tanh
from .activations import (
    compile_leaky_relu,
    compile_relu,
    compile_threshold_activation,
    compile_unit_step,
    leaky_relu,
    relu,
    threshold_activation,
    unit_step,
)
from .core import accuracy_score, compile_hinge_loss, confusion_matrix, euclidean_distance_squared, hamming_distance, hinge_loss, manhattan_distance, mean_absolute_error, mean_squared_error, true_positives, true_negatives, false_positives, false_negatives
from .matrix import dot_product, matrix_add, matrix_transpose, matrix_subtract, matrix_multiply, matrix_vector_multiply
from .models import (
    compile_decision_tree_node,
    decision_tree_inference,
    decision_tree_node,
    knn_inference,
    linear_regression_inference,
    logistic_regression_inference,
    majority_votes,
)
from .stats import array_max, array_mean, array_min, array_range, array_variance, array_count_greater
from .utils import binarize, one_hot_encode, clip_array

__all__ = [
    "array_count_greater",
    "array_max",
    "array_mean",
    "array_min",
    "array_range",
    "array_variance",
    "accuracy_score",
    "compile_decision_tree_node",
    "compile_hinge_loss",
    "compile_leaky_relu",
    "compile_relu",
    "compile_threshold_activation",
    "compile_unit_step",
    "confusion_matrix",
    "dot_product",
    "euclidean_distance_squared",
    "false_negatives",
    "false_positives",
    "hamming_distance",
    "hinge_loss",
    "knn_inference",
    "leaky_relu",
    "linear_regression_inference",
    "logistic_regression_inference",
    "majority_votes",
    "decision_tree_inference",
    "decision_tree_node",
    "manhattan_distance",
    "matrix_add",
    "matrix_multiply",
    "matrix_subtract",
    "matrix_transpose",
    "matrix_vector_multiply",
    "mean_absolute_error",
    "mean_squared_error",
    "relu",
    "sigmoid",
    "sign",
    "tanh",
    "threshold_activation",
    "true_negatives",
    "true_positives",
    "unit_step",
    "binarize",
    "clip_array",
    "one_hot_encode",
]
