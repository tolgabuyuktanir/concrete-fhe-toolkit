from typing import Any, List, Optional
from .._compat import fhe
from concrete_fhe_toolkit._utils import compile_function, validate_bounds
from concrete_fhe_toolkit.arithmetic import make_floor_divide

from concrete_fhe_toolkit.arrays import array_sum
from concrete_fhe_toolkit.math import square, maximum, equal,not_equal


def manhattan_distance(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the Manhattan (L1) distance between two encrypted arrays."""
    if(len(array1) != len(array2)):
        raise ValueError("The array sizes must be equal")
    diffs = [abs(x-y) for x,y in zip(array1,array2)]
    return array_sum(diffs)

def hamming_distance(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the Hamming distance (number of mismatches) between two encrypted arrays."""
    if(len(array1) != len(array2)):
        raise ValueError("The array sizes must be equal")
    distance = 0
    for item1,item2 in zip(array1,array2):
        distance += not_equal(item1,item2)

    return distance

def euclidean_distance_squared(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the squared Euclidean (L2) distance between two encrypted arrays."""
    if(len(array1) != len(array2)):
        raise ValueError("The array sizes must be equal")
    diffs = [square(x-y) for x,y in zip(array1,array2)]
    return array_sum(diffs)

def mean_squared_error(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the Mean Squared Error (MSE) between two encrypted arrays."""
    if(len(array1) != len(array2)):
            raise ValueError("The array sizes must be equal")
    return euclidean_distance_squared(array1,array2) // len(array1)

def mean_absolute_error(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Calculate the Mean Absolute Error (MAE) between predictions and true values."""
    distance = manhattan_distance(y_preds,y_trues)
    return distance // len(y_trues)

def accuracy_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Calculate the percentage accuracy score between predictions and true labels."""
    if(len(y_preds) != len(y_trues)):
        raise ValueError("The array sizes must be equal")
    true_predictions = 0

    for pred,true in zip(y_preds,y_trues):
        true_predictions += equal(pred,true)

    return true_predictions * 100 // len(y_trues)

def true_positives(y_preds: List[Any], y_trues: List[Any]) -> int:
    """Calculate the number of True Positives in binary classification."""
    if(len(y_preds) != len(y_trues)):
        raise ValueError("The array sizes must be equal")
    num_of_true_positives = 0

    for pred,true in zip(y_preds,y_trues):
        num_of_true_positives += equal(1,pred*true)

    return num_of_true_positives

def true_negatives(y_preds: List[Any], y_trues: List[Any]) -> int:
    """Calculate the number of True Negatives in binary classification."""
    if(len(y_preds) != len(y_trues)):
        raise ValueError("The array sizes must be equal")
    num_of_true_negatives = 0
    for pred,true in zip(y_preds,y_trues):
        num_of_true_negatives += equal(0,pred+true)

    return num_of_true_negatives

def false_negatives(y_preds: List[Any], y_trues: List[Any]) -> int:
    """Calculate the number of False Negatives in binary classification."""
    num_of_positives = array_sum(y_trues)
    return num_of_positives - true_positives(y_preds, y_trues)

def false_positives(y_preds: List[Any], y_trues: List[Any]) -> int:
    """Calculate the number of False Positives in binary classification."""
    num_of_negatives = len(y_trues) - array_sum(y_trues)
    return num_of_negatives - true_negatives(y_preds, y_trues)


def confusion_matrix(y_preds: List[Any], y_trues: List[Any]) -> List[List[Any]]:
    """Generate a confusion matrix [[TN, FP], [FN, TP]] for binary classification."""
    tp = true_positives(y_preds,y_trues)
    tn = true_negatives(y_preds,y_trues)
    fn = false_negatives(y_preds,y_trues)
    fp = false_positives(y_preds,y_trues)
    return [[tn,fp],[fn,tp]]


def hinge_loss(y_pred: Any, y_true: Any) -> Any:
    """Calculate the encrypted Hinge Loss for a single prediction and true value."""
    return maximum(0, 1-(y_true*y_pred))


def compile_hinge_loss(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted hinge loss."""
    minimum_val, maximum_val = validate_bounds(min_value, max_value)
    return compile_function(
        hinge_loss,
        {"y_pred": "encrypted", "y_true": "encrypted"},
        [
            (minimum_val, minimum_val),
            (minimum_val, maximum_val),
            (maximum_val, minimum_val),
            (maximum_val, maximum_val),
        ],
        configuration,
    )

def precision_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Integer percent precision: TP * 100 // (TP + FP), 0 with no positive predictions.

    The encrypted-by-encrypted division uses a multivariate lookup whose
    cost grows with the count range, so keep sample counts small when this
    runs under encryption (it is cheap on decrypted predictions).
    """
    divide = make_floor_divide(zero_result=0)
    tp = true_positives(y_preds, y_trues)
    fp = false_positives(y_preds, y_trues)
    return divide(tp * 100, tp + fp)


def recall_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Integer percent recall: TP * 100 // (TP + FN), 0 with no positive labels."""
    divide = make_floor_divide(zero_result=0)
    tp = true_positives(y_preds, y_trues)
    fn = false_negatives(y_preds, y_trues)
    return divide(tp * 100, tp + fn)


def f1_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Integer percent F1: 2 * P * R // (P + R), 0 when both are 0."""
    divide = make_floor_divide(zero_result=0)
    precision = precision_score(y_preds, y_trues)
    recall = recall_score(y_preds, y_trues)
    return divide(2 * precision * recall, precision + recall)
