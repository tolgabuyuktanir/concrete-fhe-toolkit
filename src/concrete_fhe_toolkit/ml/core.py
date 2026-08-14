from typing import Any, List, Optional
from concrete import fhe
from concrete_fhe_toolkit._utils import compile_function, validate_bounds

from concrete_fhe_toolkit.arrays import array_sum
from concrete_fhe_toolkit.math import absolute, square, maximum,equal


def manhattan_distance(array1: List[Any], array2: List[Any]) -> Any:
    list1 = list(array1)
    list2 = list(array2)
    if(len(list1) != len(list2)):
        raise ValueError("The array sizes must be equal")

    diffs = [absolute(x-y) for x,y in zip(list1,list2)]
    return array_sum(diffs)

def euclidean_distance_squarred(array1: List[Any], array2: List[Any]) -> Any:
    list1 = list(array1)
    list2 = list(array2)
    if(len(list1) != len(list2)):
        raise ValueError("The array sizes must be equal")

    diffs = [square(x-y) for x,y in zip(list1,list2)]
    return array_sum(diffs)

def mean_squared_error(array1: List[Any], array2: List[Any]) -> Any:
    list1 = list(array1)
    list2 = list(array2)
    if(len(list1) != len(list2)):
            raise ValueError("The array sizes must be equal")
    
    return euclidean_distance_squarred(list1,list2) // len(list1)

def mean_absolute_error(y_preds: List[Any], y_trues: List[Any]) -> Any:
    distance = manhattan_distance(y_preds,y_trues)
    return distance // len(y_trues)

def accuracy_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    true_predictions = 0
    num_of_elements = len(y_trues)

    for pred,true in zip(y_preds,y_trues):
        true_predictions += equal(pred,true)

    return true_predictions * 100 // num_of_elements        

def hinge_loss(y_pred: Any, y_true: Any) -> Any:
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