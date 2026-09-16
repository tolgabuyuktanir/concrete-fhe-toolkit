from .._compat import fhe
from typing import List,Any
import numpy as np

def one_hot_encode(label: Any, num_classes: int) -> List[Any]:
    classes = np.arange(num_classes)
    return classes == label


def binarize(array: List[Any], threshold: Any) -> List[Any]:
    tensor = fhe.array(array)
    return tensor >= threshold  


def clip_array(array: List[Any], min_val: Any, max_val: Any) -> List[Any]:
    tensor = fhe.array(array)
    return np.clip(tensor, min_val, max_val)


def normalize_array(array: List[Any], divisor: int) -> List[Any]:
    tensor = fhe.array(array)
    return tensor // divisor