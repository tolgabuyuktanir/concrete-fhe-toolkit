from .._compat import fhe
from typing import List, Any, Union
import numpy as np

def one_hot_encode(label: Any, num_classes: int) -> Union[np.ndarray, List[Any]]:
    classes = np.arange(num_classes)
    return classes == label


def binarize(array: Union[np.ndarray, List[Any]], threshold: Any) -> Union[np.ndarray, List[Any]]:
    tensor = fhe.array(array)
    return tensor >= threshold  


def clip_array(array: Union[np.ndarray, List[Any]], min_val: Any, max_val: Any) -> Union[np.ndarray, List[Any]]:
    tensor = fhe.array(array)
    return np.clip(tensor, min_val, max_val)


def normalize_array(array: Union[np.ndarray, List[Any]], divisor: int) -> Union[np.ndarray, List[Any]]:
    tensor = fhe.array(array)
    return tensor // divisor