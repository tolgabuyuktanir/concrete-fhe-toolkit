from .._compat import fhe
from typing import List, Any, Union
import numpy as np

def one_hot_encode(label: Any, num_classes: int) -> Union[np.ndarray, List[Any]]:
    """Convert an encrypted class label into a one-hot encoded boolean array.

    Args:
        label (Any): The encrypted class label.
        num_classes (int): The number of classes.

    Returns:
        Union[np.ndarray, List[Any]]: A one-hot encoded boolean array.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.utils import one_hot_encode
        
        # In FHE circuit: one_hot_encode(2, num_classes=4) -> [0, 0, 1, 0]
        ```
    """
    classes = np.arange(num_classes)
    return classes == label


def binarize(array: Union[np.ndarray, List[Any]], threshold: Any) -> Union[np.ndarray, List[Any]]:
    """Binarize an encrypted array based on a threshold.

    Args:
        array (Union[np.ndarray, List[Any]]): The encrypted array to binarize.
        threshold (Any): The threshold to compare against.

    Returns:
        Union[np.ndarray, List[Any]]: The binarized boolean array.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.utils import binarize
        
        # In FHE circuit: binarize([1, 5, 10], 5) -> [0, 1, 1]
        ```
    """
    tensor = fhe.array(array)
    return tensor >= threshold  


def clip_array(array: Union[np.ndarray, List[Any]], min_val: Any, max_val: Any) -> Union[np.ndarray, List[Any]]:
    """Clip values in an encrypted array to a minimum and maximum range.

    Args:
        array (Union[np.ndarray, List[Any]]): The encrypted array to clip.
        min_val (Any): The minimum value to clip to.
        max_val (Any): The maximum value to clip to.

    Returns:
        Union[np.ndarray, List[Any]]: The clipped encrypted array.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.utils import clip_array
        
        # In FHE circuit: clip_array([-5, 10, 25], 0, 15) -> [0, 10, 15]
        ```
    """
    tensor = fhe.array(array)
    return np.clip(tensor, min_val, max_val)


def normalize_array(array: Union[np.ndarray, List[Any]], divisor: int) -> Union[np.ndarray, List[Any]]:
    """Normalize an encrypted array by floor dividing by a constant scalar divisor.

    Args:
        array (Union[np.ndarray, List[Any]]): The encrypted array to normalize.
        divisor (int): The integer value to divide elements by.

    Returns:
        Union[np.ndarray, List[Any]]: The normalized encrypted array.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.utils import normalize_array
        
        # In FHE circuit: normalize_array([10, 25, 50], 10) -> [1, 2, 5]
        ```
    """
    tensor = fhe.array(array)
    return tensor // divisor