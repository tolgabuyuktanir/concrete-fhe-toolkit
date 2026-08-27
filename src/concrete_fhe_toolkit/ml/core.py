from typing import Any, List, Optional, Callable
from .._compat import fhe
from concrete_fhe_toolkit._utils import compile_function, validate_bounds
from concrete_fhe_toolkit.arithmetic import make_floor_divide

from concrete_fhe_toolkit.arrays import array_sum
from concrete_fhe_toolkit.math import square, maximum, equal,not_equal
from concrete_fhe_toolkit.math.special import make_log


def manhattan_distance(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the Manhattan (L1) distance between two encrypted arrays.
    
    This is often used as a robust distance metric for clustering or
    k-nearest neighbor algorithms under encryption, since it avoids 
    the expensive squaring operations required by Euclidean distance.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import manhattan_distance
        
        # Inside an FHE circuit
        # dist = manhattan_distance([enc_x1, enc_y1], [enc_x2, enc_y2])
        ```
    """
    if(len(array1) != len(array2)):
        raise ValueError("The array sizes must be equal")
    diffs = [abs(x-y) for x,y in zip(array1,array2)]
    return array_sum(diffs)

def hamming_distance(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the Hamming distance (number of mismatches) between two encrypted arrays.
    
    This is useful for comparing binary feature vectors or categorical data, 
    measuring the number of positions where the two arrays differ.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import hamming_distance
        
        # Inside an FHE circuit
        # dist = hamming_distance([enc_a1, enc_b1], [enc_a2, enc_b2])
        ```
    """
    if(len(array1) != len(array2)):
        raise ValueError("The array sizes must be equal")
    distance = 0
    for item1,item2 in zip(array1,array2):
        distance += not_equal(item1,item2)

    return distance

def euclidean_distance_squared(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the squared Euclidean (L2) distance between two encrypted arrays.
    
    Often used in k-means clustering or RBF kernels. Using the squared 
    distance avoids the need for an expensive square root operation.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import euclidean_distance_squared
        
        # Inside an FHE circuit
        # dist_sq = euclidean_distance_squared(enc_point_1, enc_point_2)
        ```
    """
    if(len(array1) != len(array2)):
        raise ValueError("The array sizes must be equal")
    diffs = [square(x-y) for x,y in zip(array1,array2)]
    return array_sum(diffs)

def mean_squared_error(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the Mean Squared Error (MSE) between two encrypted arrays.
    
    A standard metric for regression tasks. It measures the average 
    of the squares of the errors (the difference between estimated values 
    and the actual value).
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import mean_squared_error
        
        # Inside an FHE circuit
        # mse = mean_squared_error(enc_predictions, enc_true_values)
        ```
    """
    if(len(array1) != len(array2)):
            raise ValueError("The array sizes must be equal")
    return euclidean_distance_squared(array1,array2) // len(array1)

def mean_absolute_error(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Calculate the Mean Absolute Error (MAE) between predictions and true values.
    
    A robust alternative to MSE that is less sensitive to outliers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import mean_absolute_error
        
        # Inside an FHE circuit
        # mae = mean_absolute_error(enc_predictions, enc_true_values)
        ```
    """
    distance = manhattan_distance(y_preds,y_trues)
    return distance // len(y_trues)

def accuracy_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Calculate the percentage accuracy score between predictions and true labels.
    
    Returns an integer percentage [0, 100]. Useful for evaluating 
    classification models directly on encrypted data.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import accuracy_score
        
        # Inside an FHE circuit
        # acc_pct = accuracy_score(enc_predictions, enc_labels)
        ```
    """
    if(len(y_preds) != len(y_trues)):
        raise ValueError("The array sizes must be equal")
    true_predictions = 0

    for pred,true in zip(y_preds,y_trues):
        true_predictions += equal(pred,true)

    return true_predictions * 100 // len(y_trues)

def true_positives(y_preds: List[Any], y_trues: List[Any]) -> int:
    """Calculate the number of True Positives in binary classification.
    
    Assuming 1 is positive and 0 is negative, this counts instances 
    where both the prediction and the true label are 1.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import true_positives
        
        # Inside an FHE circuit
        # tp = true_positives(enc_binary_preds, enc_binary_labels)
        ```
    """
    if(len(y_preds) != len(y_trues)):
        raise ValueError("The array sizes must be equal")
    num_of_true_positives = 0

    for pred,true in zip(y_preds,y_trues):
        num_of_true_positives += equal(1,pred*true)

    return num_of_true_positives

def true_negatives(y_preds: List[Any], y_trues: List[Any]) -> int:
    """Calculate the number of True Negatives in binary classification.
    
    Assuming 1 is positive and 0 is negative, this counts instances 
    where both the prediction and the true label are 0.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import true_negatives
        
        # Inside an FHE circuit
        # tn = true_negatives(enc_binary_preds, enc_binary_labels)
        ```
    """
    if(len(y_preds) != len(y_trues)):
        raise ValueError("The array sizes must be equal")
    num_of_true_negatives = 0
    for pred,true in zip(y_preds,y_trues):
        num_of_true_negatives += equal(0,pred+true)

    return num_of_true_negatives

def false_negatives(y_preds: List[Any], y_trues: List[Any]) -> int:
    """Calculate the number of False Negatives in binary classification.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import false_negatives
        
        # Inside an FHE circuit
        # fn = false_negatives(enc_binary_preds, enc_binary_labels)
        ```
    """
    num_of_positives = array_sum(y_trues)
    return num_of_positives - true_positives(y_preds, y_trues)

def false_positives(y_preds: List[Any], y_trues: List[Any]) -> int:
    """Calculate the number of False Positives in binary classification.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import false_positives
        
        # Inside an FHE circuit
        # fp = false_positives(enc_binary_preds, enc_binary_labels)
        ```
    """
    num_of_negatives = len(y_trues) - array_sum(y_trues)
    return num_of_negatives - true_negatives(y_preds, y_trues)


def confusion_matrix(y_preds: List[Any], y_trues: List[Any]) -> List[List[Any]]:
    """Generate a confusion matrix [[TN, FP], [FN, TP]] for binary classification.
    
    This is extremely useful for calculating various ML metrics (like precision, 
    recall, F1) securely over encrypted predictions.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import confusion_matrix
        
        # Inside an FHE circuit
        # matrix = confusion_matrix(enc_binary_preds, enc_binary_labels)
        # tn, fp = matrix[0]
        # fn, tp = matrix[1]
        ```
    """
    tp = true_positives(y_preds,y_trues)
    tn = true_negatives(y_preds,y_trues)
    fn = false_negatives(y_preds,y_trues)
    fp = false_positives(y_preds,y_trues)
    return [[tn,fp],[fn,tp]]


def hinge_loss(y_pred: Any, y_true: Any) -> Any:
    """Calculate the encrypted Hinge Loss for a single prediction and true value.
    
    Hinge loss is commonly used for "maximum-margin" classification, most 
    notably for Support Vector Machines (SVMs). The true label should be +1 or -1.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import hinge_loss
        
        # Inside an FHE circuit
        # loss = hinge_loss(enc_pred_score, enc_true_label)
        ```
    """
    return maximum(0, 1-(y_true*y_pred))

def l1_norm(array: List[Any]) -> Any:
    total = 0
    for item in array:
        total += abs(item)
    return total

def compile_hinge_loss(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted hinge loss.
    
    Compiles the `hinge_loss` function into an FHE circuit so that loss 
    can be evaluated completely obliviously.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import compile_hinge_loss
        
        circuit = compile_hinge_loss(min_value=-10, max_value=10)
        # Calculates max(0, 1 - (y_true * y_pred)) under encryption
        ```
    """
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

def make_cross_entropy_loss(
    min_input: int = -127,
    max_input: int = 128,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
) -> Callable[[List[Any],List[Any]], Any]:
    """Create a scaled Binary Cross-Entropy loss function.
    
    This factory creates a cross-entropy function using a pre-configured 
    logarithm table (TLU). The probabilities should be scaled integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import make_cross_entropy_loss
        
        cross_entropy = make_cross_entropy_loss(min_input=1, max_input=100)
        # Inside an FHE circuit
        # loss = cross_entropy(enc_preds, enc_labels)
        ```
    """
    log_func = make_log(min_input, max_input, input_scale=input_scale, output_scale=output_scale, invalid_result=-999)

    def cross_entropy_loss(y_preds: List[Any], y_trues: List[Any]) -> Any:
        losses = []
        for pred, true in zip(y_preds, y_trues):
            pred_log = log_func(pred)
            minus_pred_log = log_func(input_scale - pred)
            loss = -(true * pred_log + (1 - true) * minus_pred_log)
            losses.append(loss)

        return array_sum(losses) // len(y_trues)
        
    return cross_entropy_loss 

def compile_cross_entropy_loss(
    array_size: int,
    min_value: int = -127,
    max_value: int = 127,
    *,
    input_scale: int = 100,
    output_scale: int = 100,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    function = make_cross_entropy_loss(min_value, max_value, input_scale=input_scale, output_scale=output_scale)
    minimum, maximum = validate_bounds(min_value, max_value)
    
    return compile_function(
        function,
        {"y_preds": "encrypted", "y_trues": "encrypted"},
        [
            ([minimum] * array_size, [0] * array_size), # preds ve trues
            ([maximum] * array_size, [1] * array_size),
        ],
        configuration,
    )


def precision_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Integer percent precision: TP * 100 // (TP + FP), 0 with no positive predictions.

    The encrypted-by-encrypted division uses a multivariate lookup whose
    cost grows with the count range, so keep sample counts small when this
    runs under encryption (it is cheap on decrypted predictions).
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import precision_score
        
        # Inside an FHE circuit
        # precision_pct = precision_score(enc_binary_preds, enc_binary_labels)
        ```
    """
    divide = make_floor_divide(zero_result=0)
    tp = true_positives(y_preds, y_trues)
    fp = false_positives(y_preds, y_trues)
    return divide(tp * 100, tp + fp)


def recall_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Integer percent recall: TP * 100 // (TP + FN), 0 with no positive labels.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import recall_score
        
        # Inside an FHE circuit
        # recall_pct = recall_score(enc_binary_preds, enc_binary_labels)
        ```
    """
    divide = make_floor_divide(zero_result=0)
    tp = true_positives(y_preds, y_trues)
    fn = false_negatives(y_preds, y_trues)
    return divide(tp * 100, tp + fn)


def f1_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Integer percent F1: 2 * P * R // (P + R), 0 when both are 0.
    
    Calculates the harmonic mean of precision and recall. Since it relies on 
    encrypted division, keeping sample sizes small is recommended.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.core import f1_score
        
        # Inside an FHE circuit
        # f1_pct = f1_score(enc_binary_preds, enc_binary_labels)
        ```
    """
    divide = make_floor_divide(zero_result=0)
    precision = precision_score(y_preds, y_trues)
    recall = recall_score(y_preds, y_trues)
    return divide(2 * precision * recall, precision + recall)


def r2_score(y_preds: List[Any], y_trues: List[Any]) -> Any:
    """Integer percent R²: 100 - 100 * SS_res // SS_tot.

    SS_tot uses the floor mean of ``y_trues``. When SS_tot is zero
    (constant targets) the result is 0. Like precision/recall, the
    encrypted-by-encrypted division uses a multivariate lookup — cheap on
    decrypted values, expensive under encryption for large ranges.
    """
    divide = make_floor_divide(zero_result=100)
    ss_res = euclidean_distance_squared(y_preds, y_trues)
    mean_true = array_sum(y_trues) // len(y_trues)
    ss_tot: Any = 0
    for value in y_trues:
        ss_tot = ss_tot + (value - mean_true) * (value - mean_true)
    return 100 - divide(100 * ss_res, ss_tot)
