"""Encrypted inference helpers for simple ML models."""

from typing import Any, List, Optional

from .._compat import fhe
import numpy as np

from concrete_fhe_toolkit._utils import compile_function, validate_bounds, validate_integer
from concrete_fhe_toolkit.arrays import array_sum, make_argmax, make_argmin, array_sub
from concrete_fhe_toolkit.math import equal, greater, greater_equal, select, maximum
from concrete_fhe_toolkit.math._lookup import make_unary_lookup
from concrete_fhe_toolkit.arithmetic import sign

from .activations import relu, threshold_activation
from .core import euclidean_distance_squared
from .matrix import dot_product, matrix_vector_multiply, matrix_flatten


def linear_regression_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    """Evaluate a linear regression model (dot product of weights and features plus bias).
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import linear_regression_inference
        
        # Inside an FHE circuit
        # prediction = linear_regression_inference(
        #     weights=[2, -1], bias=5, features=[enc_f1, enc_f2]
        # )
        ```
    """
    product = dot_product(weights, features)
    return bias + product


def logistic_regression_inference(
    weights: List[Any],
    bias: Any,
    features: List[Any],
    *,
    threshold: int = 0,
) -> Any:
    """Evaluate a logistic regression classifier on encrypted features.

    Computes the linear score ``weights . features + bias`` and returns the
    binary class ``score >= threshold``. Because the sigmoid is monotonic,
    thresholding the raw score at 0 is equivalent to thresholding the
    probability at 0.5, so no sigmoid lookup is needed for classification.
    Scale weights, bias, and features to integers with the same factor
    before calling.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import logistic_regression_inference
        
        # Inside an FHE circuit
        # class_pred = logistic_regression_inference(
        #     weights=[3, 1], bias=-10, features=[enc_f1, enc_f2], threshold=0
        # )
        ```
    """
    score = linear_regression_inference(weights, bias, features)
    return threshold_activation(score, threshold)


def decision_tree_node(feature_val: Any, threshold: Any, left_branch: Any, right_branch: Any) -> Any:
    """Evaluate a single decision tree node.

    Returns ``left_branch`` when ``feature_val >= threshold``, otherwise
    ``right_branch``. Branch values may be arbitrary bounded integers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import decision_tree_node
        
        # Inside an FHE circuit
        # out = decision_tree_node(enc_feature, threshold=5, left_branch=1, right_branch=0)
        ```
    """
    control = greater_equal(feature_val, threshold)
    return select(control, left_branch, right_branch)


def majority_votes(predictions: List[Any]) -> Any:
    """Perform majority voting for an ensemble of binary predictions.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import majority_votes
        
        # Inside an FHE circuit
        # final_pred = majority_votes([enc_pred1, enc_pred2, enc_pred3])
        ```
    """
    sum_predictions = array_sum(predictions)
    return greater(sum_predictions, len(predictions) // 2)


def knn_inference(
    test_sample: List[Any],
    train_samples: List[List[Any]],
    train_labels: List[Any],
    *,
    k: int = 1,
    max_distance: int = 15,
) -> Any:
    """Evaluate a k-nearest-neighbor model on an encrypted test sample.

    ``max_distance`` must be an upper bound on the squared Euclidean distance
    between the test sample and any training sample. The argmin reductions
    are built for that range, so distances above the bound silently corrupt
    the result.

    With ``k=1`` the nearest label is returned directly and labels may be
    arbitrary bounded integers. With ``k > 1`` the labels must be binary
    (0/1) and the majority vote of the k nearest labels is returned; use an
    odd ``k`` to avoid ties (a tie resolves to 0).

    The k > 1 path runs k argmin rounds, masking each selected neighbor with
    a distance penalty, so circuit cost grows linearly with ``k``.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import knn_inference
        
        # Inside an FHE circuit
        # label = knn_inference(
        #     enc_test_sample, public_train_samples, public_train_labels, k=3
        # )
        ```
    """
    maximum_distance = validate_integer("max_distance", max_distance, minimum=1)
    normalized_k = validate_integer("k", k, minimum=1)
    if len(train_samples) != len(train_labels):
        raise ValueError("train_samples and train_labels must have the same length")
    if normalized_k > len(train_samples):
        raise ValueError("k cannot exceed the number of training samples")

    distances = [
        euclidean_distance_squared(row, test_sample)
        for row in train_samples
    ]

    if normalized_k == 1:
        argmin_func = make_argmin(len(distances), 0, maximum_distance)
        min_index = argmin_func(distances)

        prediction = 0
        for index, label in enumerate(train_labels):
            prediction += equal(index, min_index) * label

        return prediction

    # k > 1: iterated argmin. Each round the selected neighbor's distance is
    # pushed above max_distance so it cannot be selected again.
    penalty = maximum_distance + 1
    argmin_func = make_argmin(len(distances), 0, maximum_distance + penalty)
    current_distances = list(distances)
    votes = []

    for _ in range(normalized_k):
        min_index = argmin_func(current_distances)

        vote = 0
        next_distances = []
        for index, (distance, label) in enumerate(zip(current_distances, train_labels)):
            selected = equal(index, min_index)
            vote += selected * label
            next_distances.append(distance + selected * penalty)

        current_distances = next_distances
        votes.append(vote)

    return majority_votes(votes)


def decision_tree_inference(features: List[Any], tree: Any) -> Any:
    """Evaluate a full decision tree on encrypted features.

    ``tree`` is a public structure. An internal node is a dict with keys
    ``"feature"`` (index into ``features``), ``"threshold"`` (public
    integer), ``"left"``, and ``"right"`` (subtrees); a leaf is a plain
    integer label or value. The left branch is taken when
    ``features[feature] >= threshold``.

    Every path of the tree is evaluated obliviously, so the visited path is
    never revealed — circuit cost grows with the total number of nodes.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import decision_tree_inference
        
        # Public tree definition (trained offline)
        tree = {
            "feature": 0, "threshold": 5, 
            "left": 1, "right": 0
        }
        
        # Inside an FHE circuit
        # label = decision_tree_inference(enc_features, tree)
        ```
    """
    if not isinstance(tree, dict):
        return validate_integer("leaf value", tree)

    missing = {"feature", "threshold", "left", "right"} - tree.keys()
    if missing:
        raise ValueError(
            f"tree node is missing keys: {', '.join(sorted(missing))}"
        )

    feature_index = validate_integer("feature", tree["feature"], minimum=0)
    if feature_index >= len(features):
        raise ValueError("tree feature index is out of range for the feature vector")
    threshold = validate_integer("threshold", tree["threshold"])

    control = greater_equal(features[feature_index], threshold)
    left_value = decision_tree_inference(features, tree["left"])
    right_value = decision_tree_inference(features, tree["right"])
    return select(control, left_value, right_value)


def compile_decision_tree_node(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile a single encrypted decision tree node.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import compile_decision_tree_node
        
        circuit = compile_decision_tree_node(min_value=-10, max_value=10)
        # Allows evaluating one tree node completely obliviously
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        decision_tree_node,
        {
            "feature_val": "encrypted",
            "threshold": "encrypted",
            "left_branch": "encrypted",
            "right_branch": "encrypted",
        },
        [
            (minimum, minimum, minimum, minimum),
            (minimum, maximum, minimum, maximum),
            (maximum, minimum, maximum, minimum),
            (maximum, maximum, maximum, maximum),
        ],
        configuration,
    )


def random_forest_inference(features: List[Any], trees: List[Any]) -> Any:
    """Evaluate a random forest with binary (0/1) leaves via majority vote.

    Each tree uses the public dict structure accepted by
    :func:`decision_tree_inference`. Use an odd number of trees to avoid
    ties (a tie resolves to 0).
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import random_forest_inference
        
        # public_trees is a list of tree dictionaries
        # Inside an FHE circuit
        # label = random_forest_inference(enc_features, public_trees)
        ```
    """
    if not trees:
        raise ValueError("trees must contain at least one tree")
    predictions = [decision_tree_inference(features, tree) for tree in trees]
    return majority_votes(predictions)


def mlp_inference(features: List[Any], layers: List[Any]) -> List[Any]:
    """Evaluate a small multilayer perceptron with public integer weights.

    ``layers`` is a list of ``(weights_matrix, biases)`` pairs. Hidden
    layers apply ReLU; the final layer stays linear and returns the raw
    score vector. Weights, biases, and features must share one integer
    scale; note that every layer multiplies scales together, so keep the
    network shallow or rescale between layers.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import mlp_inference
        
        # layers = [(W1, b1), (W2, b2)] where Wi and bi are public lists
        # Inside an FHE circuit
        # scores = mlp_inference(enc_features, layers)
        ```
    """
    if not layers:
        raise ValueError("layers must contain at least one (weights, biases) pair")

    activations_vector = list(features)
    for layer_index, (weights, biases) in enumerate(layers):
        if len(weights) != len(biases):
            raise ValueError("each layer needs one bias per output row")
        scores = matrix_vector_multiply(weights, activations_vector)
        scores = [score + bias for score, bias in zip(scores, biases)]
        if layer_index < len(layers) - 1:
            scores = [relu(score) for score in scores]
        activations_vector = scores
    return activations_vector


def nearest_centroid_inference(
    sample: List[Any],
    centroids: List[List[Any]],
    labels: Optional[List[Any]] = None,
    *,
    max_distance: int = 15,
) -> Any:
    """Assign an encrypted sample to the nearest public centroid (k-means step).

    Returns the centroid index, or the matching label when ``labels`` is
    given. ``max_distance`` must bound the squared distance to any centroid.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import nearest_centroid_inference
        
        # Inside an FHE circuit
        # cluster_idx = nearest_centroid_inference(
        #     enc_sample, public_centroids, max_distance=100
        # )
        ```
    """
    maximum_distance = validate_integer("max_distance", max_distance, minimum=1)
    if not centroids:
        raise ValueError("centroids must contain at least one centroid")
    if labels is not None and len(labels) != len(centroids):
        raise ValueError("labels and centroids must have the same length")

    distances = [
        euclidean_distance_squared(centroid, sample)
        for centroid in centroids
    ]
    arg_min = make_argmin(len(distances), 0, maximum_distance)
    nearest_index = arg_min(distances)

    if labels is None:
        return nearest_index

    prediction: Any = 0
    for index, label in enumerate(labels):
        prediction = prediction + equal(index, nearest_index) * label
    return prediction


def argmax_inference(scores: List[Any], min_score: int, max_score: int) -> Any:
    """Return the index of the highest class score (multi-class head).
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import argmax_inference
        
        # Inside an FHE circuit
        # pred_class = argmax_inference(enc_scores, min_score=0, max_score=50)
        ```
    """
    items = list(scores)
    if not items:
        raise ValueError("scores must contain at least one score")
    arg_max = make_argmax(len(items), min_score, max_score)
    return arg_max(items)


def naive_bayes_inference(
    features: List[Any],
    log_prob_tables: List[List[List[int]]],
    priors: List[int],
    *,
    min_feature: int = 0,
) -> Any:
    """Evaluate a categorical naive Bayes classifier on encrypted features.

    ``log_prob_tables[class][feature]`` is a public list of scaled integer
    log-probabilities indexed by ``feature_value - min_feature``; ``priors``
    holds the scaled log-priors per class. Returns the encrypted index of
    the best class. Score bounds for the final argmax are derived from the
    public tables.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import naive_bayes_inference
        
        # Inside an FHE circuit
        # pred_class = naive_bayes_inference(
        #     enc_features, public_log_prob_tables, public_priors
        # )
        ```
    """
    if len(log_prob_tables) != len(priors):
        raise ValueError("log_prob_tables and priors must have the same length")
    if len(log_prob_tables) == 0:
        raise ValueError("at least one class is required")
    minimum_feature = validate_integer("min_feature", min_feature)

    scores = []
    lower_bounds = []
    upper_bounds = []
    for class_index, tables in enumerate(log_prob_tables):
        if len(tables) != len(features):
            raise ValueError("each class needs one table per feature")
        prior = validate_integer("prior", priors[class_index])
        score: Any = prior
        low = prior
        high = prior
        for feature_index, table in enumerate(tables):
            values = [validate_integer("log probability", value) for value in table]
            lookup = make_unary_lookup(values, minimum_feature)
            score = score + lookup(features[feature_index])
            low += min(values)
            high += max(values)
        scores.append(score)
        lower_bounds.append(low)
        upper_bounds.append(high)

    return argmax_inference(scores, min(lower_bounds), max(upper_bounds))

def svm_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    """Evaluate a linear Support Vector Machine (SVM) on encrypted features.
    
    Returns 1 if the sample is on the positive side of the hyperplane,
    -1 if it's on the negative side, and 0 if it lies exactly on the boundary.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import svm_inference
        
        # Inside an FHE circuit
        # sign_pred = svm_inference(public_weights, public_bias, enc_features)
        ```
    """
    regression_result = linear_regression_inference(weights, bias, features)
    return sign(regression_result)

def pca_inference(features: List[Any],means: List[Any],components: List[List[Any]]) -> List[Any]:
    """Apply Principal Component Analysis (PCA) to reduce dimensionality of encrypted data.

    Args:
        features: The encrypted feature vector (list of encrypted integers).
        mean: The public mean vector (list of integers).
        components: The public principal components matrix (list of lists of integers).

    Returns:
        The encrypted principal components vector (list of encrypted integers).
        
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import pca_inference
        
        # Inside an FHE circuit
        # enc_pca_features = pca_inference(enc_features, public_mean, public_components)
        ```
    """
    diffs = array_sub(features,means)
    return matrix_vector_multiply(components,diffs)

def xgboost_inference(features: List[Any],trees: List[Any]) -> Any:
    """
    Evaluate a XGBoost classifier on encrypted features.

    Args:
        features: The encrypted feature vector (list of encrypted integers).
        trees: The list of encrypted decision trees.

    Returns:
        The encrypted prediction of the XGBoost classifier.
        
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import xgboost_inference
        
        # Inside an FHE circuit
        # pred = xgboost_inference(enc_features, public_trees)
        ```
    """
    tree_sum = 0
    for tree in trees:
        tree_sum += decision_tree_inference(features,tree)

    return greater(tree_sum,0)

def cnn_inference(filters: List[List[List[Any]]], bias: List[Any], image: List[List[List[Any]]]) -> List[Any]:
    """Apply a 2D convolutional layer (CNN) to an encrypted image.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import cnn_inference
        
        # Inside an FHE circuit
        # enc_feature_map = cnn_inference(public_filters, public_bias, enc_image)
        ```
    """
    feature_map = []
    num_rows = len(filters[0])
    num_columns = len(filters[0][0])

    for index,filter in enumerate(filters):
        flatten_filter = matrix_flatten(filter)
        for i in range(len(image[0]) - num_rows + 1):
            conv_rows = image[0][i:i+num_rows]
            for j in range(len(image[0][0]) - num_columns + 1):
                conv_matrix = [row[j:j+num_columns] for row in conv_rows]
                flatten_conv_matrix = matrix_flatten(conv_matrix)
                product = dot_product(flatten_filter,flatten_conv_matrix)
                feature_map.append(product+bias[index])

    return feature_map            
    
def max_pooling_2d(image: List[List[List[Any]]]) -> List[Any]:
    """Apply 2D max pooling (2x2 kernel, stride 2) to an encrypted feature map.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import max_pooling_2d
        
        # Inside an FHE circuit
        # enc_pooled = max_pooling_2d(enc_image)
        ```
    """
    pooling_size = 2
    pooling_values = []

    for i in range(0,len(image[0]) - pooling_size + 1,pooling_size):
        pooling_rows = image[0][i:i + pooling_size]

        for j in range(0,len(image[0][0]) - pooling_size + 1,pooling_size):
            pooling_matrix = [row[j:j + pooling_size] for row in pooling_rows]
            flatten_matrix = matrix_flatten(pooling_matrix)

            max_element = flatten_matrix[0]
            for val in flatten_matrix[1:]:
                max_element = maximum(val,max_element)

            pooling_values.append(max_element)

    return pooling_values

def avg_pooling_2d(image: List[List[List[Any]]]) -> List[Any]:
    """Apply 2D average pooling (2x2 kernel, stride 2) to an encrypted feature map.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import avg_pooling_2d
        
        # Inside an FHE circuit
        # enc_pooled = avg_pooling_2d(enc_image)
        ```
    """
    pooling_size = 2
    pooling_values = []

    for i in range(0,len(image[0]) - pooling_size + 1,pooling_size):
        pooling_rows = image[0][i:i + pooling_size]

        for j in range(0,len(image[0][0]) - pooling_size + 1,pooling_size):
            pooling_matrix = [row[j:j + pooling_size] for row in pooling_rows]
            flatten_matrix = matrix_flatten(pooling_matrix)
            pooling_values.append(array_sum(flatten_matrix)//4)

    return pooling_values    

def auto_quantizer(images: List[List[List[List[Any]]]], filters: List[List[List[Any]]], model: Any, mode: str="optimal") -> Any:
    """Calculate the optimal scaling factor for quantizing network inputs/weights.
    
    This helps keep intermediate multiplications within the FHE bit-width limit.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.models import auto_quantizer
        
        scale_factor = auto_quantizer(plain_images, plain_filters, sklearn_model)
        ```
    """
    max_pixel_val = images[0][0][0][0]
    for image in images:
        for channel in image:
            for row in channel:
                for pixel in row:
                    max_pixel_val = max(abs(pixel),max_pixel_val)

    if mode == "optimal":
        weight_power = np.max(np.sum(np.abs(model.coef_), axis=1))
        filter_power = np.sum(np.abs(filters))
        cnn_pixel_count = 1
        num_of_features = 1
    
    elif mode == "worst_case":
        weight_power = np.max(np.abs(model.coef_))
        filter_power = filters[0][0][0]
        for filt in filters:
            for row in filt:
                for val in row:
                    filter_power = max(abs(val),filter_power)

        cnn_pixel_count = len(filters[0]) * len(filters[0][0])
        num_of_features = len(model.coef_[0])

    else:
        raise ValueError("mode should be optimal or worst_case")    
    
    max_product = max_pixel_val * filter_power * cnn_pixel_count * num_of_features * weight_power

    return max(1,int(32767 // max_product))