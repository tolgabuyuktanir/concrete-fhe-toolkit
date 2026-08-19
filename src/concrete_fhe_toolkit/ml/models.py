"""Encrypted inference helpers for simple ML models."""

from typing import Any, List, Optional

from .._compat import fhe

from concrete_fhe_toolkit._utils import compile_function, validate_bounds, validate_integer
from concrete_fhe_toolkit.arrays import array_sum, make_argmax, make_argmin, array_sub
from concrete_fhe_toolkit.math import equal, greater, greater_equal, select
from concrete_fhe_toolkit.math._lookup import make_unary_lookup
from concrete_fhe_toolkit.arithmetic import sign

from .activations import relu, threshold_activation
from .core import euclidean_distance_squared
from .matrix import dot_product, matrix_vector_multiply, matrix_flatten


def linear_regression_inference(weights: List[Any], bias: Any, features: List[Any]) -> Any:
    """Evaluate a linear regression model (dot product of weights and features plus bias)."""
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
    """
    score = linear_regression_inference(weights, bias, features)
    return threshold_activation(score, threshold)


def decision_tree_node(feature_val: Any, threshold: Any, left_branch: Any, right_branch: Any) -> Any:
    """Evaluate a single decision tree node.

    Returns ``left_branch`` when ``feature_val >= threshold``, otherwise
    ``right_branch``. Branch values may be arbitrary bounded integers.
    """
    control = greater_equal(feature_val, threshold)
    return select(control, left_branch, right_branch)


def majority_votes(predictions: List[Any]) -> Any:
    """Perform majority voting for an ensemble of binary predictions."""
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
    """Compile a single encrypted decision tree node."""
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
    """Return the index of the highest class score (multi-class head)."""
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
    """
    if len(log_prob_tables) != len(priors):
        raise ValueError("log_prob_tables and priors must have the same length")
    if not log_prob_tables:
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
    """
    tree_sum = 0
    for tree in trees:
        tree_sum += decision_tree_inference(features,tree)

    return greater(tree_sum,0)

def cnn_inference(filters: List[List[List[Any]]], bias: List[Any], image: List[List[List[Any]]]) -> Any:
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
    
