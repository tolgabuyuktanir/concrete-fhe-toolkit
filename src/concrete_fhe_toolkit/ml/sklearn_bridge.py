"""Convert trained scikit-learn models into toolkit FHE models.

These helpers kill the manual scaling ritual: train with sklearn on clear
data as usual, convert once, and run encrypted inference with the toolkit.
scikit-learn is imported lazily, so it stays an optional dependency.

Remember the shared-scale rule: features fed to the FHE model must be
quantized with the **same** ``scale`` used here (``round(value * scale)``).
"""

from __future__ import annotations

from typing import Any, List

from .._utils import validate_integer
from .classes import (
    FHEDecisionTree,
    FHELinearRegression,
    FHELogisticRegression,
    FHERandomForest,
)


def from_sklearn_linear(model: Any, *, scale: int = 100) -> Any:
    """Convert a fitted sklearn linear model into an FHE model.

    ``LogisticRegression`` (binary) becomes an
    :class:`FHELogisticRegression`; ``LinearRegression`` (and other
    regressors exposing ``coef_``/``intercept_``) become an
    :class:`FHELinearRegression` whose predictions are scaled by ``scale``.

    Args:
        model: A fitted sklearn estimator with ``coef_`` and ``intercept_``.
        scale: Integer factor used to quantize the float weights. Quantize
            inference features with the same factor.

    Returns:
        A ready-to-compile toolkit model.

    Example:
        ```python
        from sklearn.linear_model import LogisticRegression
        from concrete_fhe_toolkit.ml.sklearn_bridge import from_sklearn_linear

        clf = LogisticRegression().fit(X_scaled, y)
        fhe_model = from_sklearn_linear(clf, scale=10)
        fhe_model.compile(quantized_inputset)
        fhe_model.predict(quantized_sample)
        ```
    """
    normalized_scale = validate_integer("scale", scale, minimum=1)
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("model must be a fitted sklearn linear estimator")

    import numpy as np

    coefficients = np.atleast_2d(np.asarray(model.coef_))
    intercepts = np.atleast_1d(np.asarray(model.intercept_))
    if coefficients.shape[0] != 1:
        raise ValueError(
            "only binary/single-output linear models are supported; "
            "use one converted model per class for multi-class"
        )

    weights = [int(round(value * normalized_scale)) for value in coefficients[0]]
    bias = int(round(float(intercepts[0]) * normalized_scale))

    is_classifier = hasattr(model, "classes_")
    if is_classifier:
        return FHELogisticRegression(weights, bias)
    converted = FHELinearRegression(weights, bias)
    converted.output_scale = normalized_scale
    return converted


def _convert_tree_node(tree: Any, node: int, *, scale: int, leaf_scale: int) -> Any:
    left = tree.children_left[node]
    right = tree.children_right[node]
    if left == -1:  # leaf
        values = tree.value[node][0]
        if len(values) == 1:  # regression leaf
            return int(round(float(values[0]) * leaf_scale))
        return int(values.argmax())  # classification leaf: majority class

    # sklearn splits as "feature <= threshold goes left"; the toolkit splits
    # as "feature >= threshold goes left" — so sklearn's right child is our
    # left branch. For integers, x > t is x >= floor(t) + 1 (floor, not
    # round: rounding up would shift fractional thresholds off by one).
    import math

    threshold = math.floor(float(tree.threshold[node]) * scale) + 1
    return {
        "feature": int(tree.feature[node]),
        "threshold": threshold,
        "left": _convert_tree_node(tree, right, scale=scale, leaf_scale=leaf_scale),
        "right": _convert_tree_node(tree, left, scale=scale, leaf_scale=leaf_scale),
    }


def from_sklearn_tree(model: Any, *, scale: int = 1, leaf_scale: int = 1) -> Any:
    """Convert a fitted sklearn decision tree into an :class:`FHEDecisionTree`.

    Classification trees keep their majority-class integer leaves;
    regression trees get leaves quantized by ``leaf_scale``. Thresholds are
    quantized by ``scale`` — quantize inference features with the same
    factor.

    Example:
        ```python
        from sklearn.tree import DecisionTreeClassifier
        from concrete_fhe_toolkit.ml.sklearn_bridge import from_sklearn_tree

        clf = DecisionTreeClassifier(max_depth=3).fit(X_int, y)
        fhe_tree = from_sklearn_tree(clf)
        fhe_tree.compile(inputset)
        ```
    """
    normalized_scale = validate_integer("scale", scale, minimum=1)
    normalized_leaf_scale = validate_integer("leaf_scale", leaf_scale, minimum=1)
    if not hasattr(model, "tree_"):
        raise ValueError("model must be a fitted sklearn decision tree")
    tree_dict = _convert_tree_node(
        model.tree_, 0, scale=normalized_scale, leaf_scale=normalized_leaf_scale
    )
    return FHEDecisionTree(tree_dict)


def from_sklearn_forest(model: Any, *, scale: int = 1) -> Any:
    """Convert a fitted sklearn random-forest classifier into an
    :class:`FHERandomForest` (binary labels, majority vote).

    Example:
        ```python
        from sklearn.ensemble import RandomForestClassifier
        from concrete_fhe_toolkit.ml.sklearn_bridge import from_sklearn_forest

        clf = RandomForestClassifier(n_estimators=5, max_depth=3).fit(X_int, y)
        fhe_forest = from_sklearn_forest(clf)
        ```
    """
    if not hasattr(model, "estimators_"):
        raise ValueError("model must be a fitted sklearn forest")
    trees: List[Any] = [
        from_sklearn_tree(estimator, scale=scale).tree
        for estimator in model.estimators_
    ]
    return FHERandomForest(trees)


__all__ = ["from_sklearn_forest", "from_sklearn_linear", "from_sklearn_tree"]
