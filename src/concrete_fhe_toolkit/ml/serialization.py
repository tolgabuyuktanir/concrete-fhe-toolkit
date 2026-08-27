"""Save and load FHE models as portable JSON files.

Every parametric model class can be serialized: its public parameters
(weights, trees, centroids, tables) go into a small JSON document that can
be versioned, shipped to a server, or reloaded years later. Compiled
circuits and keys are **not** serialized — recompile after loading (that is
cheap and keeps key material out of model files).

Example:
    ```python
    from concrete_fhe_toolkit.ml import FHELogisticRegression
    from concrete_fhe_toolkit.ml.serialization import load_model, save_model

    model = FHELogisticRegression(weights=[3, 2], bias=-7)
    save_model(model, "diagnosis.json")

    restored = load_model("diagnosis.json")
    restored.compile(inputset)
    restored.predict(sample)
    ```
"""

from __future__ import annotations

import json
from typing import Any

from .classes import (
    FHECNN,
    FHEDecisionTree,
    FHEKMeans,
    FHEKNN,
    FHELinearRegression,
    FHELogisticRegression,
    FHEMLP,
    FHENaiveBayes,
    FHEPCA,
    FHERandomForest,
    FHESVM,
    FHEXGBoost,
)

FORMAT = "concrete-fhe-toolkit/model"
FORMAT_VERSION = 1

# class -> (positional constructor attrs, keyword-only constructor attrs)
_REGISTRY = {
    "FHELogisticRegression": (FHELogisticRegression, ["weights", "bias"], []),
    "FHELinearRegression": (FHELinearRegression, ["weights", "bias"], []),
    "FHEDecisionTree": (FHEDecisionTree, ["tree"], []),
    "FHERandomForest": (FHERandomForest, ["trees"], []),
    "FHEXGBoost": (FHEXGBoost, ["trees"], []),
    "FHESVM": (FHESVM, ["weights", "bias"], []),
    "FHEKNN": (FHEKNN, ["X_train", "y_train", "k"], []),
    "FHENaiveBayes": (FHENaiveBayes, ["log_prob_tables", "priors"], []),
    "FHEMLP": (FHEMLP, ["mlp_layers"], []),
    "FHEPCA": (FHEPCA, ["means", "components"], []),
    "FHECNN": (FHECNN, ["filters", "bias"], []),
    "FHEKMeans": (FHEKMeans, ["centroids"], ["max_distance"]),
}

# Optional attributes preserved when present (set by trainers).
_EXTRAS = ("output_scale",)


def save_model(model: Any, path: str) -> None:
    """Serialize a model's public parameters to a JSON file.

    Args:
        model: Any registered ``FHEModel`` subclass instance (including
            trainer outputs).
        path: Destination file path.

    Raises:
        ValueError: When the model class is not serializable (for example
            a custom subclass or an ``FHEPipeline`` — persist a pipeline by
            saving its final model and reconstructing the transformers).

    Example:
        ```python
        model = trainer.fit_encrypted(X_train, y_train)
        save_model(model, "scorecard.json")
        ```
    """
    name = type(model).__name__
    if name not in _REGISTRY:
        raise ValueError(
            f"{name} is not serializable; supported models: "
            + ", ".join(sorted(_REGISTRY))
        )
    _, positional, keyword_only = _REGISTRY[name]
    params = {attr: getattr(model, attr) for attr in positional + keyword_only}
    extras = {
        attr: getattr(model, attr)
        for attr in _EXTRAS
        if hasattr(model, attr)
    }
    document = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model": name,
        "params": params,
        "extras": extras,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def load_model(path: str) -> Any:
    """Load a model saved with :func:`save_model`.

    The returned model is uncompiled — call ``compile(inputset)`` before
    predicting.

    Args:
        path: Path to a JSON file produced by :func:`save_model`.

    Returns:
        A fresh instance of the original model class.

    Example:
        ```python
        model = load_model("scorecard.json")
        model.compile(inputset)
        print(model.predict(sample))
        ```
    """
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)

    if document.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    name = document.get("model")
    if name not in _REGISTRY:
        raise ValueError(f"unknown model class in file: {name!r}")

    cls, positional, keyword_only = _REGISTRY[name]
    params = document.get("params", {})
    missing = [attr for attr in positional + keyword_only if attr not in params]
    if missing:
        raise ValueError(f"model file is missing parameters: {', '.join(missing)}")

    model = cls(
        *[params[attr] for attr in positional],
        **{attr: params[attr] for attr in keyword_only},
    )
    for attr, value in document.get("extras", {}).items():
        if attr in _EXTRAS:
            setattr(model, attr, value)
    return model


__all__ = ["FORMAT", "FORMAT_VERSION", "load_model", "save_model"]
