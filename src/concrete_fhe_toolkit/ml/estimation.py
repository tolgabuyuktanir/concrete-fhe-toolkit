"""Static cost estimation for FHE models — before compiling anything.

Answers "how expensive will this model be?" from the model's public
structure and the feature bounds alone: it counts the encrypted operations
the circuit will contain (comparisons, table lookups, multiplications) and
the widest lookup domain, then maps them to the same qualitative levels the
lookup guardrails use. Estimates are structural, not wall-clock promises —
use them to compare models and to catch circuits that will not fit before
paying for a compile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from .._utils import validate_bounds


@dataclass
class ModelCostEstimate:
    """Structural cost summary for one model at given feature bounds.

    Attributes:
        comparisons: Encrypted comparison operations (each becomes a TLU).
        lookups: Explicit table lookups (argmin/argmax encodings included).
        multiplications: Encrypted multiplications.
        max_lookup_input_bits: Widest lookup-domain bit width in the
            circuit — the main driver of key size and runtime.
        level: ``small`` / ``moderate`` / ``large`` / ``very-large``,
            aligned with the ``estimate_lookup_cost`` guardrail levels.
        notes: Human-readable advisories (for example when an argmin
            encoding approaches Concrete's practical bit-width limits).
    """

    comparisons: int = 0
    lookups: int = 0
    multiplications: int = 0
    max_lookup_input_bits: int = 0
    level: str = "small"
    notes: List[str] = field(default_factory=list)


def _bits(domain_size: int) -> int:
    return max(1, (max(1, domain_size) - 1).bit_length())


def _tree_stats(tree: Any) -> tuple:
    if not isinstance(tree, dict):
        return 0, 0  # leaf: no comparison, depth 0
    left_nodes, left_depth = _tree_stats(tree["left"])
    right_nodes, right_depth = _tree_stats(tree["right"])
    return 1 + left_nodes + right_nodes, 1 + max(left_depth, right_depth)


def _arg_extreme_cost(size: int, value_span: int) -> tuple:
    """(lookups, input_bits) of one argmin/argmax reduction."""
    encoded_span = value_span * size + size - 1
    return size, _bits(2 * encoded_span + 1)


def estimate_model_cost(
    model: Any,
    *,
    min_feature: int,
    max_feature: int,
) -> ModelCostEstimate:
    """Estimate a model's encrypted-operation footprint before compiling.

    Args:
        model: Any toolkit model instance (``FHEModel`` subclasses,
            including pipelines — steps are summed).
        min_feature: Inclusive lower bound of the encrypted features the
            model will see.
        max_feature: Inclusive upper bound of those features.

    Returns:
        A :class:`ModelCostEstimate` with operation counts and a
        qualitative level.

    Example:
        ```python
        from concrete_fhe_toolkit.ml import FHEKNN
        from concrete_fhe_toolkit.ml.estimation import estimate_model_cost

        model = FHEKNN(X_train, y_train, k=3)
        report = estimate_model_cost(model, min_feature=0, max_feature=15)
        print(report.level, report.max_lookup_input_bits, *report.notes)
        ```
    """
    minimum, maximum = validate_bounds(min_feature, max_feature)
    span = maximum - minimum
    estimate = ModelCostEstimate()
    _accumulate(model, estimate, span)

    bits = estimate.max_lookup_input_bits
    if bits >= 16:
        estimate.level = "very-large"
        estimate.notes.append(
            f"a lookup needs {bits}-bit inputs — likely beyond Concrete's "
            "practical parameter space; shrink bounds or sizes"
        )
    elif bits >= 10 or estimate.lookups + estimate.comparisons >= 200:
        estimate.level = "very-large"
    elif bits >= 8 or estimate.lookups + estimate.comparisons >= 60:
        estimate.level = "large"
    elif bits >= 7 or estimate.lookups + estimate.comparisons >= 20:
        estimate.level = "moderate"
    else:
        estimate.level = "small"
    return estimate


def _accumulate(model: Any, out: ModelCostEstimate, span: int) -> None:
    name = type(model).__name__

    if name == "FHEPipeline":
        for step in model.steps:
            _accumulate(step, out, span)
        return

    if name == "FHEBinner":
        out.comparisons += sum(len(edges) for edges in model.bin_edges)
        return
    if name in ("FHEStandardScaler", "FHEMinMaxScaler"):
        count = len(getattr(model, "means", getattr(model, "minimums", [])))
        out.multiplications += count
        out.lookups += count  # division by a public constant becomes a TLU
        return

    if name in ("FHELogisticRegression", "FHELinearRegression", "FHESVM"):
        out.multiplications += len(model.weights)
        if name != "FHELinearRegression":
            out.comparisons += 1
        return

    if name == "FHEDecisionTree":
        nodes, _ = _tree_stats(model.tree)
        out.comparisons += nodes
        out.multiplications += 2 * nodes  # oblivious select per node
        return

    if name in ("FHERandomForest", "FHEXGBoost"):
        for tree in model.trees:
            nodes, _ = _tree_stats(tree)
            out.comparisons += nodes
            out.multiplications += 2 * nodes
        out.comparisons += 1  # vote / threshold
        return

    if name == "FHEKNN":
        samples = len(model.X_train)
        features = len(model.X_train[0]) if samples else 0
        rounds = max(1, int(getattr(model, "k", 1)))
        out.multiplications += samples * features  # squared differences
        lookups, bits = _arg_extreme_cost(samples, features * span * span)
        out.lookups += lookups * rounds + samples * rounds  # argmin + masking
        out.max_lookup_input_bits = max(out.max_lookup_input_bits, bits)
        return

    if name == "FHEKMeans":
        clusters = len(model.centroids)
        features = len(model.centroids[0]) if clusters else 0
        out.multiplications += clusters * features
        lookups, bits = _arg_extreme_cost(clusters, features * span * span)
        out.lookups += lookups
        out.max_lookup_input_bits = max(out.max_lookup_input_bits, bits)
        return

    if name == "FHENaiveBayes":
        classes = len(model.priors)
        features = len(model.log_prob_tables[0]) if classes else 0
        out.lookups += classes * features
        score_span = sum(
            max(table) - min(table)
            for table in model.log_prob_tables[0]
        ) if classes else 0
        lookups, bits = _arg_extreme_cost(classes, max(1, score_span))
        out.lookups += lookups
        out.max_lookup_input_bits = max(out.max_lookup_input_bits, bits)
        return

    if name == "FHEMLP":
        for layer_index, (weights, _biases) in enumerate(model.mlp_layers):
            rows = len(weights)
            columns = len(weights[0]) if rows else 0
            out.multiplications += rows * columns
            if layer_index < len(model.mlp_layers) - 1:
                out.comparisons += rows  # relu per hidden unit
        return

    raise ValueError(
        f"cannot estimate {name}; supported: pipeline steps and the "
        "parametric FHE model classes"
    )


__all__ = ["ModelCostEstimate", "estimate_model_cost"]
