"""Encrypted preprocessing transformers for FHE pipelines.

Every transformer here exposes ``_transform_logic(features) -> features`` so
it can be chained inside an :class:`~concrete_fhe_toolkit.ml.pipeline.FHEPipeline`
in front of a model — the whole chain compiles into a single circuit.
Transformer parameters (means, scales, bin edges) are public integers fitted
clear-side; only the features flowing through them are encrypted.
"""

from __future__ import annotations

from typing import Any, List

from .._utils import validate_integer
from ..math import greater_equal


def bin_feature(value: Any, bin_edges: List[int]) -> Any:
    """Map an encrypted value to its ordinal bucket index.

    Counts how many public bin edges the value has reached, so the result
    is ``0`` below the first edge, ``1`` between the first and second edge,
    and so on up to ``len(bin_edges)``. This is the standard scorecard
    building block: it turns a continuous feature into a small categorical
    one without revealing the value.

    Args:
        value: The encrypted value to discretize.
        bin_edges: Public, ascending list of bucket boundaries. A value
            lands in bucket ``i`` when ``bin_edges[i-1] <= value < bin_edges[i]``.

    Returns:
        The encrypted bucket index in ``[0, len(bin_edges)]``.

    Example:
        ```python
        from concrete_fhe_toolkit.ml.preprocessing import bin_feature

        print(bin_feature(35, [18, 30, 45, 65]))   # 2  (30 <= 35 < 45)
        print(bin_feature(70, [18, 30, 45, 65]))   # 4  (past every edge)
        # Inside an FHE circuit the same call works on encrypted values.
        ```
    """
    if not bin_edges:
        raise ValueError("bin_edges must contain at least one edge")
    bucket: Any = 0
    for edge in bin_edges:
        bucket = bucket + greater_equal(value, validate_integer("bin edge", edge))
    return bucket


class FHEBinner:
    """Discretize every feature into ordinal buckets (scorecard binning).

    Args:
        bin_edges: Per feature, the public ascending list of bucket
            boundaries used by :func:`bin_feature`.

    Example:
        ```python
        from concrete_fhe_toolkit.ml.preprocessing import FHEBinner

        binner = FHEBinner([[18, 30, 45, 65], [1000, 5000]])
        print(binner._transform_logic([35, 7000]))  # [2, 2]
        # Typically used as a pipeline step:
        # FHEPipeline([binner, FHELogisticRegression(weights, bias)])
        ```
    """

    def __init__(self, bin_edges: List[List[int]]) -> None:
        if not bin_edges:
            raise ValueError("bin_edges must describe at least one feature")
        self.bin_edges = [list(edges) for edges in bin_edges]

    def _transform_logic(self, features: Any) -> List[Any]:
        if isinstance(features, (list, tuple)) and len(features) != len(self.bin_edges):
            raise ValueError("expected one bin-edge list per feature")
        return [
            bin_feature(features[index], edges)
            for index, edges in enumerate(self.bin_edges)
        ]


class FHEStandardScaler:
    """Standardize features with public means and standard deviations.

    Computes ``((x - mean) * scale) // std`` per feature, i.e. the z-score
    kept as a scaled integer. Fit the means/stds clear-side (for example
    with sklearn's ``StandardScaler``) and round them to integers.

    Args:
        means: Public per-feature integer means.
        stds: Public per-feature integer standard deviations (must be >= 1).
        scale: Output scale of the z-scores (default 10, so ``13`` means 1.3).

    Example:
        ```python
        from concrete_fhe_toolkit.ml.preprocessing import FHEStandardScaler

        scaler = FHEStandardScaler(means=[50, 100], stds=[10, 20], scale=10)
        print(scaler._transform_logic([65, 60]))  # [15, -20]  (z = 1.5, -2.0)
        ```
    """

    def __init__(self, means: List[int], stds: List[int], *, scale: int = 10) -> None:
        if len(means) != len(stds):
            raise ValueError("means and stds must have the same length")
        self.means = [validate_integer("mean", value) for value in means]
        self.stds = [validate_integer("std", value, minimum=1) for value in stds]
        self.scale = validate_integer("scale", scale, minimum=1)

    def _transform_logic(self, features: Any) -> List[Any]:
        if isinstance(features, (list, tuple)) and len(features) != len(self.means):
            raise ValueError("expected one feature per fitted mean")
        return [
            ((features[index] - mean) * self.scale) // std
            for index, (mean, std) in enumerate(zip(self.means, self.stds))
        ]


class FHEMinMaxScaler:
    """Rescale features into ``[0, scale]`` with public min/max bounds.

    Computes ``((x - minimum) * scale) // (maximum - minimum)`` per feature.

    Args:
        minimums: Public per-feature lower bounds.
        maximums: Public per-feature upper bounds (strictly greater).
        scale: Upper end of the output range (default 100).

    Example:
        ```python
        from concrete_fhe_toolkit.ml.preprocessing import FHEMinMaxScaler

        scaler = FHEMinMaxScaler([0, 10], [200, 20], scale=100)
        print(scaler._transform_logic([50, 15]))  # [25, 50]
        ```
    """

    def __init__(
        self,
        minimums: List[int],
        maximums: List[int],
        *,
        scale: int = 100,
    ) -> None:
        if len(minimums) != len(maximums):
            raise ValueError("minimums and maximums must have the same length")
        self.minimums = [validate_integer("minimum", value) for value in minimums]
        self.maximums = [validate_integer("maximum", value) for value in maximums]
        for low, high in zip(self.minimums, self.maximums):
            if high <= low:
                raise ValueError("every maximum must be greater than its minimum")
        self.scale = validate_integer("scale", scale, minimum=1)

    def _transform_logic(self, features: Any) -> List[Any]:
        if isinstance(features, (list, tuple)) and len(features) != len(self.minimums):
            raise ValueError("expected one feature per fitted bound")
        return [
            ((features[index] - low) * self.scale) // (high - low)
            for index, (low, high) in enumerate(zip(self.minimums, self.maximums))
        ]


__all__ = [
    "FHEBinner",
    "FHEMinMaxScaler",
    "FHEStandardScaler",
    "bin_feature",
]
