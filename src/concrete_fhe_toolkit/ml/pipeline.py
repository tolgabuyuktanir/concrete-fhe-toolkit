"""sklearn-style pipelines that compile to a single FHE circuit."""

from __future__ import annotations

from typing import Any, List

from .classes import FHEModel


class FHEPipeline(FHEModel):
    """Chain preprocessing transformers and a final model into one circuit.

    Every intermediate step must expose ``_transform_logic(features)`` (the
    preprocessing transformers in
    :mod:`concrete_fhe_toolkit.ml.preprocessing` do); the final step must be
    an :class:`~concrete_fhe_toolkit.ml.classes.FHEModel`. The whole chain
    is traced into a **single** circuit, so there is one compile, one key
    set, and one encrypted round trip per prediction — intermediate values
    never leave the encrypted domain.

    Args:
        steps: Ordered list of transformers followed by exactly one model.

    Example:
        ```python
        from concrete_fhe_toolkit.ml import FHELogisticRegression
        from concrete_fhe_toolkit.ml.pipeline import FHEPipeline
        from concrete_fhe_toolkit.ml.preprocessing import FHEBinner

        pipeline = FHEPipeline([
            FHEBinner([[18, 30, 45, 65], [1000, 5000, 20000]]),
            FHELogisticRegression(weights=[3, 2], bias=-7),
        ])

        pipeline.compile(inputset=[[20, 800], [70, 30000], [40, 4000]])
        print(pipeline.predict([35, 7000]))
        # 1  (binned to [2, 2] -> score 3*2 + 2*2 - 7 = 3 >= 0)
        ```
    """

    def __init__(self, steps: List[Any]) -> None:
        super().__init__()
        if not steps:
            raise ValueError("steps must contain at least a final model")
        *transformers, model = steps
        for step in transformers:
            if not hasattr(step, "_transform_logic"):
                raise ValueError(
                    f"{type(step).__name__} is not a transformer: intermediate "
                    "pipeline steps must define _transform_logic(features)"
                )
        if not hasattr(model, "_circuit_logic"):
            raise ValueError(
                f"{type(model).__name__} is not a model: the final pipeline "
                "step must define _circuit_logic(features)"
            )
        self.steps = list(steps)

    def _circuit_logic(self, features: Any) -> Any:
        *transformers, model = self.steps
        current = features
        for step in transformers:
            current = step._transform_logic(current)
        return model._circuit_logic(current)


__all__ = ["FHEPipeline"]
