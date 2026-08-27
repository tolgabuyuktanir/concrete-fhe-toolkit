"""Differential-privacy mechanisms for released (decrypted) aggregates.

The encrypted trainers in :mod:`concrete_fhe_toolkit.ml.trainers` follow the
aggregate-decrypt pattern: raw samples stay encrypted, but aggregate
statistics (counts, sums) are decrypted to build the model. Those aggregates
still carry *some* information about individuals. Adding calibrated noise to
them right after decryption turns that informal leak into a formal
differential-privacy guarantee — the classic FHE + DP combination.

These helpers are **clear-side**: apply them to already-decrypted integers.
Accounting is the caller's responsibility — when one individual affects
``k`` released values, either scale ``sensitivity`` accordingly or split
your epsilon budget across the releases.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


def _validate_positive(name: str, value: float) -> float:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def laplace_mechanism(
    value: int,
    *,
    sensitivity: float,
    epsilon: float,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Release an integer aggregate with epsilon-DP Laplace noise.

    Adds noise drawn from ``Laplace(0, sensitivity / epsilon)`` and rounds
    back to an integer. ``sensitivity`` is how much the aggregate can change
    when one individual's data changes (1 for a simple count).

    Args:
        value: The decrypted aggregate to release.
        sensitivity: L1 sensitivity of the aggregate.
        epsilon: Privacy budget for this release (smaller = more private).
        rng: Optional ``numpy.random.Generator`` for reproducible noise.

    Returns:
        The noised integer, safe to publish under epsilon-DP.

    Example:
        ```python
        import numpy as np
        from concrete_fhe_toolkit.privacy import laplace_mechanism

        # A decrypted patient count, released with a privacy guarantee:
        noisy = laplace_mechanism(
            842, sensitivity=1, epsilon=1.0,
            rng=np.random.default_rng(42),
        )
        print(noisy)  # 842 +/- a few
        ```
    """
    scale = _validate_positive("sensitivity", sensitivity) / _validate_positive(
        "epsilon", epsilon
    )
    generator = rng if rng is not None else np.random.default_rng()
    return int(round(int(value) + generator.laplace(0.0, scale)))


def gaussian_mechanism(
    value: int,
    *,
    sensitivity: float,
    epsilon: float,
    delta: float,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Release an integer aggregate with (epsilon, delta)-DP Gaussian noise.

    Uses the analytic calibration ``sigma = sensitivity * sqrt(2 ln(1.25/delta))
    / epsilon`` (valid for ``epsilon <= 1``). Prefer this over Laplace when
    many values are released together, since Gaussian noise composes better.

    Args:
        value: The decrypted aggregate to release.
        sensitivity: L2 sensitivity of the aggregate.
        epsilon: Privacy budget (must be in ``(0, 1]`` for this calibration).
        delta: Failure probability (for example ``1e-5``).
        rng: Optional ``numpy.random.Generator`` for reproducible noise.

    Returns:
        The noised integer.

    Example:
        ```python
        import numpy as np
        from concrete_fhe_toolkit.privacy import gaussian_mechanism

        noisy = gaussian_mechanism(
            842, sensitivity=1, epsilon=0.5, delta=1e-5,
            rng=np.random.default_rng(7),
        )
        ```
    """
    eps = _validate_positive("epsilon", epsilon)
    if eps > 1:
        raise ValueError("epsilon must be at most 1 for the Gaussian calibration")
    dlt = _validate_positive("delta", delta)
    if dlt >= 1:
        raise ValueError("delta must be smaller than 1")
    sigma = (
        _validate_positive("sensitivity", sensitivity)
        * math.sqrt(2 * math.log(1.25 / dlt))
        / eps
    )
    generator = rng if rng is not None else np.random.default_rng()
    return int(round(int(value) + generator.normal(0.0, sigma)))


def dp_release(
    values: Sequence[int],
    *,
    sensitivity: float,
    epsilon: float,
    mechanism: str = "laplace",
    delta: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Noise a whole vector of decrypted aggregates with one shared budget.

    The given ``epsilon`` is the budget for the *entire* release: it is
    split evenly across the values (simple composition), so releasing more
    values means more noise per value. Use it on the aggregate vectors the
    encrypted trainers decrypt (class counts, cluster sums, ...).

    Args:
        values: The decrypted aggregates to release together.
        sensitivity: Per-value sensitivity to one individual's change.
        epsilon: Total privacy budget for the whole vector.
        mechanism: ``"laplace"`` (pure epsilon-DP) or ``"gaussian"``
            (requires ``delta``).
        delta: Failure probability for the Gaussian mechanism.
        rng: Optional ``numpy.random.Generator`` for reproducible noise.

    Returns:
        The noised integers, in the same order.

    Example:
        ```python
        import numpy as np
        from concrete_fhe_toolkit.privacy import dp_release

        # Class counts decrypted by an encrypted trainer:
        counts = [412, 430]
        noisy = dp_release(
            counts, sensitivity=1, epsilon=1.0,
            rng=np.random.default_rng(0),
        )
        print(noisy)  # e.g. [410, 431] — safe to use clear-side
        ```
    """
    items = list(values)
    if not items:
        return []
    per_value_epsilon = _validate_positive("epsilon", epsilon) / len(items)
    generator = rng if rng is not None else np.random.default_rng()
    if mechanism == "laplace":
        return [
            laplace_mechanism(
                value,
                sensitivity=sensitivity,
                epsilon=per_value_epsilon,
                rng=generator,
            )
            for value in items
        ]
    if mechanism == "gaussian":
        if delta is None:
            raise ValueError("the gaussian mechanism requires delta")
        return [
            gaussian_mechanism(
                value,
                sensitivity=sensitivity,
                epsilon=per_value_epsilon,
                delta=delta,
                rng=generator,
            )
            for value in items
        ]
    raise ValueError("mechanism must be 'laplace' or 'gaussian'")


__all__ = [
    "dp_release",
    "gaussian_mechanism",
    "laplace_mechanism",
]
