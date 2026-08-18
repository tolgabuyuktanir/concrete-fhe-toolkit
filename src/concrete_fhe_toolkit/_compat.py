"""Single import point for the Concrete compiler API.

Every module in this package imports Concrete through this facade, so an
upstream rename or move in ``concrete-python`` only needs to be absorbed
here instead of in every file.
"""

from __future__ import annotations

try:
    from concrete import fhe
except ImportError as error:  # pragma: no cover - depends on environment
    raise ImportError(
        "concrete-fhe-toolkit requires concrete-python. "
        "Install it with: pip install concrete-python"
    ) from error

try:
    from importlib.metadata import version

    CONCRETE_VERSION = version("concrete-python")
except Exception:  # pragma: no cover - metadata may be unavailable
    CONCRETE_VERSION = "unknown"

# The toolkit's entire runtime surface of Concrete. If any of these are
# missing after an upgrade, fail at import time with a clear message instead
# of deep inside a compile call.
_REQUIRED_ATTRIBUTES = (
    "Circuit",
    "Compiler",
    "Configuration",
    "LookupTable",
    "array",
    "multivariate",
)
_missing = [name for name in _REQUIRED_ATTRIBUTES if not hasattr(fhe, name)]
if _missing:
    raise ImportError(
        f"concrete-python {CONCRETE_VERSION} is missing expected API: "
        f"{', '.join(_missing)}. This concrete-fhe-toolkit release may not "
        "support that Concrete version."
    )

__all__ = ["CONCRETE_VERSION", "fhe"]
