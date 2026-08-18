# API reference overview

These pages are generated directly from the package docstrings, so they
always match the version of the code they were built from. Use the version
selector in the header to switch between releases.

## Layers

| Layer | Modules | What lives there |
| --- | --- | --- |
| Core | [`arithmetic`](arithmetic.md), [`arrays`](arrays.md) | compare/sign, bounded division, sorting networks, reductions, array utilities |
| Math | [`basic`](math-basic.md), [`combinatorics`](math-combinatorics.md), [`number_theory`](math-number-theory.md), [`fixed_point`](math-fixed-point.md), [`special`](math-special.md), [`bits`](math-bits.md), [`binary_division`](math-binary-division.md) | Python-`math`-style bounded helpers, LUT guardrails, bit-level circuits |
| Machine learning | [`core`](ml-core.md), [`models`](ml-models.md), [`matrix`](ml-matrix.md), [`activations`](ml-activations.md), [`stats`](ml-stats.md), [`utils`](ml-utils.md) | encrypted metrics, model inference, matrix algebra, activations |
| Finance | [`core`](finance-core.md), [`transactions`](finance-transactions.md) | rate/tax/interest helpers under the `RATE_SCALE` money convention |

## Calling conventions

Most operations come in three forms:

- **`operation(...)`** — friendly `BoundedOperation` objects in
  `concrete_fhe_toolkit.math` compile a circuit directly.
- **`compile_*(...)`** — explicit compiler returning an `fhe.Circuit`.
- **`make_*(...)`** — returns a traceable function for composing your own
  larger Concrete program.

All inputs are bounded integers: choose public bounds at compile time and
keep runtime inputs inside them. See the
[API usage guide](../api-reference.md) and
[Bounds and costs](../bounds-and-costs.md) for the full model.
