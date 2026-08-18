# Roadmap / TODO

Working list for the team. Check items off in the PR that completes them.

## High priority

- [x] **Real k-NN**: `knn_inference` now takes `k` (iterated argmin with
  masking + `majority_votes`; binary labels for `k > 1`).
- [x] **Logistic regression helper**: `logistic_regression_inference`
  (linear score + threshold; sigmoid not needed for classification).
- [x] **Full decision-tree evaluator**: `decision_tree_inference` evaluates
  a public dict-based tree obliviously over encrypted features.

## API consistency

- [ ] Move `ml` (and later `finance`) toward the `make_*` / `compile_*` /
  `BoundedOperation` pattern with enforced bounds and
  `LookupResourceError` guardrails, matching the `math` subpackage.
- [ ] Resolve naming overlap: `math.maximum`/`minimum` (scalar pair),
  `math.compile_maximum`/`compile_minimum` (pair circuit), and root
  `compile_maximum`/`compile_minimum` (array reduction).
- [x] Decide whether `ml.sigmoid` / `ml.tanh` re-exports (compile factories)
  should stay in the ml namespace — kept for backwards compatibility, now
  imported from their real homes and documented as compile factories in
  `ml/__init__.py`.

## Quality gates

- [x] Add a **domain-module acceptance bar** to CONTRIBUTING.md: data
  contract, bounds, `compile_*` helpers, tests, docs — no thin wrappers.
- [x] Enforce lint: ruff adopted (pyproject config + pre-commit hooks +
  enforcing `lint.yml`); the `--exit-zero` pylint workflow was removed.
  Next step: widen the ruff rule set beyond E/F/B once convenient.
- [ ] Meaningful typing: replace blanket `Any` with documented aliases
  (e.g. `EncryptedValue`) and add a mypy CI step.

## Candidate modules (ordered by priority)

Growth rule: deepen capability layers (math/arrays/stats/ml) freely; new
domain modules must clear the acceptance bar in CONTRIBUTING.md.

- [ ] **finance.scoring** — scorecard/credit-score evaluation: weighted sum
  + clamp + LUT score banding (strategic differentiator).
- [ ] **finance.portfolio** — encrypted portfolio value, position-limit
  checks, risk-threshold counts.
- [ ] **stats** (top-level) — promote `ml.stats` and add private-analytics
  primitives: median/percentile (reuse the bitonic sort), top-k,
  histogram/bincount, mode (bincount + argmax).
- [ ] **ml ensembles** — `random_forest_inference`
  (decision_tree_inference x N + majority_votes), small `mlp_inference`
  (matrix_vector_multiply + relu layers); both compose existing pieces.
- [ ] Later, use-case driven: `sets` (PSI primitives: intersection size,
  membership), `text` (bounded-alphabet string equality/search),
  `timeseries` (windowed sums/averages, threshold-crossing counts).
- Out of scope: model training (Concrete-ML's territory), free-form
  floating point, protocol-level crypto/key management.

## Dependencies / upstream tracking

- [ ] Relax the `numpy<2` pin once the weekly `concrete-canary` workflow
  stays green against newer `concrete-python`.
- [ ] Drop the temporary `setuptools<81` dependency when `concrete-python`
  stops importing `pkg_resources` (canary will show when).
- [ ] Reduce hardcoded "Concrete 2.11" mentions (README badge + docs) to a
  single "supported versions" section.

## Documentation

- [x] Versioned API docs site generated from docstrings (MkDocs Material +
  mkdocstrings + mike, deployed to GitHub Pages per release).
- [ ] Write module docstrings/examples for `ml.stats`, `ml.utils`,
  `ml.matrix`, and `finance.transactions` so the generated API pages are
  self-explanatory.
- [ ] **Track Zensical**: MkDocs 1.x is unmaintained and MkDocs 2.0 drops
  the plugin system, so Material for MkDocs is in maintenance mode; its
  team's successor, Zensical (zensical.org), reads mkdocs.yml as-is.
  Migrate once its mkdocstrings + mike (versioning) support is solid —
  until then the current pin (mkdocs-material constrains mkdocs<2) is safe.
