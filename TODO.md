# Roadmap / TODO

Working list for the team. Check items off in the PR that completes them.

## High priority

- [ ] **Real k-NN**: add a `k` parameter to `knn_inference` and combine with
  `majority_votes` (currently 1-NN only).
- [ ] **Logistic regression helper**: `linear_regression_inference` +
  sigmoid/threshold as a single documented ml helper.
- [ ] **Full decision-tree evaluator**: `make_decision_tree` that takes a tree
  structure (thresholds, feature indices, leaf values) and composes
  `decision_tree_node` — not just a single node.

## API consistency

- [ ] Move `ml` (and later `finance`) toward the `make_*` / `compile_*` /
  `BoundedOperation` pattern with enforced bounds and
  `LookupResourceError` guardrails, matching the `math` subpackage.
- [ ] Resolve naming overlap: `math.maximum`/`minimum` (scalar pair),
  `math.compile_maximum`/`compile_minimum` (pair circuit), and root
  `compile_maximum`/`compile_minimum` (array reduction).
- [ ] Decide whether `ml.sigmoid` / `ml.tanh` re-exports (compile factories)
  should stay in the ml namespace or be replaced with traceable activations.

## Quality gates

- [ ] Add a **domain-module acceptance bar** to CONTRIBUTING.md: a domain
  module (like `finance`) must bring its own data contract (e.g. money
  scale), bounds, `compile_*` helpers, and tests — no thin wrappers.
- [ ] Enforce lint: adopt **ruff** (+ pre-commit) and remove `--exit-zero`
  from the pylint workflow (currently the lint job can never fail).
- [ ] Meaningful typing: replace blanket `Any` with documented aliases
  (e.g. `EncryptedValue`) and add a mypy CI step.

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
