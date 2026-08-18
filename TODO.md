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
  + clamp + LUT score banding (strategic differentiator). *(owner: Tolga —
  domain know-how in house; do not pick up without coordinating)*
- [ ] **finance.portfolio** — encrypted portfolio value, position-limit
  checks, risk-threshold counts. *(owner: Tolga)*
- [x] **stats** (top-level) — promote `ml.stats` and add private-analytics
  primitives: median/percentile (reuse the bitonic sort), top-k,
  histogram/bincount, mode (bincount + argmax).
- [x] **ml ensembles** — `random_forest_inference`
  (decision_tree_inference x N + majority_votes), small `mlp_inference`
  (matrix_vector_multiply + relu layers); both compose existing pieces.
- [ ] Later, use-case driven: `sets` (PSI primitives: intersection size,
  membership), `text` (bounded-alphabet string equality/search),
  `timeseries` (windowed sums/averages, threshold-crossing counts).
- Out of scope: model training (Concrete-ML's territory), free-form
  floating point, protocol-level crypto/key management.

## Expansion backlog by layer (full inventory, 2026-08-18 review)

### math

- [x] **`atan2(y, x)`** — two-input scaled LUT (quadrant-aware angle);
  completes the inverse-trig set.
- [x] **`powmod`** — `pow(base, exponent, modulus)` with public base and
  modulus, encrypted exponent (unary LUT; toy-crypto and hashing demos).
- [x] **`copysign(x, y)`** — via abs LUT + comparison, or a two-input LUT.
- [x] **`log(x, base)`** — arbitrary-base parameter on the existing log.
- [x] **`gamma` / `lgamma`** — scaled unary LUTs, same pattern as erf.
- [x] **Saturating arithmetic** — `saturating_add/sub/mul` (op + clamp LUT)
  so pipelines can bound growth without manual clamping.
- [x] **`round_to_multiple(value, step)`** — quantization primitive for
  rescaling pipelines.
- [x] **fixed_point: `modf`** — (integer_part, fractional_part) pair, same
  two-LUT pattern as floor_ceil.
- [x] **fixed_point: `fixed_point_multiply`** — multiply two scaled values
  and rescale the product back (`a * b // scale`); the missing primitive
  for chaining fixed-point math.
- [x] **Clear-side codec helpers** — `encode(value, scale)` /
  `decode(value, scale)` utilities so users stop hand-rolling the scaling
  shown in the README examples.

### math.bits / binary_division

- [x] **Shifts and rotates** — `shift_left/right_bits`,
  `rotate_left/right_bits` (re-indexing, nearly free).
- [x] **`popcount` / `parity`** — bit-count via tournament sum; parity via
  xor-reduce (bit_xor_many exists).
- [x] **`bit_length` / leading zeros** — LUT or mux ladder.
- [x] **`unsigned_compare_bits`** — comparator from bit lists (enables
  sorting/threshold logic in pure bit circuits).
- [x] **Expose remainder from restoring division** — `unsigned_mod_bits`
  (the restoring circuit already computes it internally).
- [ ] **Signed division** — sign-magnitude wrapper around the unsigned
  restoring divider.
- [ ] **Barrel shifter by encrypted amount** — mux layers of bit_select
  (advanced; enables variable shifts).

### arrays

- [x] **`array_index(array, index)`** — oblivious read: `sum(equal(i, idx)
  * arr[i])`; currently hand-rolled inside knn_inference — promote to a
  named primitive and reuse it there.
- [x] **`array_set(array, index, value)`** — oblivious write via select.
- [x] **`array_index_of(array, value)`** — first index of a value
  (equal flags + argmax with tie_break="first").
- [x] **`array_cumsum`** — prefix sums (native adds).
- [x] **`array_top_k`** — k-round argmax with masking (same pattern as
  knn's iterated argmin; factor the shared loop out).
- [x] **`array_reverse`, `array_concat`** — trivial clear-side utilities
  for completeness.

### stats (new top-level module — see candidate list)

- [x] median / percentile via the existing bitonic sort.
- [x] histogram / bincount (equal-flag sums over a bounded value range);
  mode = bincount + argmax.
- [x] std deviation = isqrt(variance) and covariance. (Scaled correlation
  still open — needs a fixed-point divide design.)
- [x] z-score normalization with public mean/scale (affine transform).
- [x] Absorb `ml.stats` into this module and re-export for compatibility.

### ml

- [x] **`random_forest_inference`** — decision_tree_inference over a list
  of trees + majority_votes (pure composition, cheap win).
- [x] **`mlp_inference`** — small multilayer perceptron:
  matrix_vector_multiply + relu per layer, public weights.
- [x] **`nearest_centroid_inference`** — k-means assignment step: argmin
  of distances to public centroids (knn machinery with public rows).
- [x] **`naive_bayes_inference`** — sum of public log-probability tables
  indexed by encrypted features (LUT per feature + argmax).
- [x] **`argmax_inference`** — multi-class head: argmax over class scores
  (replaces softmax for classification).
- [x] **precision / recall / f1** — from the existing confusion counts,
  returned as scaled percentages like accuracy_score.
- [ ] **cosine similarity (squared/scaled)** — dot² scaled by norms via
  fixed-point divide.
- [ ] Migrate ml to the make_/compile_ bounds pattern (tracked above).
  Decision trees already rebuilt on math.select.

### ml — second wave (2026-08-18 review, ordered by value)

- [ ] **`gbdt_inference`** — boosted trees: integer score leaves summed
  across trees + threshold (decision_tree_inference + add +
  threshold_activation). The de-facto credit-scoring model; direct synergy
  with finance.scoring.
- [ ] **sklearn bridge (clear-side)** — `from_sklearn_linear(model, scale)`
  and `from_sklearn_tree(model, scale)` converters into our integer
  weight-list / dict-tree formats; sklearn as an optional import inside
  the functions. Kills the manual scaling ritual in the README example.
- [ ] **`linear_multiclass_inference`** — per-class W.x+b scores +
  argmax_inference (pure composition).
- [ ] **`logistic_regression_proba`** — scaled probability output via a
  sigmoid LUT over the known score bounds.
- [ ] **`svm_rbf_inference`** — kernel trick under FHE: K = exp(-gamma*d^2)
  is a single LUT over the squared-distance range; weighted sum of kernels
  + threshold.
- [ ] **`bin_feature(value, bin_edges)`** — ordinal bucketing as a sum of
  greater_equal over public edges (scorecard building block).
- [ ] **Top-k similarity retrieval** — query x corpus dot products +
  make_top_k (private-search demo in one composition).
- [ ] **Multi-class confusion matrix** and **scaled R² score** — existing
  counter + floor_divide patterns.
- [ ] **Weighted majority vote** — majority_votes with public weights
  (also a GBDT sub-piece).
- Deliberately out: ROC/AUC under encryption (sort cost unjustifiable —
  compute post-decryption), softmax (argmax suffices), deep networks
  (Concrete-ML's territory).

### Infrastructure / DX (non-finance)

- [ ] **`deploy` helpers** — thin wrappers over Concrete's client/server
  split (fhe.Client / fhe.Server, circuit save/load) with a worked
  example; turns the toolkit from notebook-ware into deployable services.
- [ ] **examples/ growth** — one runnable example per subpackage (only
  quickstart.py exists today); wire them into CI as smoke tests.
- [ ] **benchmarks/** — script measuring compile/keygen/run times per
  operation and bounds size; feeds real numbers into
  docs/bounds-and-costs.md.
- [ ] **Batching guide** — document tensor-shaped inputs (compile once,
  run element-wise on arrays) with an example.

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
