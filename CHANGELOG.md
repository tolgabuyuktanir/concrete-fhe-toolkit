# Changelog

## Unreleased

### Added

- Model serialization (`ml.serialization`): `save_model` / `load_model`
  persist every parametric model class (trees, weights, centroids,
  tables) as portable JSON; circuits and keys are never serialized —
  recompile after loading.

## 0.5.0 - 2026-08-26

The machine-learning release: a sklearn-style class API, encrypted
training for four model families, single-circuit pipelines, and
differential-privacy helpers for released aggregates.

### Added

- sklearn-style class API (`ml.classes`): `FHEModel` base with
  compile/predict lifecycle and eleven ready models —
  `FHELogisticRegression`, `FHELinearRegression`, `FHEDecisionTree`,
  `FHERandomForest`, `FHEXGBoost`, `FHESVM`, `FHEKNN`, `FHENaiveBayes`,
  `FHEMLP`, `FHEPCA`, `FHECNN`.
- New inference and matrix helpers: `pca_inference`, `cnn_inference`,
  `svm_inference`, `xgboost_inference`, `max/avg_pooling_2d`,
  `auto_quantizer`, `normalize_array`, `matrix_exp`, `covariance_matrix`,
  `matrix_elementwise_multiply`, `matrix_flatten`, `tensor_flatten`,
  `l1_norm`; plus a docstring-example pass across the ml modules.
- Encrypted Naive Bayes training: `naive_bayes_training` and
  `FHENaiveBayesTrainer` (Laplace smoothing and log-prob tables built
  clear-side from decrypted counts).
- Encrypted training via the aggregate-decrypt pattern (`ml.trainers`):
  `FHETrainer` base (with a `simulate` mode for fast prototyping),
  `FHELinearRegressionTrainer` (sufficient statistics + clear-side normal
  equations), `FHEDecisionTreeTrainer` (hybrid level-wise Gini counting),
  and `FHEKMeansTrainer` with the `FHEKMeans` model class.
- sklearn-style task namespaces: `ml.classification`, `ml.regression`,
  `ml.clustering` with `*Classifier`/`*Regressor` aliases; flat names
  remain for backwards compatibility.
- Metrics: `r2_score` (integer percent R²) and clustering `inertia`.
- `FHEPipeline`: preprocessing transformers and a model chained into a
  single compiled circuit, plus `ml.preprocessing` transformers
  (`FHEBinner` scorecard binning, `FHEStandardScaler`, `FHEMinMaxScaler`)
  and the `bin_feature` primitive.
- Top-level `privacy` subpackage: `laplace_mechanism`,
  `gaussian_mechanism`, and `dp_release` add calibrated differential-
  privacy noise to decrypted trainer aggregates (FHE + DP pattern).
- `FHEModel` gains `simulate`, `predict_many`, and `simulate_many`:
  one compile and one key set serve any number of predictions.

## 0.4.0 - 2026-08-18

Big capability release: Python-math and C-math parity for the math
subpackage, a new top-level stats subpackage, oblivious array primitives,
and a second wave of ML inference models.

### Added

- Math, inverse trigonometry and conversions: `asin`, `acos`, `atan` (scaled,
  with `invalid_result` domain handling for `asin`/`acos`), `cbrt`,
  `degrees`, `radians`.
- Math, number theory: `totient` (Euler's phi), `next_prime`,
  `mod_inverse` (with explicit invalid behavior), and rounded integer
  `hypot`.
- Math, basics: `select(control, when_true, when_false)` for oblivious
  branching over arbitrary bounded integers (plus `compile_select`) and
  `abs_diff` (|left - right| via a difference-domain lookup).
- Math, more coverage: `atan2`, `gamma`, `lgamma`, arbitrary-base `log`,
  `powmod` (public base/modulus, encrypted exponent), `copysign`,
  saturating `add`/`subtract`/`multiply`, `round_to_multiple`, `modf`,
  `fixed_point_multiply` (rescaled product), and clear-side
  `encode_fixed_point` / `decode_fixed_point` codecs.
- Python math-module parity: summation/product group `fsum`, `prod`,
  `sumprod`, rounded Euclidean `dist` over coordinate lists,
  encrypted-base `pow` (bounded two-input lookup), and clear-side
  constants `pi`, `e`, `tau` (pair with `encode_fixed_point`).
- C math.h parity: `asinh`, `acosh`, `atanh`, `exp2`, `fdim`, `fma`,
  IEEE-style `remainder` (ties-to-even), `ilogb`, `ldexp`/`scalbn`
  (public power-of-two scaling), plus aliases `tgamma`, `fabs`, `fmax`,
  `fmin`. NaN/nextafter-style float machinery is intentionally out of
  scope for integer FHE.
- Bit level: `shift_left/right_bits` (logical/arithmetic), rotates,
  `popcount_bits`, `parity_bits`, `bit_length_bits`,
  `unsigned_compare_bits`, and the restoring divider's remainder exposed as
  `unsigned_mod_bits` / `make_unsigned_mod` / `compile_unsigned_mod`.
- Arrays: oblivious `array_index` / `array_set` / `array_index_of`,
  `array_cumsum`, `array_reverse`, `array_concat`, and `make_top_k` /
  `compile_top_k` (k largest or smallest values via masked arg-extreme
  rounds).
- New top-level `concrete_fhe_toolkit.stats` subpackage: mean, variance,
  std (isqrt), covariance, min/max/range, count-greater, median and
  percentile (via the bitonic sort), histogram/bincount, mode, and z-score
  normalization. `ml.stats` re-exports for backwards compatibility.
- ML: `random_forest_inference`, `mlp_inference` (ReLU hidden layers),
  `nearest_centroid_inference` (k-means assignment), `naive_bayes_inference`
  (categorical, public log-prob tables), `argmax_inference` (multi-class
  head), and `precision_score` / `recall_score` / `f1_score`. Decision
  trees now build on `math.select`.
- `knn_inference` now supports real k-NN via a `k` parameter (iterated
  argmin with distance masking, majority vote; binary labels for `k > 1`).
- `logistic_regression_inference`: binary classification from the linear
  score with an optional public threshold.
- `decision_tree_inference`: oblivious evaluation of a full public decision
  tree (dict nodes, integer leaves) over encrypted features.
- Ruff configuration in `pyproject.toml`, pre-commit hooks, and an
  enforcing `lint.yml` workflow (the `--exit-zero` pylint workflow was
  removed).
- CONTRIBUTING.md: domain-module acceptance bar and dependency-direction
  rule.

## 0.3.0 - 2026-08-18

Project home is now
[tolgabuyuktanir/concrete-fhe-toolkit](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit).

### Added

- Add the `concrete_fhe_toolkit.ml` subpackage: encrypted metrics
  (accuracy, confusion matrix, MSE/MAE, distances, hinge loss), matrix and
  vector algebra, model inference helpers (linear regression, decision tree
  node, 1-NN, majority voting), activations (ReLU, leaky ReLU, unit step,
  threshold), array statistics, and preprocessing utilities.
- Add the `concrete_fhe_toolkit.finance` subpackage with a documented
  `RATE_SCALE` money convention: `apply_rate`, `calculate_tax`, `discount`,
  `simple_interest`, `transfer`, and `return_actual_value`.
- Add array utilities: `array_add`, `array_sub`, `array_multiply`,
  `array_scale`, `array_sum`, `array_pad`, `array_slice`, `array_contains`,
  `array_count`, `array_all_equal`.
- Add bit helpers: `bit_op_many`, `bit_and_many`, `bit_or_many`,
  `bit_xor_many`, `multiply_bits`, `twos_complement_add_bits` (carry-lookahead).
- Add scalar helpers: unary `sign`, `compile_sign`, `is_zero`,
  `compile_is_zero`, `cube`, `compile_cube`, pairwise `maximum`/`minimum`
  with `compile_maximum`/`compile_minimum`.
- Add an internal `_compat` facade: all Concrete imports flow through one
  module that verifies the required `concrete-python` API at import time.
- Add a weekly "Concrete canary" CI workflow that tests against the newest
  (including pre-release) `concrete-python` for early upstream warnings.
- Add cleartext and compilation test suites for the ml and finance
  subpackages.

### Changed

- `sign` is now unary (`sign(x)` returns -1/0/1); the two-input comparison
  remains available as `compare`. Tests and docs updated accordingly.
- `unit_step` now returns the plain Heaviside step (0 for negatives,
  otherwise 1) instead of a doubled 0..2 encoding.
- `knn_inference` takes an explicit `max_distance` bound for its argmin
  reduction instead of silently using the default 0..15 range.
- `leaky_relu` truncates the scaled negative branch toward zero and
  validates that `alpha` is the reciprocal of a positive integer;
  `compile_leaky_relu` accepts `alpha`.
- The publish workflow now uses the protected `pypi` GitHub environment.

### Fixed

- `decision_tree_node` now selects branches arithmetically; the previous
  `bit_select` implementation was only valid for 0/1 branch values.
- `compile_threshold_activation` now declares both encrypted inputs and
  compiles successfully.
- `finance.apply_rate` computes exact integer rates (for example 0.29 no
  longer truncates to 28 due to binary floating-point error).
- Removed empty `finance/potfolio.py` and `finance/scoring.py` stubs.
- Fixed swapped or incorrect docstrings on `compile_relu`,
  `compile_leaky_relu`, `compile_maximum`, `compile_minimum`,
  `compile_is_zero`, and `compile_cube`.
- Fixed the gated FHE smoke test calling the sign circuit with two inputs.
- Stabilized the flaky bit-level division simulation test by compiling with
  a tight `p_error` so simulated TFHE noise cannot flip quotient bits.

## 0.2.2 - 2026-07-23

- Add the temporary `setuptools<81` runtime compatibility dependency required
  by `concrete-python`, which still imports `pkg_resources` during startup.
- Restore clean-environment CI compatibility with current Python and pip
  releases.

## 0.2.1 - 2026-07-23

- Promote the package development status from Beta to Production/Stable.

## 0.2.0 - 2026-06-23

- Add the `concrete_fhe_toolkit.math` subpackage for Python-math-style bounded
  encrypted integer and fixed-point helpers.
- Add native arithmetic, comparison, scalar multiplication, absolute value,
  clamp, modulo, divmod, and closeness helpers.
- Add combinatorics helpers for factorial, Fibonacci, public-base powers,
  combinations, and permutations.
- Add number-theory helpers for GCD, LCM, coprime/divisibility predicates,
  integer square root, parity, and primality.
- Add fixed-point floor, ceil, truncation, rounding, rescaling, trigonometric,
  logarithmic, square-root, error-function, hyperbolic, and sigmoid helpers.
- Add friendly bounded operation objects such as `math.gcd(...)` and
  `math.sin(...)` that compile by default while exposing `.make(...)` and
  `.compile(...)` for advanced usage.
- Add bit-level LUT primitives, two's-complement helpers, unsigned restoring
  division, and fixed-point binary division helpers.
- Add dedicated documentation pages for bounds/costs, API usage, and
  pre-release testing checks.
- Add large lookup cost warnings and `allow_large_lookup=True` opt-in guards.
- Add exhaustive cleartext tests, compiler-simulation tests, and opt-in real
  encrypted math smoke tests for the new math API.

## 0.1.3 - 2026-06-18

- Promote the package development status from Alpha to Beta.

## 0.1.2 - 2026-06-16

- Add GitHub repository and example notebook links to package metadata.
- Clarify that README bounds are public application limits, not values learned
  from encrypted arrays.

## 0.1.1 - 2026-06-15

- Remove references to unrelated encryption libraries from package metadata.
- Add contributor and company acknowledgements.

## 0.1.0 - 2026-06-09

- Add bounded compare-swap, sorting, min/max, and argmin/argmax operations.
- Add safe multivariate floor division and division by an encrypted product.
- Add compiler helpers with boundary-complete inputsets.
- Add cleartext, compiler simulation, and opt-in encrypted smoke tests.
