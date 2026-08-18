<div align="center">

# 🔐 concrete-fhe-toolkit

**Bounded math helpers for compiling common [Zama Concrete](https://docs.zama.ai/concrete) FHE circuits.**

Compare, sort, and run real math — GCD, factorial, `sin`, `sqrt`, division — directly
on **encrypted** integers, without ever decrypting them.

<!-- Badges -->
[![PyPI version](https://img.shields.io/pypi/v/concrete-fhe-toolkit?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/concrete-fhe-toolkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/concrete-fhe-toolkit?logo=python&logoColor=white)](https://pypi.org/project/concrete-fhe-toolkit/)
[![CI](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Concrete](https://img.shields.io/badge/Concrete-2.11-8A2BE2)](https://docs.zama.ai/concrete)
[![Status](https://img.shields.io/pypi/status/concrete-fhe-toolkit)](https://pypi.org/project/concrete-fhe-toolkit/)
[![Downloads](https://img.shields.io/pypi/dm/concrete-fhe-toolkit?color=informational)](https://pypi.org/project/concrete-fhe-toolkit/)

[Installation](#installation) ·
[Quick Start](#quick-start) ·
[Documentation](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit/tree/main/docs) ·
[Examples](#examples) ·
[Public API](#public-api)

</div>

---

> [!NOTE]
> `concrete-fhe-toolkit` is an **unofficial** helper package for
> [Zama Concrete](https://docs.zama.ai/concrete). It provides reusable circuit
> builders for common bounded math operations on encrypted Concrete inputs.
> This project is not affiliated with or endorsed by Zama.

## ✨ Highlights

- **Privacy-Preserving Machine Learning** — built-in support for encrypted Linear Regression, KNN, Decision Trees, and ML metrics (accuracy, confusion matrix).
- **Batteries-included math** — arithmetic, comparisons, GCD/LCM, factorial,
  primality, `isqrt`, and combinatorics on encrypted integers.
- **Fixed-point transcendentals** — `sin`, `cos`, `log`, `sqrt`, `erf`, `tanh`,
  and `sigmoid` via lookup tables over a declared domain.
- **Array circuits** — bitonic sort, min/max, and argmin/argmax with
  deterministic tie handling.
- **Array utilities** — native FHE support for standard array manipulations including element-wise math, slicing, scaling, and padding.
- **Real division** — bounded floor division and bit-level restoring division
  that scales with bit width instead of a giant lookup table.
- **Friendly, three-layer API** — call `fhe_math.gcd(...)` to compile,
  `.make(...)` to compose, or `.compile(...)` to be explicit.
- **Cost guardrails** — static lookup-size estimates warn (or block) before you
  accidentally compile an enormous, memory-hungry circuit.

## 📑 Table of Contents

- [Installation](#installation)
- [Important Concept](#important-concept)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Public API](#public-api)
- [Examples](#examples)
- [Bounds and Limitations](#bounds-and-limitations)
- [Notebook provenance and examples](#notebook-provenance-and-examples)
- [License and Concrete Terms](#license-and-concrete-terms)
- [Author and Contributors](#author-and-contributors)
- [Thanks](#thanks)
- [References](#references)

## What it does

The package focuses on explicit, bounded FHE circuits:

- compare two encrypted integers
- add, subtract, multiply, negate, square, and compare encrypted integers
- compute encrypted integer math such as `abs`, clamp, modulo, GCD, LCM,
  `isqrt`, primality, factorial, Fibonacci, combinations, and permutations
- approximate fixed-point math such as floor, ceil, round, rescale, `sin`,
  `cos`, `log`, `sqrt`, `erf`, `tanh`, and sigmoid with lookup tables
- compare-swap two encrypted integers
- sort encrypted integer arrays
- find encrypted-array min/max values
- find encrypted-array argmin/argmax indices
- perform bounded encrypted floor division with a table lookup
- perform bounded `numerator // (left * right)`
- run machine learning models (Linear Regression, KNN) and metrics directly on encrypted data
- perform matrix and array algebra (dot products, matrix multiplication, array addition) on encrypted tensors

## Installation

```bash
pip install concrete-fhe-toolkit
```

To install the latest source directly from GitHub:

```bash
pip install git+https://github.com/tolgabuyuktanir/concrete-fhe-toolkit.git
```

Or clone the repository and install it locally:

```bash
git clone https://github.com/tolgabuyuktanir/concrete-fhe-toolkit.git
cd concrete-fhe-toolkit
pip install .
```

> [!IMPORTANT]
> Concrete 2.11 supports **Python 3.9 through 3.12 on Linux and macOS** only.
> There is no Windows wheel for `concrete-python`; on Windows use WSL2 or a
> Linux container.

### Development setup

For contributing or running the test suite, install the pinned dependencies and
the package in editable mode:

```bash
git clone https://github.com/tolgabuyuktanir/concrete-fhe-toolkit.git
cd concrete-fhe-toolkit
python -m pip install -r requirements-dev.txt
python -m pip install -e ".[dev]"
python -m pytest -q
```

Two convenience requirements files are provided:

| File | Contents |
| --- | --- |
| [`requirements.txt`](requirements.txt) | Runtime dependencies (`concrete-python`, `numpy`) |
| [`requirements-dev.txt`](requirements-dev.txt) | Runtime plus `build`, `pytest`, and `twine` |

## Important Concept

Most users can start with the plain operation names in
`concrete_fhe_toolkit.math`. Calling one of these operations compiles a Concrete
circuit by default:

```python
from concrete_fhe_toolkit import math as fhe_math

gcd = fhe_math.gcd(min_value=0, max_value=8)
print(gcd.encrypt_run_decrypt(6, 4))
# 2
```

These operation objects also expose the lower-level builders:

- `fhe_math.gcd.compile(...)` compiles a ready-to-run encrypted circuit.
- `fhe_math.gcd.make(...)` returns a traceable function that can be composed
  into a larger Concrete program.

The explicit `compile_*` and `make_*` names remain available for advanced users
and backwards compatibility. For example, `fhe_math.compile_gcd(...)` is the
same compiler used by `fhe_math.gcd(...)`, and `fhe_math.make_gcd(...)` is the
same builder used by `fhe_math.gcd.make(...)`.

Use `make_*` only when you are manually composing several operations into one
larger circuit. For ordinary one-operation usage, prefer the plain operation
name or the explicit `compile_*` helper.

All operations use integer inputs. You must choose fixed input bounds when
compiling a circuit, and runtime inputs must stay inside those bounds.

The bounds are not the hidden minimum and maximum of a particular encrypted
array. They are public, application-level limits that you know before
compilation. For example, if your encrypted scores are always percentages, use
`min_value=0` and `max_value=100`. If your private balances are stored in a
range from `-1_000` to `1_000`, use those as the bounds. Wider bounds are more
flexible, but they usually make the compiled circuit more expensive.

The `concrete_fhe_toolkit.math` subpackage is intentionally closer to Python's
`math` module, but every operation still needs explicit bounds because FHE
circuits are compiled ahead of time. Fixed-point helpers accept encrypted
integers that represent scaled real values. For example, with
`input_scale=100`, the encrypted integer `314` represents `3.14`.

Common bound parameters:

| Parameter style | Meaning |
| --- | --- |
| `min_value`, `max_value` | Inclusive input range for one or two integer inputs |
| `min_input`, `max_input` | Inclusive input range for scaled fixed-point values |
| `max_n` | Encrypted index range `0..max_n`, used by factorial/Fibonacci/combinatorics |
| `numerator_width`, `denominator_width` | Unsigned scalar bit widths for bit-level division |
| `fractional_bits` | Number of binary fractional bits in fixed-point division output |

## Quick Start

```python
import numpy as np

from concrete_fhe_toolkit import compile_argmin, compile_sort

values = np.array([12, 3, 7, 1, 15, 0, 4, 9], dtype=np.int64)

sort_circuit = compile_sort(size=8, min_value=0, max_value=100)
print(sort_circuit.encrypt_run_decrypt(values))
# [ 0  1  3  4  7  9 12 15]

argmin_circuit = compile_argmin(size=8, min_value=0, max_value=100)
print(argmin_circuit.encrypt_run_decrypt(values))
# 5
```

Math helpers live under `concrete_fhe_toolkit.math`:

```python
from concrete_fhe_toolkit import math as fhe_math

gcd_circuit = fhe_math.gcd(min_value=0, max_value=8)
print(gcd_circuit.encrypt_run_decrypt(6, 4))
# 2

sin_circuit = fhe_math.sin(
    min_input=0,
    max_input=90,
    input_scale=1,
    output_scale=100,
    angle_unit="degrees",
)
print(sin_circuit.encrypt_run_decrypt(30))
# 50  (represents 0.50)
```

## Documentation

The README is the shared GitHub and PyPI landing page. It gives the core usage
model, quick examples, and the public API overview.

For the fuller guide, use the documentation site — versioned per release and
generated from the package docstrings:

- [Documentation site (latest)](https://tolgabuyuktanir.github.io/concrete-fhe-toolkit/)
- [Documentation index on GitHub](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit/tree/main/docs)
- [Bounds and costs](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit/blob/main/docs/bounds-and-costs.md)
- [API reference](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit/blob/main/docs/api-reference.md)
- [Maintainer testing and release checks](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit/blob/main/docs/testing-and-release.md)

The short version: use `fhe_math.gcd(...)` or another friendly operation name
for a ready-to-run circuit, use `.make(...)` when composing a larger Concrete
program, and choose public bounds that cover all runtime inputs.

## Public API

Scalar arithmetic:

- `compare(x, y)`
- `compile_compare(min_value=-15, max_value=15, configuration=None)`
- `sign(x)`
- `compile_sign(min_value=-15, max_value=15, configuration=None)`
- `make_floor_divide(zero_result=0)`
- `compile_floor_divide(max_numerator, max_denominator, zero_result=0, configuration=None)`
- `make_floor_divide_by_product(zero_result=0)`
- `compile_floor_divide_by_product(max_numerator, max_left, max_right, zero_result=0, configuration=None)`

Array operations:

- `make_compare_swap(min_value=0, max_value=15)`
- `compile_compare_swap(min_value=0, max_value=15, configuration=None)`
- `make_sort(size, min_value=0, max_value=15, descending=False)`
- `compile_sort(size, min_value=0, max_value=15, descending=False, configuration=None)`
- `make_minimum(size, min_value=0, max_value=15)`
- `compile_minimum(size, min_value=0, max_value=15, configuration=None)`
- `make_maximum(size, min_value=0, max_value=15)`
- `compile_maximum(size, min_value=0, max_value=15, configuration=None)`
- `make_argmin(size, min_value=0, max_value=15, tie_break="first")`
- `compile_argmin(size, min_value=0, max_value=15, tie_break="first", configuration=None)`
- `make_argmax(size, min_value=0, max_value=15, tie_break="first")`
- `compile_argmax(size, min_value=0, max_value=15, tie_break="first", configuration=None)`
- Array math: `array_add`, `array_sub`, `array_multiply`, `array_scale`, `array_sum`
- Array utilities: `array_all_equal`, `array_contains`, `array_count`, `array_pad`, `array_slice`

Machine Learning (ML) subpackage:

```python
from concrete_fhe_toolkit import ml
```

- **Metrics & Data**: `accuracy_score`, `confusion_matrix`, `mean_squared_error`, `mean_absolute_error`, `one_hot_encode`, `binarize`
- **Models**: `linear_regression_inference`, `decision_tree_node`, `knn_inference`, `majority_votes`
- **Matrix Operations**: `matrix_multiply`, `matrix_vector_multiply`, `matrix_add`, `matrix_subtract`, `matrix_transpose`, `dot_product`

Math subpackage:

```python
from concrete_fhe_toolkit import math as fhe_math
```

Friendly operation objects compile by default and expose `.make(...)` and
`.compile(...)` for explicit control:

- `absolute`, `clamp`, `modulo`, `divmod`, `is_close`
- `factorial`, `fibonacci`, `power`, `comb`, `perm`
- `gcd`, `lcm`, `is_coprime`, `is_divisible`, `isqrt`, `is_even`, `is_odd`,
  `is_prime`
- `floor`, `ceil`, `trunc`, `round`, `floor_ceil`, `rescale`
- `sin`, `cos`, `tan`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`,
  `sqrt`, `erf`, `erfc`, `tanh`, `sinh`, `cosh`, `sigmoid`
- `unsigned_floor_divide`, `fixed_point_divide`

Basic integer operations:

- native arithmetic: `add`, `subtract`, `multiply`, `negate`, `square`
- comparisons: `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`
- compilers: `compile_add`, `compile_subtract`, `compile_multiply`,
  `compile_negate`, `compile_square`, `compile_scalar_multiply`
- lookup helpers: `make_absolute`, `compile_absolute`, `make_clamp`,
  `compile_clamp`, `make_modulo`, `compile_modulo`, `make_divmod`,
  `compile_divmod`, `make_is_close`, `compile_is_close`

Combinatorics:

- `make_factorial`, `compile_factorial`
- `make_fibonacci`, `compile_fibonacci`
- `make_power`, `compile_power`
- `make_comb`, `compile_comb`
- `make_perm`, `compile_perm`

Number theory:

- `make_gcd`, `compile_gcd`
- `make_lcm`, `compile_lcm`
- `make_is_coprime`, `compile_is_coprime`
- `make_is_divisible`, `compile_is_divisible`
- `make_isqrt`, `compile_isqrt`
- `make_is_even`, `compile_is_even`
- `make_is_odd`, `compile_is_odd`
- `make_is_prime`, `compile_is_prime`

Fixed-point helpers:

- `make_floor`, `compile_floor`
- `make_ceil`, `compile_ceil`
- `make_trunc`, `compile_trunc`
- `make_round`, `compile_round`
- `make_floor_ceil`, `compile_floor_ceil`
- `make_rescale`, `compile_rescale`

Special fixed-point helpers:

- trigonometric: `make_sin`, `compile_sin`, `make_cos`, `compile_cos`,
  `make_tan`, `compile_tan`
- exponential/logarithmic: `make_exp`, `compile_exp`, `make_expm1`,
  `compile_expm1`, `make_log`, `compile_log`, `make_log1p`,
  `compile_log1p`, `make_log2`, `compile_log2`, `make_log10`,
  `compile_log10`
- roots/special activations: `make_sqrt`, `compile_sqrt`, `make_erf`,
  `compile_erf`, `make_erfc`, `compile_erfc`, `make_tanh`, `compile_tanh`,
  `make_sinh`, `compile_sinh`, `make_cosh`, `compile_cosh`,
  `make_sigmoid`, `compile_sigmoid`

Bit-level arithmetic helpers:

- bit primitives: `bit_not`, `bit_and`, `bit_or`, `bit_xor`, `bit_select`,
  `full_adder_bit`, `full_subtractor_bit`
- conversion helpers: `integer_to_bits`, `bits_to_unsigned`,
  `unsigned_to_bits`, `twos_complement_bits`
- signed bit helpers: `sign_magnitude_to_twos_complement_bits`,
  `twos_complement_add_bits`, `twos_complement_multiply_by_constant_bits`
- binary division: `unsigned_divide_bits`, `fixed_point_divide_bits`,
  `make_unsigned_floor_divide`, `compile_unsigned_floor_divide`,
  `make_fixed_point_divide`, `compile_fixed_point_divide`

## Examples

### Privacy-Preserving Breast Cancer Diagnosis (ML Subpackage)

The `concrete_fhe_toolkit.ml` subpackage abstracts away complex FHE cryptography, allowing you to run standard ML models (Linear Regression, KNN, Decision Trees) and metrics directly on encrypted data effortlessly.

In this example, we predict Breast Cancer securely. The server evaluates the Logistic Regression model on encrypted patient features without ever seeing the raw medical data!

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from concrete import fhe

from concrete_fhe_toolkit.ml import (
    linear_regression_inference, 
    threshold_activation,
    accuracy_score,
    confusion_matrix
)

# 1. Train a standard model (Cleartext)
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# FHE precision requires scaling. Standardize features to keep them small.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_scaled, y_train)

# Scale weights to integers to avoid 16-bit FHE lookup limit
scale = 10
weights = np.round(clf.coef_[0] * scale).astype(int).tolist()
bias = int(np.round(clf.intercept_[0] * scale))
X_test_int = np.round(X_test_scaled * scale).astype(int).tolist()

# 2. Define the FHE Circuit
@fhe.compiler({"features": "encrypted"})
def encrypted_diagnosis(features):
    # Toolkit handles the encrypted dot product natively
    raw_pred = linear_regression_inference(weights, bias, features)
    # Toolkit safely thresholds the encrypted output (val >= 0)
    return threshold_activation(raw_pred, 0)

# 3. Compile and Run Encrypted Inference
inputset = [X_test_int[i] for i in range(15)]
circuit = encrypted_diagnosis.compile(inputset)

encrypted_preds = []
true_labels = y_test.tolist()[:10]

for i in range(10):
    # Encrypt -> Predict (Cloud) -> Decrypt
    prediction = circuit.encrypt_run_decrypt(X_test_int[i])
    encrypted_preds.append(prediction)

# 4. Evaluate using toolkit metrics
acc = accuracy_score(encrypted_preds, true_labels)
cm = confusion_matrix(encrypted_preds, true_labels)

print(f"Encrypted Diagnosis Accuracy: %{acc}")
print(f"Confusion Matrix [[TN, FP], [FN, TP]]: \n{cm}")
```

### Compare and Sign

`compile_compare` returns `1` when `x > y`, `0` when `x == y`, and `-1` when `x < y`.
`compile_sign` does the same but explicitly compares `x` against `0`.

```python
from concrete_fhe_toolkit import compile_compare, compile_sign

circuit_comp = compile_compare(min_value=-20, max_value=20)
print(circuit_comp.encrypt_run_decrypt(15, 3))
# 1

print(circuit_comp.encrypt_run_decrypt(7, 7))
# 0

circuit_sign = compile_sign(min_value=-20, max_value=20)
print(circuit_sign.encrypt_run_decrypt(-15))
# -1
```

### Compare-Swap

`compile_compare_swap` returns two values in ascending order.

```python
from concrete_fhe_toolkit import compile_compare_swap

circuit = compile_compare_swap(min_value=0, max_value=100)

print(circuit.encrypt_run_decrypt(12, 3))
# (3, 12)
```

### Sort

`compile_sort` sorts a fixed-size encrypted array. The size must be a power of
two.

```python
import numpy as np

from concrete_fhe_toolkit import compile_sort

values = np.array([12, 3, 7, 1, 15, 0, 4, 9], dtype=np.int64)

ascending = compile_sort(size=8, min_value=0, max_value=100)
print(ascending.encrypt_run_decrypt(values))
# [ 0  1  3  4  7  9 12 15]

descending = compile_sort(size=8, min_value=0, max_value=100, descending=True)
print(descending.encrypt_run_decrypt(values))
# [15 12  9  7  4  3  1  0]
```

### Minimum and Maximum

`compile_minimum` and `compile_maximum` return the smallest or largest value in
a fixed-size encrypted array.

```python
import numpy as np

from concrete_fhe_toolkit import compile_maximum, compile_minimum

values = np.array([12, 5, 7, 2, 15, 9, 4, 14], dtype=np.int64)

minimum = compile_minimum(size=8, min_value=0, max_value=100)
print(minimum.encrypt_run_decrypt(values))
# 2

maximum = compile_maximum(size=8, min_value=0, max_value=100)
print(maximum.encrypt_run_decrypt(values))
# 15
```

### Argmin and Argmax

`compile_argmin` and `compile_argmax` return the index of the smallest or
largest value. By default, ties return the first matching index.

```python
import numpy as np

from concrete_fhe_toolkit import compile_argmax, compile_argmin

values = np.array([12, 5, 7, 1, 15, 9, 4, 14], dtype=np.int64)

argmin = compile_argmin(size=8, min_value=0, max_value=100)
print(argmin.encrypt_run_decrypt(values))
# 3

argmax = compile_argmax(size=8, min_value=0, max_value=100)
print(argmax.encrypt_run_decrypt(values))
# 4
```

Tie handling is explicit:

```python
import numpy as np

from concrete_fhe_toolkit import compile_argmin

values = np.array([4, 1, 1, 3], dtype=np.int64)

first = compile_argmin(size=4, min_value=0, max_value=10, tie_break="first")
print(first.encrypt_run_decrypt(values))
# 1

last = compile_argmin(size=4, min_value=0, max_value=10, tie_break="last")
print(last.encrypt_run_decrypt(values))
# 2
```

### Array Math and Utilities

The package provides native FHE support for standard array manipulation and element-wise math. 

```python
import numpy as np
from concrete_fhe_toolkit import array_add, array_sum, array_scale

encrypted_array1 = np.array([1, 2, 3])
encrypted_array2 = np.array([10, 20, 30])

# Element-wise addition
added = array_add(encrypted_array1, encrypted_array2)
# [11, 22, 33]

# Array summation
total = array_sum(encrypted_array1)
# 6

# Scalar multiplication
scaled = array_scale(encrypted_array1, 5)
# [5, 10, 15]
```

### Floor Division

Direct `x // y` does not compile when both values are encrypted in Concrete
2.11. This package implements bounded floor division with
`fhe.multivariate`.

```python
from concrete_fhe_toolkit import compile_floor_divide

circuit = compile_floor_divide(
    max_numerator=100,
    max_denominator=10,
    zero_result=-1,
)

print(circuit.encrypt_run_decrypt(15, 3))
# 5

print(circuit.encrypt_run_decrypt(15, 0))
# -1
```

`zero_result` is returned when the encrypted denominator is zero.

### Division by an Encrypted Product

`compile_floor_divide_by_product` computes
`numerator // (left * right)`.

```python
from concrete_fhe_toolkit import compile_floor_divide_by_product

circuit = compile_floor_divide_by_product(
    max_numerator=100,
    max_left=10,
    max_right=10,
    zero_result=-1,
)

print(circuit.encrypt_run_decrypt(20, 2, 5))
# 2

print(circuit.encrypt_run_decrypt(20, 0, 5))
# -1
```

### Math Subpackage

Use `concrete_fhe_toolkit.math` for Python-math-style helpers. Some compile to
native Concrete arithmetic, while functions such as GCD, factorial, prime
testing, and fixed-point `sin` use lookup tables over the declared domain.

```python
from concrete_fhe_toolkit import math as fhe_math

factorial = fhe_math.factorial(max_n=7)
print(factorial.encrypt_run_decrypt(5))
# 120

modulo = fhe_math.modulo(
    min_numerator=0,
    max_numerator=20,
    min_denominator=0,
    max_denominator=10,
    zero_result=-1,
)
print(modulo.encrypt_run_decrypt(17, 5))
# 2

tanh = fhe_math.tanh(
    min_input=-30,
    max_input=30,
    input_scale=10,
    output_scale=100,
)
print(tanh.encrypt_run_decrypt(20))
# 96  (represents tanh(2.0) ~= 0.96)
```

The plain operation name compiles by default. If you want to be explicit, use
`.compile(...)`:

```python
from concrete_fhe_toolkit import math as fhe_math

gcd = fhe_math.gcd.compile(min_value=0, max_value=8)
print(gcd.simulate(6, 4))
# 2
```

Use `.make(...)` when composing a larger circuit yourself:

```python
from concrete import fhe
from concrete_fhe_toolkit import math as fhe_math

absolute = fhe_math.absolute.make(-8, 8)
gcd = fhe_math.gcd.make(0, 8)

def program(x, y):
    return gcd(absolute(x), y)

compiler = fhe.Compiler(program, {"x": "encrypted", "y": "encrypted"})
circuit = compiler.compile([(x, y) for x in range(-8, 9) for y in range(9)])
print(circuit.encrypt_run_decrypt(-6, 4))
# 2
```

Functions with undefined mathematical inputs require an explicit encrypted
fallback if the compiled domain includes invalid values:

```python
from concrete_fhe_toolkit import math as fhe_math

log2 = fhe_math.log2(
    min_input=0,
    max_input=16,
    invalid_result=-1,
)
print(log2.encrypt_run_decrypt(0))
# -1
```

Large lookup tables are available, but the package asks you to opt in when a
static estimate suggests the circuit may need substantial Concrete resources:

```python
from concrete_fhe_toolkit import math as fhe_math

gcd = fhe_math.gcd(
    min_value=0,
    max_value=31,
    allow_large_lookup=True,
)
print(gcd.simulate(24, 18))
# 6
```

### Bit-Level Fixed-Point Division

`unsigned_floor_divide` and `fixed_point_divide` use a restoring binary
division circuit instead of a full lookup table. They are useful when you want
division logic that scales with bit width rather than with every possible
`(numerator, denominator)` value.

```python
from concrete_fhe_toolkit import math as fhe_math

floor_divide = fhe_math.unsigned_floor_divide(
    numerator_width=8,
    denominator_width=8,
    zero_result=0,
)
print(floor_divide.encrypt_run_decrypt(100, 7))
# 14

fixed_divide = fhe_math.fixed_point_divide(
    numerator_width=8,
    denominator_width=8,
    fractional_bits=8,
)
print(fixed_divide.encrypt_run_decrypt(1, 3))
# 85  (represents floor((1 / 3) * 256))
```

Inputs for these helpers must be unsigned integers that fit in the declared bit
widths. Division by zero returns `zero_result`.

### Composing `make_*` Helpers

Use `make_*` helpers when you want to build a larger Concrete function yourself.
When compiling manually, your inputset must cover the bounds you declared.

```python
import numpy as np
from concrete import fhe

from concrete_fhe_toolkit import make_minimum

minimum_of_four = make_minimum(size=4, min_value=0, max_value=100)
compiler = fhe.Compiler(minimum_of_four, {"x": "encrypted"})

inputset = [
    np.array([0, 0, 0, 0], dtype=np.int64),
    np.array([100, 100, 100, 100], dtype=np.int64),
    np.array([0, 100, 0, 100], dtype=np.int64),
    np.array([100, 0, 100, 0], dtype=np.int64),
]

circuit = compiler.compile(inputset)
print(circuit.encrypt_run_decrypt(np.array([8, 3, 12, 5], dtype=np.int64)))
# 3
```

For most use cases, prefer the `compile_*` helpers because they generate
boundary-aware inputsets automatically.

## Bounds and Limitations

- Inputs must stay inside the bounds used at compilation.
- Plain `concrete_fhe_toolkit.math` operation names compile by default. Use
  `.make(...)` only when composing larger circuits manually.
- Sorting requires a power-of-two array size.
- Min/max and argmin/argmax support any positive fixed size.
- Division helpers currently support nonnegative bounded inputs.
- Bit-level division helpers support unsigned scalar inputs that fit in the
  declared bit widths and return integer or scaled-integer outputs.
- Larger bounds increase table-lookup bit width and cost. Some math helpers
  raise `LookupResourceError` unless you pass `allow_large_lookup=True`.
- `concrete_fhe_toolkit.math` fixed-point helpers return scaled integers, not
  Python floats.
- Lookup-table outputs that require very large integer bit widths may exceed
  Concrete's available parameter set even if you opt in to large lookups.
- Concrete supports integer inputs and outputs, not arbitrary floating point.
- FHE execution is probabilistic; choose Concrete error parameters appropriate
  for the application's risk.

## Notebook provenance and examples

This package was derived from these Kaggle experiments:

- [Concrete min/max/index finding](https://www.kaggle.com/code/erkankbacak/concrete-minmax-index-finding)
- [All encrypted division operations](https://www.kaggle.com/code/erkankbacak/all-encrypted-division-operations)
- [Concrete math library](https://www.kaggle.com/code/erkankbacak/concrete-math-library)

Additional usage notebook:

- [concrete-fhe-toolkit-test](https://www.kaggle.com/code/erkankbacak/concrete-fhe-toolkit-test)

The notebooks are useful background and examples, but the README and
[`docs/`](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit/tree/main/docs)
are the canonical documentation for the current package API. Existing
`make_*` and `compile_*` package APIs were kept for compatibility, while the
new `concrete_fhe_toolkit.math` friendly operation names provide the preferred
high-level interface for new code.

## Contributing

Contributions are welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to
set up a development environment, run the test suite, and submit changes. In
short:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e ".[dev]"
python -m pytest -q
```

## License and Concrete Terms

The package's original code is available under the MIT License. MIT permits
commercial and private use, modification, and redistribution while requiring
the copyright and license notice to be preserved.

Concrete is a separate dependency with its own BSD-3-Clause-Clear license and
patent terms. This package does not change or grant rights under Concrete's
license. Review Zama's current licensing terms before commercial use.

This project is not affiliated with or endorsed by Zama.

## Author and Contributors

**Author and maintainer**

- Erkan Işık Bacak ([@ErkanIsikB](https://github.com/ErkanIsikB))

**Contributors**

- Tolga Büyüktanır ([@tolgabuyuktanir](https://github.com/tolgabuyuktanir))
- Yusuf Emir Alakuş ([@YusufAlakus](https://github.com/YusufAlakus)
- Muharrem Uğurelli ([@mugurelli](https://github.com/mugurelli))
- Yücel Pehlevan ([@YucelPehlevan](https://github.com/YucelPehlevan)) 

**Organizational contributor**

- [AGRA Fintech Yazılım Çözümleri A.Ş.](https://www.agrafintech.com)

## Thanks

- [Zama](https://www.zama.ai) and the [Concrete](https://docs.zama.ai/concrete)
  team, whose FHE compiler makes this toolkit possible.
- The wider Zama open-source and FHE research community for tutorials,
  discussions, and example circuits.
- Everyone who tested the Kaggle notebooks and reported feedback that shaped the
  bounded, cost-aware API.

If you use `concrete-fhe-toolkit` in your project, a ⭐ on the
[GitHub repository](https://github.com/tolgabuyuktanir/concrete-fhe-toolkit) is always
appreciated.

## References

Background reading on the tools and techniques this package builds on:

- [Zama Concrete documentation](https://docs.zama.ai/concrete) — the FHE
  compiler this toolkit targets.
- [Concrete Python on GitHub](https://github.com/zama-ai/concrete) — source and
  release notes for `concrete-python`.
- [Zama blog](https://www.zama.ai/blog) — articles on FHE and TFHE.
- [TFHE: Fast Fully Homomorphic Encryption over the Torus](https://eprint.iacr.org/2018/421)
  (Chillotti, Gama, Georgieva, Izabachène) — the scheme underlying Concrete.
- [Programmable Bootstrapping Enables Efficient Homomorphic Inference of Deep Neural Networks](https://eprint.iacr.org/2021/091)
  — programmable bootstrapping, the mechanism behind the lookup-table helpers.
- [Batcher, "Sorting Networks and Their Applications" (1968)](https://doi.org/10.1145/1468075.1468121)
  — the bitonic sorting network used by `sort`.
- [Python `math` module](https://docs.python.org/3/library/math.html) — the
  reference semantics matched by the `concrete_fhe_toolkit.math` helpers.
