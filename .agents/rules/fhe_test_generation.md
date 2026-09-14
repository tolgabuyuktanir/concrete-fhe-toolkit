---
trigger: always_on
description: Mandatory constraints for generating or modifying Concrete FHE test cases and notebook tutorials.
---

# FHE Test Generation Constraints

When generating tests, notebook tutorials, or automation scripts for `concrete-fhe-toolkit` (especially the `math` modules), you MUST strictly adhere to both the **structural notebook rules** and the **technical FHE compiler limits** below.

## Part 1: Structural Notebook Rules

These rules dictate how the Jupyter Notebook `.ipynb` files must be formatted and organized:

1. **One Notebook per File**: There must be exactly one dedicated notebook for each `.py` file being tested.
2. **Complete Coverage**: Every FHE-compatible function in the source file must be tested. This explicitly includes all `make_` and `compile_` variations.
3. **Isolation (One Function = One Cell)**: Each function must be tested in its own independent Jupyter code cell. Do not group multiple functions into a single code cell.
4. **Self-Contained & Clean Imports**: Every single code cell must include all the `import` statements required to run its specific function (e.g., `from concrete import fhe`, `import math`, etc.). Do not rely on imports from top-level cells. Do not include unnecessary imports.
5. **Function Ordering**: The sequence of tests in the notebook MUST strictly match the exact top-to-bottom order that the functions appear in the original source `.py` file.
6. **Alias Grouping**: For functions that have aliases (e.g., `compile_sub` and `compile_subtract`), do NOT write separate test cells. Write a single test cell for the primary function and clearly state the aliases in the Markdown explanation cell above it.
7. **Ignore External Dependencies**: Do NOT generate tests for imported helper functions or private functions (e.g., `validate_`, `unary_values`, `_make_scaled`). Only test the functions inherently defined within that specific module.
8. **English Markdown Documentation**: Above every test cell, there must be a markdown cell containing a simple, concise ENGLISH explanation of what the function does and what specific conditions/edge cases are being tested. All operations, variables, and documentation must exclusively use English.
9. **Assertion Verification**: Each test cell must include an `assert` block that verifies the FHE evaluation (`circuit.encrypt_run_decrypt()`) matches the expected cleartext Python evaluation. This ensures mathematical correctness.

## Part 2: Technical FHE Compiler & RAM Constraints

Concrete FHE Table Lookups (TLUs) can easily explode in bit-width and crash the compilation (RAM exhaustion) if limits are not respected.

1. **Tight Boundary Limits (RAM Protection)**: 
   Do not use generic large limits like `[-15, 15]` for operations involving division, clamping, modular arithmetic, or dynamic lookups. 
   - Constrain default numeric test limits to `[-7, 7]` to restrict TLU size (usually under 6-8 bits).
   - Bitwise parameters (e.g. `quotient_width`, `fractional_bits`) must be constrained strictly to `[1, 4]`.
2. **Deterministic Inputs**:
   Test input sets (`inputset`) should be dynamically generated to extensively cover positive, negative, hybrid, and zero-edge cases (e.g. division by zero). However, you MUST use a fixed random seed (e.g., `random.seed(42)`) so the generated sets are perfectly identical and predictable on every developer run.
3. **Boolean / Control Flags**:
   Any parameter named `control`, `condition`, or `flag` must strictly receive `0` or `1`. Never pass negative or arbitrary integers to these inputs.
4. **Iterable/Array Parameters (`fsum`, `sumprod`)**:
   Functions expecting iterables must be given actual Python lists (e.g., `[-1, 2, -3]`), not scalar tuples. Otherwise, iteration errors (`int object is not iterable`) will occur. `sumprod` requires two arrays of equal length.
5. **Cleartext Compiler Mapping**:
   Setup or configuration parameters (like `start`, `step`, `size`, `k`, `angle_unit`, `rounding`) cannot be traced natively by Concrete FHE. They must be explicitly mapped as `'clear'` in the `fhe.Compiler` dictionary (e.g., `compiler = fhe.Compiler(func, {'values': 'encrypted', 'start': 'clear'})`).
6. **Compiler Integration vs. Cleartext Helpers**: 
   Every FHE-compatible function MUST be compiled and tested using `fhe.Compiler`. However, true client-side preprocessing or helper functions that inherently cannot operate on encrypted `Tracer` variables (e.g., `encode_fixed_point`, evaluation metrics like `accuracy_score`) are EXEMPT from compiler integration. These specific helpers must be tested in standard cleartext-only mode, and their markdown block must explicitly state that they are client-side helpers not meant for FHE compilation.
   - **STRICT ANTI-FAKING RULE**: NEVER label a core mathematical or model function (e.g., `softmax`, `cross_entropy_loss`) as a "cleartext helper" just to bypass the compiler because it fails FHE constraints (like 16-bit TLU limits or array returns). If a function is conceptually meant to be part of the FHE circuit, it MUST be tested with `fhe.Compiler`, even if the compilation crashes. Failing tests are required to highlight where developers need to apply scale tuning or fix FHE limitations. Only use cleartext tests for TRUE client-side logic.
