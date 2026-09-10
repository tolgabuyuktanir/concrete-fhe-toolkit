"""Resource rejection must happen before allocating lookup values."""

import pytest
from concrete_fhe_toolkit.math import basic, combinatorics, fixed_point, number_theory, special
from concrete_fhe_toolkit.math._lookup import check_lookup_domain, LookupResourceError


def _must_not_evaluate(*args, **kwargs):
    raise AssertionError("lookup values were evaluated before the domain guard")


@pytest.mark.parametrize(
    "module,function,args,allocator",
    [
        (number_theory, "compile_gcd", (0, 10**9), "_binary_math_values"),
        (number_theory, "compile_dist", (2, 0, 10**9), "unary_values"),
        (combinatorics, "compile_factorial", (10**9,), "_factorials"),
        (combinatorics, "compile_comb", (10**9,), "binary_values"),
        (special, "compile_sin", (0, 10**9), "_scaled_values"),
        (special, "compile_atan2", (0, 10**9), "_atan2_values"),
        (fixed_point, "compile_floor", (0, 10**9), "_scaled_values"),
        (basic, "compile_divmod", (0, 10**9, 0, 10**9), "make_divmod"),
        (basic, "compile_saturating_multiply", (-(10**9), 10**9), "unary_values"),
    ],
)
def test_reject_before_values(module, function, args, allocator, monkeypatch):
    monkeypatch.setattr(module, allocator, _must_not_evaluate)
    with pytest.raises(LookupResourceError):
        getattr(module, function)(*args)


def test_domain_boundary_and_explicit_opt_in():
    check_lookup_domain("probe", (0, 511), allow_large_lookup=False)
    with pytest.raises(LookupResourceError):
        check_lookup_domain("probe", (0, 512), allow_large_lookup=False)
    check_lookup_domain("probe", (0, 10**100), allow_large_lookup=True)
    with pytest.raises(ValueError):
        check_lookup_domain("probe", (1, 0), allow_large_lookup=True)


def test_compile_opt_in_reaches_evaluation(monkeypatch):
    monkeypatch.setattr(number_theory, "_binary_math_values", _must_not_evaluate)
    with pytest.raises(AssertionError, match="evaluated"):
        number_theory.compile_gcd(0, 31, allow_large_lookup=True)
