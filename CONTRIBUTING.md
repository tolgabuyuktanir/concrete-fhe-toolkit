# Contributing to concrete-fhe-toolkit

Thanks for your interest in improving `concrete-fhe-toolkit`! This guide covers
the basics for getting a development environment running and submitting changes.

## Development environment

Concrete 2.11 supports **Python 3.9–3.12 on Linux and macOS**. On Windows, use
WSL2 or a Linux container.

```bash
git clone https://github.com/tolgabuyuktanir/concrete-fhe-toolkit.git
cd concrete-fhe-toolkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e ".[dev]"
```

## Running the tests

The fast suite runs cleartext and compiler/simulation tests:

```bash
python -m pytest -q
```

The expensive encrypted smoke tests are opt-in through environment variables:

```bash
RUN_FHE_TESTS=1 python -m pytest -q tests/test_fhe_smoke.py
RUN_FHE_MATH_TESTS=1 python -m pytest -q tests/test_math_fhe_smoke.py
RUN_FHE_NOTEBOOK_TESTS=1 python -m pytest -q tests/test_notebook_regressions.py
```

See [`docs/testing-and-release.md`](docs/testing-and-release.md) for the full
maintainer checklist.

## Coding guidelines

- Keep circuit builders **bounded and explicit** — every operation should take
  public input bounds and stay side-effect free so it can be traced by Concrete.
- Match the surrounding style: descriptive names, docstrings on public
  functions, and type hints.
- Add tests for new behavior. Prefer exhaustive cleartext checks over small
  domains plus a representative compiler/simulation test.
- Keep large lookup tables behind the `allow_large_lookup=True` opt-in so users
  do not accidentally compile huge circuits.

## Submitting changes

1. Create a feature branch from `main`.
2. Make your change, add tests, and ensure `python -m pytest -q` passes.
3. Update the relevant docs and [`CHANGELOG.md`](CHANGELOG.md).
4. Open a pull request describing the motivation and the behavior change.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
