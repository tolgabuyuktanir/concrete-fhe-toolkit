"""Smoke tests for the example gallery, cost estimator, and rounded lookups."""

import math
import os
import subprocess
import sys

import pytest

from concrete_fhe_toolkit import ml
from concrete_fhe_toolkit.math._lookup import make_unary_lookup
from concrete_fhe_toolkit.ml.estimation import estimate_model_cost

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _run_example(name):
    return subprocess.run(
        [sys.executable, os.path.join(EXAMPLES, name)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_credit_scorecard_example_runs():
    result = _run_example("credit_scorecard.py")
    assert result.returncode == 0, result.stderr
    assert "ONAY" in result.stdout and "RED" in result.stdout


def test_kmeans_segmentation_example_runs():
    result = _run_example("kmeans_segmentation.py")
    assert result.returncode == 0, result.stderr
    assert "centroids:" in result.stdout


def test_breast_cancer_example_runs():
    pytest.importorskip("sklearn")
    result = _run_example("breast_cancer_diagnosis.py")
    assert result.returncode == 0, result.stderr
    assert "encrypted accuracy" in result.stdout


def test_estimate_model_cost_counts():
    linear = ml.FHELogisticRegression(weights=[1, 2, 3], bias=0)
    report = estimate_model_cost(linear, min_feature=0, max_feature=15)
    assert report.multiplications == 3 and report.comparisons == 1
    assert report.level == "small"

    tree = {"feature": 0, "threshold": 5,
            "left": {"feature": 1, "threshold": 3, "left": 1, "right": 0},
            "right": 0}
    forest = ml.FHERandomForest([tree] * 3)
    report = estimate_model_cost(forest, min_feature=0, max_feature=15)
    assert report.comparisons == 3 * 2 + 1
    assert report.multiplications == 3 * 2 * 2

    knn = ml.FHEKNN([[0, 0]] * 30, [0] * 30, 3)
    report = estimate_model_cost(knn, min_feature=0, max_feature=100)
    assert report.level == "very-large"
    assert any("bit" in note for note in report.notes)

    with pytest.raises(ValueError):
        estimate_model_cost(object(), min_feature=0, max_feature=1)


def test_estimate_pipeline_sums_steps():
    from concrete_fhe_toolkit.ml.pipeline import FHEPipeline
    from concrete_fhe_toolkit.ml.preprocessing import FHEBinner

    pipeline = FHEPipeline([
        FHEBinner([[1, 2, 3], [4, 5]]),
        ml.FHELogisticRegression(weights=[1, 1], bias=0),
    ])
    report = estimate_model_cost(pipeline, min_feature=0, max_feature=10)
    assert report.comparisons == 5 + 1
    assert report.multiplications == 2


def test_rounded_lookup_approximates():
    values = [round(math.sqrt(v) * 10) for v in range(1024)]

    exact = make_unary_lookup(values, 0)
    rounded = make_unary_lookup(values, 0, precision=6)

    for probe in (0, 100, 500, 1000, 1023):
        exact_value = int(exact(probe))
        approximate = int(rounded(probe))
        # 10-bit domain at 6-bit precision: input snaps to 16-step grid.
        assert abs(approximate - exact_value) <= 30
        assert approximate in values

    # Full precision requested -> identical to the exact lookup.
    same = make_unary_lookup(values, 0, precision=10)
    assert int(same(777)) == int(exact(777))
