"""Model format and numeric regression contract tests."""

import json
import pytest
from concrete import fhe
from concrete_fhe_toolkit import ml, privacy
from concrete_fhe_toolkit.ml.regression import FHEXGBoostRegressor
from concrete_fhe_toolkit.ml.serialization import save_model, load_model


def test_boosting_regressor_preserves_numeric_scores(tmp_path):
    assert FHEXGBoostRegressor([5, 7])._circuit_logic([0]) == 12
    tree = {"feature": 0, "threshold": 2, "left": 5, "right": -3}
    model = FHEXGBoostRegressor([tree, tree])
    assert model._circuit_logic([0]) == -6
    assert model._circuit_logic([3]) == 10
    assert int(ml.FHEXGBoost([tree, tree])._circuit_logic([3])) == 1
    path = tmp_path / "regressor.json"
    save_model(model, str(path))
    restored = load_model(str(path))
    assert type(restored) is FHEXGBoostRegressor
    restored.compile([[0], [3]], configuration=fhe.Configuration(p_error=2**-40))
    assert [int(v) for v in restored.simulate_many([[0], [3]])] == [-6, 10]
    report = ml.estimate_model_cost(model, min_feature=0, max_feature=3)
    assert report.comparisons == 2


@pytest.mark.parametrize("version", [None, 0, 2, 999, True, 1.0, "1"])
def test_load_rejects_unknown_or_invalid_version(tmp_path, version):
    path = tmp_path / "model.json"
    save_model(ml.FHELinearRegression([2], 1), str(path))
    doc = json.loads(path.read_text())
    if version is None:
        del doc["format_version"]
    else:
        doc["format_version"] = version
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="format_version"):
        load_model(str(path))


def test_model_format_preserves_scales(tmp_path):
    model = ml.FHELinearRegression([20], 100)
    model.output_scale = 100
    model.input_scale = 10
    path = tmp_path / "model.json"
    save_model(model, str(path))
    restored = load_model(str(path))
    assert restored.output_scale == 100
    assert restored.input_scale == 10
    assert restored._circuit_logic([30]) == 700


def test_gaussian_total_budget_validation_and_noise():
    import numpy as np

    values = [1, 2, 3]
    with pytest.raises(ValueError, match="total epsilon"):
        privacy.dp_release(values, sensitivity=1, epsilon=2, delta=1e-5, mechanism="gaussian")
    for delta in [0, 1, None]:
        with pytest.raises(ValueError):
            privacy.dp_release(values, sensitivity=1, epsilon=1, delta=delta, mechanism="gaussian")
    # The existing conservative noise stays unchanged in the supported range.
    actual = privacy.dp_release(
        values,
        sensitivity=1,
        epsilon=1,
        delta=1e-5,
        mechanism="gaussian",
        rng=np.random.default_rng(12),
    )
    rng = np.random.default_rng(12)
    expected = [
        privacy.gaussian_mechanism(v, sensitivity=1, epsilon=1 / 3, delta=1e-5, rng=rng)
        for v in values
    ]
    assert actual == expected
