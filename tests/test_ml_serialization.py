import os
import json
import pytest

from concrete_fhe_toolkit.ml import (
    FHELogisticRegression,
    FHEKMeans,
    FHEDecisionTree
)
from concrete_fhe_toolkit.ml.serialization import save_model, load_model

def test_serialization_roundtrip_logistic_regression(tmp_path):
    model = FHELogisticRegression(weights=[3, 2], bias=-7)
    model.input_scale = 10  # an extra attribute
    path = tmp_path / "lr.json"
    
    save_model(model, str(path))
    
    assert path.exists()
    
    restored = load_model(str(path))
    
    assert isinstance(restored, FHELogisticRegression)
    assert restored.weights == [3, 2]
    assert restored.bias == -7
    assert restored.input_scale == 10

def test_serialization_roundtrip_kmeans_kwargs(tmp_path):
    model = FHEKMeans(centroids=[[1, 1], [2, 2]], max_distance=100)
    path = tmp_path / "kmeans.json"
    
    save_model(model, str(path))
    restored = load_model(str(path))
    
    assert isinstance(restored, FHEKMeans)
    assert restored.centroids == [[1, 1], [2, 2]]
    assert restored.max_distance == 100

def test_save_model_invalid_class(tmp_path):
    class InvalidModel:
        pass
    
    with pytest.raises(ValueError, match="is not serializable"):
        save_model(InvalidModel(), str(tmp_path / "invalid.json"))

def test_load_model_invalid_format(tmp_path):
    path = tmp_path / "bad.json"
    with open(path, "w") as f:
        json.dump({"format": "wrong/format"}, f)
        
    with pytest.raises(ValueError, match="is not a .* file"):
        load_model(str(path))

def test_load_model_unsupported_version(tmp_path):
    path = tmp_path / "bad_version.json"
    with open(path, "w") as f:
        json.dump({"format": "concrete-fhe-toolkit/model", "format_version": 999}, f)
        
    with pytest.raises(ValueError, match="unsupported model format_version"):
        load_model(str(path))

def test_load_model_unknown_class(tmp_path):
    path = tmp_path / "unknown.json"
    with open(path, "w") as f:
        json.dump({"format": "concrete-fhe-toolkit/model", "format_version": 1, "model": "UnknownFHEModel"}, f)
        
    with pytest.raises(ValueError, match="unknown model class in file"):
        load_model(str(path))

def test_load_model_missing_parameters(tmp_path):
    path = tmp_path / "missing.json"
    with open(path, "w") as f:
        json.dump({
            "format": "concrete-fhe-toolkit/model", 
            "format_version": 1, 
            "model": "FHELogisticRegression",
            "params": {"weights": [1, 2]} # missing bias
        }, f)
        
    with pytest.raises(ValueError, match="model file is missing parameters: bias"):
        load_model(str(path))
