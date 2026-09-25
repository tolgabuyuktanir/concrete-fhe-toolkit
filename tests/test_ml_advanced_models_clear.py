import pytest
import numpy as np

from concrete_fhe_toolkit.ml import models

def test_svm_inference_clear():
    weights = [2, -1]
    bias = -1
    
    # 2*2 + (-1)*1 - 1 = 2 -> positive side => 1
    assert int(models.svm_inference(weights, bias, [2, 1])) == 1
    # 2*1 + (-1)*3 - 1 = -2 -> negative side => -1
    assert int(models.svm_inference(weights, bias, [1, 3])) == -1

def test_pca_inference_clear():
    features = [10, 20]
    means = [5, 10]
    components = [[1, 0], [0, 1]]
    
    # (10-5, 20-10) * I = (5, 10)
    res = models.pca_inference(features, means, components)
    assert hasattr(res, "tolist")
    assert res.tolist() == [5, 10]
    
    components_proj = [[1, 1]]
    res_proj = models.pca_inference(features, means, components_proj)
    assert res_proj.tolist() == [15]

def test_xgboost_inference_clear():
    # Simple tree: feature 0 >= 5 -> left(return 10), right(return -5)
    tree1 = {
        "feature": 0,
        "threshold": 5,
        "left": 10,
        "right": -5
    }
    tree2 = {
        "feature": 1,
        "threshold": 3,
        "left": 5,
        "right": -2
    }
    
    # x=[6, 4] -> tree1 left(10), tree2 left(5) -> sum=15 > 0 -> 1
    assert int(models.xgboost_inference([6, 4], [tree1, tree2])) == 1
    # x=[4, 2] -> tree1 right(-5), tree2 right(-2) -> sum=-7 < 0 -> 0
    assert int(models.xgboost_inference([4, 2], [tree1, tree2])) == 0

def test_pooling_2d_clear():
    # Concrete maxpool/avgpool might only support 1 channel internally right now
    image = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    ]
    
    max_pool = models.max_pooling_2d(image)
    # Expected channel 1: max of [[1,2],[4,5]] => 5
    assert hasattr(max_pool, "tolist")
    assert max_pool.tolist() == [5]
    
    avg_pool = models.avg_pooling_2d(image)
    # Expected channel 1: avg of [1,2,4,5] => 12//4 => 3
    assert avg_pool.tolist() == [3]

def test_cnn_inference_clear():
    # 1 channel, 3x3 image
    image = [
        [[1, 2, 3], 
         [4, 5, 6], 
         [7, 8, 9]]
    ]
    
    # 1 filter (F=1), H=2, W=2 (no channel dim, expanded internally)
    filters = [
        [[1, 0], 
         [0, -1]]
    ]
    bias = [10]
    
    # Output should be 2x2, flattened
    # top-left: 1*1 + 2*0 + 4*0 + 5*-1 + 10 = 1 - 5 + 10 = 6
    # top-right: 2*1 + 3*0 + 5*0 + 6*-1 + 10 = 2 - 6 + 10 = 6
    # bottom-left: 4*1 + 5*0 + 7*0 + 8*-1 + 10 = 4 - 8 + 10 = 6
    # bottom-right: 5*1 + 6*0 + 8*0 + 9*-1 + 10 = 5 - 9 + 10 = 6
    
    res = models.cnn_inference(filters, bias, image)
    assert hasattr(res, "tolist")
    assert res.tolist() == [6, 6, 6, 6]

def test_auto_quantizer_clear():
    images = [
        [
            [[1, 2], [3, 4]]
        ],
        [
            [[5, 6], [7, 8]]
        ]
    ]
    filters = [
        [
            [[1, 1], [1, 1]]
        ]
    ]
    
    class DummyModel:
        def __init__(self):
            self.coef_ = np.array([[0.5, 0.5]])
    
    res = models.auto_quantizer(images, filters, DummyModel(), mode="optimal")
    assert isinstance(res, (int, float))
