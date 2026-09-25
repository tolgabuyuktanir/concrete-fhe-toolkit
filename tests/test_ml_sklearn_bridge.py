import pytest
import numpy as np

from concrete_fhe_toolkit.ml import (
    FHELinearRegression,
    FHELogisticRegression,
    FHEDecisionTree,
    FHERandomForest
)
from concrete_fhe_toolkit.ml.sklearn_bridge import (
    from_sklearn_linear,
    from_sklearn_tree,
    from_sklearn_forest
)

# --- Mocks for sklearn models ---

class MockLinearRegression:
    def __init__(self, coef, intercept):
        self.coef_ = np.array(coef)
        self.intercept_ = np.array(intercept)

class MockLogisticRegression(MockLinearRegression):
    def __init__(self, coef, intercept):
        super().__init__(coef, intercept)
        self.classes_ = np.array([0, 1])

class MockTree:
    def __init__(self, children_left, children_right, value, threshold, feature):
        self.children_left = children_left
        self.children_right = children_right
        self.value = value
        self.threshold = threshold
        self.feature = feature

class MockDecisionTree:
    def __init__(self, tree):
        self.tree_ = tree

class MockRandomForest:
    def __init__(self, estimators):
        self.estimators_ = estimators

# --- Tests for linear bridge ---

def test_from_sklearn_linear_regression():
    # 1D coef
    model = MockLinearRegression(coef=[0.5, -1.2], intercept=[3.0])
    fhe_model = from_sklearn_linear(model, scale=100, input_scale=2)
    
    assert isinstance(fhe_model, FHELinearRegression)
    # weights: round(0.5 * 100) = 50, round(-1.2 * 100) = -120
    assert fhe_model.weights == [50, -120]
    # bias: round(3.0 * 100 * 2) = 600
    assert fhe_model.bias == 600
    assert getattr(fhe_model, "input_scale", 1) == 2
    assert getattr(fhe_model, "output_scale", 1) == 200

def test_from_sklearn_logistic_regression():
    model = MockLogisticRegression(coef=[[0.5, -1.2]], intercept=[3.0])
    fhe_model = from_sklearn_linear(model, scale=10)
    
    assert isinstance(fhe_model, FHELogisticRegression)
    assert fhe_model.weights == [5, -12]
    assert fhe_model.bias == 30

def test_from_sklearn_linear_invalid():
    class BadModel:
        pass
    with pytest.raises(ValueError, match="must be a fitted sklearn linear estimator"):
        from_sklearn_linear(BadModel())
        
    model = MockLinearRegression(coef=[[1, 2], [3, 4]], intercept=[1, 2])
    with pytest.raises(ValueError, match="only binary/single-output linear models"):
        from_sklearn_linear(model)

# --- Tests for tree bridge ---

def test_from_sklearn_tree_classification():
    # simple tree with 1 split and 2 leaves
    tree_data = MockTree(
        children_left=np.array([1, -1, -1]),
        children_right=np.array([2, -1, -1]),
        value=np.array([[[10, 20]], [[30, 5]], [[2, 40]]]), # majority class: root 1, left 0, right 1
        threshold=np.array([2.5, -2, -2]),
        feature=np.array([0, -2, -2])
    )
    model = MockDecisionTree(tree_data)
    fhe_model = from_sklearn_tree(model, scale=10)
    
    assert isinstance(fhe_model, FHEDecisionTree)
    
    # Sklearn right goes to FHE left
    # feature >= floor(2.5 * 10) + 1 => feature >= 26
    # Right child (index 2) has value [[2, 40]] -> majority class 1 (index 1 is max)
    # Left child (index 1) has value [[30, 5]] -> majority class 0
    tree_dict = fhe_model.tree
    assert tree_dict["feature"] == 0
    assert tree_dict["threshold"] == 26
    assert tree_dict["left"] == 1  # Right child from sklearn
    assert tree_dict["right"] == 0 # Left child from sklearn

def test_from_sklearn_tree_invalid():
    with pytest.raises(ValueError, match="must be a fitted sklearn decision tree"):
        from_sklearn_tree(object())

# --- Tests for forest bridge ---

def test_from_sklearn_forest():
    tree_data = MockTree(
        children_left=np.array([-1]),
        children_right=np.array([-1]),
        value=np.array([[[10, 20]]]),
        threshold=np.array([-2]),
        feature=np.array([-2])
    )
    t1 = MockDecisionTree(tree_data)
    t2 = MockDecisionTree(tree_data)
    model = MockRandomForest([t1, t2])
    
    fhe_model = from_sklearn_forest(model)
    assert isinstance(fhe_model, FHERandomForest)
    assert len(fhe_model.trees) == 2
    assert fhe_model.trees[0] == 1 # leaf majority class 1

def test_from_sklearn_forest_invalid():
    with pytest.raises(ValueError, match="must be a fitted sklearn forest"):
        from_sklearn_forest(object())
