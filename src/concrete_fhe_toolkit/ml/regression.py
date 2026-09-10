"""Regression task namespace: models, trainers, and metrics.

sklearn-style entry point — every class here predicts a numeric value from
encrypted features. Tree and MLP outputs are interpreted as scaled numeric
values. The boosting regressor sums tree leaves without a classification threshold.
"""

from .classes import FHEDecisionTree, FHELinearRegression, FHEMLP, FHEXGBoost
from .core import mean_absolute_error, mean_squared_error, r2_score
from .models import decision_tree_inference
from .trainers import FHELinearRegressionTrainer, linear_regression_training

FHEDecisionTreeRegressor = FHEDecisionTree
FHEMLPRegressor = FHEMLP


class FHEXGBoostRegressor(FHEXGBoost):
    """Sum public tree leaves to predict an integer regression score.

    Tree leaves must already incorporate any training-time learning rate.
    """

    def _circuit_logic(self, features):
        return sum(decision_tree_inference(features, tree) for tree in self.trees)

__all__ = [
    "FHEDecisionTreeRegressor",
    "FHELinearRegression",
    "FHELinearRegressionTrainer",
    "FHEMLPRegressor",
    "FHEXGBoostRegressor",
    "linear_regression_training",
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
]
