"""Regression task namespace: models, trainers, and metrics.

sklearn-style entry point — every class here predicts a numeric value from
encrypted features. Tree/forest/MLP regressors are the same engines as
their classifier counterparts; their integer leaves and outputs are simply
interpreted as (scaled) numeric values.
"""

from .classes import FHEDecisionTree, FHELinearRegression, FHEMLP, FHEXGBoost
from .core import mean_absolute_error, mean_squared_error, r2_score
from .trainers import FHELinearRegressionTrainer, linear_regression_training

FHEDecisionTreeRegressor = FHEDecisionTree
FHEXGBoostRegressor = FHEXGBoost
FHEMLPRegressor = FHEMLP

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
