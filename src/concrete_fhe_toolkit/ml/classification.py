"""Classification task namespace: models, trainers, and metrics.

sklearn-style entry point — every class here predicts a class label from
encrypted features. The flat ``concrete_fhe_toolkit.ml`` names remain
available for backwards compatibility; the ``*Classifier`` names are
aliases of the same classes.
"""

from .classes import (
    FHEDecisionTree,
    FHEKNN,
    FHELogisticRegression,
    FHEMLP,
    FHENaiveBayes,
    FHENaiveBayesTrainer,
    FHERandomForest,
    FHESVM,
    FHEXGBoost,
)
from .core import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hinge_loss,
    precision_score,
    recall_score,
)
from .trainers import FHEDecisionTreeTrainer

FHEDecisionTreeClassifier = FHEDecisionTree
FHERandomForestClassifier = FHERandomForest
FHEXGBoostClassifier = FHEXGBoost
FHESVMClassifier = FHESVM
FHEKNNClassifier = FHEKNN
FHEMLPClassifier = FHEMLP

__all__ = [
    "FHEDecisionTree",
    "FHEDecisionTreeClassifier",
    "FHEDecisionTreeTrainer",
    "FHEKNN",
    "FHEKNNClassifier",
    "FHELogisticRegression",
    "FHEMLP",
    "FHEMLPClassifier",
    "FHENaiveBayes",
    "FHENaiveBayesTrainer",
    "FHERandomForest",
    "FHERandomForestClassifier",
    "FHESVM",
    "FHESVMClassifier",
    "FHEXGBoost",
    "FHEXGBoostClassifier",
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "hinge_loss",
    "precision_score",
    "recall_score",
]
