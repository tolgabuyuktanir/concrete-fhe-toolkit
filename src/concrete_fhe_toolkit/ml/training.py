from typing import List, Any
from concrete_fhe_toolkit.ml import matrix_transpose, dot_product
from concrete import fhe

def naive_bayes_training(X_train: List[List[Any]], y_train_one_hot: List[List[Any]]) -> tuple[List[Any],List[Any]]:
    class_counts = [0] * len(y_train_one_hot[0])
    for row in y_train_one_hot:
        for i in range(len(row)):
            class_counts[i] += row[i]

    X_train_transpose = matrix_transpose(X_train)
    y_train_tranpose = matrix_transpose(y_train_one_hot)

    feature_counts = []
    for row_y in y_train_tranpose:
        class_features = []
        for row_x in X_train_transpose:
            class_features.append(dot_product(row_x,row_y))
        feature_counts.append(class_features)

    return fhe.array(feature_counts), fhe.array(class_counts)    