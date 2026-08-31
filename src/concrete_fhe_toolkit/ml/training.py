from typing import List, Any
from concrete_fhe_toolkit.ml import matrix_transpose, dot_product
from concrete import fhe

def naive_bayes_training(X_train: List[List[Any]], y_train_one_hot: List[List[Any]]) -> tuple[List[List[Any]],List[Any]]:
    """Encrypted training logic for Bernoulli Naive Bayes.
    
    Computes feature counts and class counts directly on encrypted data using FHE.
    Because FHE circuits cannot use traditional if-else logic or dynamic loops to filter 
    datasets by class, we use a mathematical matrix trick to count occurrences.

    Mathematical Trick (Why Binary and One-Hot?):
    - `X_train` must contain binary features (0 or 1).
    - `y_train_one_hot` must be one-hot encoded labels (e.g., [0, 1, 0]).
    - By transposing the matrices and computing the dot product between a feature 
      column and a class column, we multiply `Feature * Is_Class`. 
    - Since both are 0 or 1, the product is 1 ONLY if the sample has the feature AND 
      belongs to the class. Summing these products gives the exact count of how many 
      times the feature appeared in that class, completely within FHE.

    Args:
        X_train (List[List[Any]]): The encrypted binary feature matrix (samples x features).
        y_train_one_hot (List[List[Any]]): The encrypted one-hot encoded labels (samples x classes).

    Returns:
        tuple: (feature_counts, class_counts) as encrypted arrays.
    """
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