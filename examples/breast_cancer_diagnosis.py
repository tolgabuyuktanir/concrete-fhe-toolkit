"""Privacy-preserving breast-cancer diagnosis with the sklearn bridge.

Train a standard sklearn model on clear data, convert it with one call, and
diagnose encrypted patients: the server evaluates the model without ever
seeing the medical features. Requires scikit-learn.

Run:  python examples/breast_cancer_diagnosis.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from concrete_fhe_toolkit.ml import accuracy_score, from_sklearn_linear

SCALE = 10


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler().fit(X_train)
    quantize = lambda X: np.round(scaler.transform(X) * SCALE).astype(np.int64)  # noqa: E731

    clf = LogisticRegression(max_iter=10000).fit(quantize(X_train), y_train)

    # One call replaces the whole manual weight-scaling ritual.
    model = from_sklearn_linear(clf, scale=SCALE)
    model.compile([row for row in quantize(X_test)[:20]])

    samples = quantize(X_test)[:15]
    predictions = [int(model.simulate(row)) for row in samples]
    truth = list(y_test[:15])
    print("predictions:", predictions)
    print("truth:      ", truth)
    print(f"encrypted accuracy: %{int(accuracy_score(predictions, truth))}")


if __name__ == "__main__":
    main()
