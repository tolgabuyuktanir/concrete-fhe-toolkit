"""Encrypted trainers following the aggregate-decrypt pattern.

Every trainer here works the same way: a fixed circuit computes *aggregate
statistics* (counts, sums, X^T X, ...) over encrypted training data; the key
holder decrypts only those aggregates — never the raw data — and the model
is assembled clear-side from them. This is the pattern established by
``FHENaiveBayesTrainer`` and it is what makes encrypted training practical
under Concrete's bounded-circuit model:

- no data-dependent loops (iteration counts are fixed and public),
- divisions and transcendental steps happen clear-side on aggregates,
- circuit bit widths stay small because only counts/sums are accumulated.

Set ``simulate=True`` to run the circuits in Concrete's simulator (fast, no
key generation) for prototyping and tests; simulation does NOT protect the
data.
"""

from __future__ import annotations

import math as _pymath
from typing import Any, List, Optional

import numpy as np

from .._compat import fhe
from .._utils import validate_bounds, validate_integer
from ..math import equal, greater_equal
from .core import euclidean_distance_squared
from .matrix import matrix_multiply, matrix_transpose, matrix_vector_multiply
from ..arrays import make_argmin
from .classes import FHEDecisionTree, FHEKMeans, FHELinearRegression


class FHETrainer:
    """Base class for encrypted trainers (aggregate-decrypt pattern).

    Subclasses implement ``fit_encrypted(...)``, using ``_run_circuit`` to
    compile and execute one aggregate circuit over the encrypted inputs.

    Args:
        simulate: When True, run circuits in simulation instead of real
            encrypted execution (fast; for prototyping and tests only).
        configuration: Optional ``fhe.Configuration`` forwarded to compile.
    """

    def __init__(
        self,
        *,
        simulate: bool = False,
        configuration: Optional[fhe.Configuration] = None,
    ) -> None:
        self.simulate = simulate
        self.configuration = configuration
        self.circuit = None

    def _run_circuit(self, function, parameter_encryption, inputset, args):
        compiler = fhe.Compiler(function, parameter_encryption)
        if self.configuration is None:
            self.circuit = compiler.compile(inputset)
        else:
            self.circuit = compiler.compile(inputset, configuration=self.configuration)
        if self.simulate:
            return self.circuit.simulate(*args)
        return self.circuit.encrypt_run_decrypt(*args)

    def fit_encrypted(self, *args, **kwargs):
        raise NotImplementedError("fit_encrypted must be implemented by subclasses")


def linear_regression_training(
    X_train: Any,
    y_train: Any,
    n_samples: int,
    n_features: int,
) -> Any:
    """Traceable sufficient statistics for linear regression.

    Computes the flattened ``A^T A`` and ``A^T y`` aggregates over the
    encrypted design matrix ``A = [X | 1]`` (intercept column appended).
    The normal equations are solved clear-side after decryption.
    """
    augmented = [
        [X_train[row][column] for column in range(n_features)] + [1]
        for row in range(n_samples)
    ]
    transposed = matrix_transpose(augmented)
    xtx = matrix_multiply(transposed, augmented)
    xty = matrix_vector_multiply(
        transposed,
        [y_train[row] for row in range(n_samples)],
    )
    flat = [cell for row in xtx for cell in row] + list(xty)
    return fhe.array(flat)


class FHELinearRegressionTrainer(FHETrainer):
    """Encrypted linear regression training via sufficient statistics.

    One circuit computes ``A^T A`` and ``A^T y`` (with an intercept column)
    over encrypted samples; only those aggregates are decrypted and the
    normal equations are solved clear-side.

    The returned :class:`FHELinearRegression` carries integer weights scaled
    by ``weight_scale``, so its predictions are ``weight_scale`` times the
    real value — decode with ``prediction / weight_scale``.
    """

    def __init__(
        self,
        *,
        weight_scale: int = 100,
        simulate: bool = False,
        configuration: Optional[fhe.Configuration] = None,
    ) -> None:
        super().__init__(simulate=simulate, configuration=configuration)
        self.weight_scale = validate_integer("weight_scale", weight_scale, minimum=1)

    def fit_encrypted(self, X_train: List[List[int]], y_train: List[int]):
        n_samples = len(X_train)
        if n_samples == 0:
            raise ValueError("X_train must contain at least one sample")
        n_features = len(X_train[0])
        if any(len(row) != n_features for row in X_train):
            raise ValueError("every sample must have the same number of features")
        if len(y_train) != n_samples:
            raise ValueError("X_train and y_train must have the same length")

        def training_circuit(X_train, y_train):
            return linear_regression_training(X_train, y_train, n_samples, n_features)

        X_array = np.array(X_train, dtype=np.int64)
        y_array = np.array(y_train, dtype=np.int64)
        flat = np.array(
            self._run_circuit(
                training_circuit,
                {"X_train": "encrypted", "y_train": "encrypted"},
                [(X_array, y_array)],
                (X_array, y_array),
            )
        )

        dimension = n_features + 1
        xtx = flat[: dimension * dimension].reshape(dimension, dimension).astype(float)
        xty = flat[dimension * dimension:].astype(float)
        try:
            solution = np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "normal equations are singular; provide more varied training samples"
            ) from error

        weights = [int(round(value * self.weight_scale)) for value in solution[:-1]]
        bias = int(round(solution[-1] * self.weight_scale))
        model = FHELinearRegression(weights, bias)
        model.output_scale = self.weight_scale
        return model


def _gini(class_counts: List[int]) -> float:
    total = sum(class_counts)
    if total == 0:
        return 0.0
    return 1.0 - sum((count / total) ** 2 for count in class_counts)


class FHEDecisionTreeTrainer(FHETrainer):
    """Hybrid encrypted decision-tree training (level-wise aggregate counting).

    For each tree level, one circuit computes — under encryption — the
    class counts of every (node, candidate split) combination. Only those
    counts are decrypted; Gini impurity and split selection happen
    clear-side, and the chosen splits become public constants of the next
    level's circuit. Raw samples and labels are never decrypted; what leaks
    is the per-candidate aggregate histogram (the same trust model as
    ``FHENaiveBayesTrainer``'s decrypted counts).

    Args:
        candidate_thresholds: Per feature, the public list of candidate
            thresholds (splits test ``feature >= threshold``). Keep these
            lists short — cost grows with nodes x candidates x samples.
        max_depth: Maximum tree depth (levels of splits).
        num_classes: Number of label classes (labels are 0..num_classes-1).
        min_samples_leaf: A split is rejected when either side would hold
            fewer samples than this.
    """

    def __init__(
        self,
        candidate_thresholds: List[List[int]],
        *,
        max_depth: int = 2,
        num_classes: int = 2,
        min_samples_leaf: int = 1,
        simulate: bool = False,
        configuration: Optional[fhe.Configuration] = None,
    ) -> None:
        super().__init__(simulate=simulate, configuration=configuration)
        if not candidate_thresholds or any(
            not isinstance(row, (list, tuple)) for row in candidate_thresholds
        ):
            raise ValueError(
                "candidate_thresholds must be a per-feature list of threshold lists"
            )
        self.candidate_thresholds = [
            [validate_integer("threshold", value) for value in row]
            for row in candidate_thresholds
        ]
        self.max_depth = validate_integer("max_depth", max_depth, minimum=1)
        self.num_classes = validate_integer("num_classes", num_classes, minimum=2)
        self.min_samples_leaf = validate_integer(
            "min_samples_leaf", min_samples_leaf, minimum=1
        )

    def _sample_mask(self, X_train, row: int, path) -> Any:
        mask: Any = 1
        for feature, threshold, side in path:
            comparison = greater_equal(X_train[row][feature], threshold)
            if side == "ge":
                mask = mask * comparison
            else:
                mask = mask * (1 - comparison)
        return mask

    def _level_circuit(self, paths, n_samples: int, with_candidates: bool):
        candidates = self.candidate_thresholds
        num_classes = self.num_classes

        def level_counts(X_train, y_train):
            outputs = []
            for path in paths:
                masks = [
                    self._sample_mask(X_train, row, path)
                    for row in range(n_samples)
                ]
                class_flags = [
                    [equal(y_train[row], label) for label in range(num_classes)]
                    for row in range(n_samples)
                ]
                for label in range(num_classes):
                    total: Any = 0
                    for row in range(n_samples):
                        total = total + masks[row] * class_flags[row][label]
                    outputs.append(total)
                if with_candidates:
                    for feature, thresholds in enumerate(candidates):
                        for threshold in thresholds:
                            for label in range(num_classes):
                                total = 0
                                for row in range(n_samples):
                                    goes_left = greater_equal(
                                        X_train[row][feature], threshold
                                    )
                                    total = total + (
                                        masks[row]
                                        * goes_left
                                        * class_flags[row][label]
                                    )
                                outputs.append(total)
            return fhe.array(outputs)

        return level_counts

    def _choose_split(self, node_counts, candidate_counts):
        best = None
        node_total = sum(node_counts)
        for (feature, threshold), left_counts in candidate_counts:
            left_total = sum(left_counts)
            right_counts = [
                node - left for node, left in zip(node_counts, left_counts)
            ]
            right_total = node_total - left_total
            if left_total < self.min_samples_leaf or right_total < self.min_samples_leaf:
                continue
            score = left_total * _gini(left_counts) + right_total * _gini(right_counts)
            if best is None or score < best[0]:
                best = (score, feature, threshold)
        return best

    def fit_encrypted(self, X_train: List[List[int]], y_train: List[int]):
        n_samples = len(X_train)
        if n_samples == 0:
            raise ValueError("X_train must contain at least one sample")
        n_features = len(X_train[0])
        if len(self.candidate_thresholds) != n_features:
            raise ValueError("candidate_thresholds must list thresholds per feature")
        if len(y_train) != n_samples:
            raise ValueError("X_train and y_train must have the same length")

        X_array = np.array(X_train, dtype=np.int64)
        y_array = np.array(y_train, dtype=np.int64)

        candidate_keys = [
            (feature, threshold)
            for feature, thresholds in enumerate(self.candidate_thresholds)
            for threshold in thresholds
        ]

        container: dict = {}
        frontier = [{"path": [], "attach": (container, "root")}]

        for depth in range(self.max_depth + 1):
            if not frontier:
                break
            with_candidates = depth < self.max_depth
            paths = [node["path"] for node in frontier]
            circuit_fn = self._level_circuit(paths, n_samples, with_candidates)
            flat = list(
                np.array(
                    self._run_circuit(
                        circuit_fn,
                        {"X_train": "encrypted", "y_train": "encrypted"},
                        [(X_array, y_array)],
                        (X_array, y_array),
                    )
                ).astype(int)
            )

            per_node = self.num_classes * (
                1 + (len(candidate_keys) if with_candidates else 0)
            )
            next_frontier = []
            for index, node in enumerate(frontier):
                chunk = flat[index * per_node: (index + 1) * per_node]
                node_counts = chunk[: self.num_classes]
                parent, key = node["attach"]

                majority = int(np.argmax(node_counts))
                is_pure = sum(1 for count in node_counts if count > 0) <= 1
                best = None
                if with_candidates and not is_pure:
                    candidate_counts = []
                    for c_index, candidate in enumerate(candidate_keys):
                        offset = self.num_classes * (1 + c_index)
                        candidate_counts.append(
                            (candidate, chunk[offset: offset + self.num_classes])
                        )
                    best = self._choose_split(node_counts, candidate_counts)

                if best is None:
                    parent[key] = majority
                    continue

                _, feature, threshold = best
                subtree = {
                    "feature": feature,
                    "threshold": threshold,
                    "left": None,
                    "right": None,
                }
                parent[key] = subtree
                next_frontier.append(
                    {
                        "path": node["path"] + [(feature, threshold, "ge")],
                        "attach": (subtree, "left"),
                    }
                )
                next_frontier.append(
                    {
                        "path": node["path"] + [(feature, threshold, "lt")],
                        "attach": (subtree, "right"),
                    }
                )
            frontier = next_frontier

        return FHEDecisionTree(container["root"])


class FHEKMeansTrainer(FHETrainer):
    """Hybrid encrypted k-means training (fixed iterations).

    Each iteration compiles one circuit with the current public centroids:
    it assigns every encrypted sample to its nearest centroid and returns
    per-cluster sums and counts. Only those aggregates are decrypted; the
    centroid update (division) happens clear-side, and the new centroids
    become the next iteration's public constants.

    Args:
        initial_centroids: Public, data-independent starting centroids.
        min_value: Inclusive lower bound of every feature value.
        max_value: Inclusive upper bound of every feature value.
        n_iterations: Fixed number of Lloyd iterations.
    """

    def __init__(
        self,
        initial_centroids: List[List[int]],
        *,
        min_value: int,
        max_value: int,
        n_iterations: int = 5,
        simulate: bool = False,
        configuration: Optional[fhe.Configuration] = None,
    ) -> None:
        super().__init__(simulate=simulate, configuration=configuration)
        if not initial_centroids:
            raise ValueError("initial_centroids must contain at least one centroid")
        self.initial_centroids = [list(centroid) for centroid in initial_centroids]
        self.minimum, self.maximum = validate_bounds(min_value, max_value)
        self.n_iterations = validate_integer("n_iterations", n_iterations, minimum=1)

    def _step_circuit(self, centroids, n_samples: int, n_features: int, max_distance: int):
        n_clusters = len(centroids)
        arg_min = make_argmin(n_clusters, 0, max_distance)

        def assign_and_aggregate(X_train):
            flags = []
            for row in range(n_samples):
                sample = [X_train[row][column] for column in range(n_features)]
                distances = [
                    euclidean_distance_squared(centroid, sample)
                    for centroid in centroids
                ]
                nearest = arg_min(distances)
                flags.append(
                    [equal(cluster, nearest) for cluster in range(n_clusters)]
                )

            outputs = []
            for cluster in range(n_clusters):
                count: Any = 0
                for row in range(n_samples):
                    count = count + flags[row][cluster]
                outputs.append(count)
            for cluster in range(n_clusters):
                for column in range(n_features):
                    total: Any = 0
                    for row in range(n_samples):
                        total = total + flags[row][cluster] * X_train[row][column]
                    outputs.append(total)
            return fhe.array(outputs)

        return assign_and_aggregate

    def fit_encrypted(self, X_train: List[List[int]]):
        n_samples = len(X_train)
        if n_samples == 0:
            raise ValueError("X_train must contain at least one sample")
        n_features = len(X_train[0])
        span = self.maximum - self.minimum
        max_distance = max(1, n_features * span * span)
        X_array = np.array(X_train, dtype=np.int64)

        centroids = [list(centroid) for centroid in self.initial_centroids]
        n_clusters = len(centroids)

        for _ in range(self.n_iterations):
            circuit_fn = self._step_circuit(
                centroids, n_samples, n_features, max_distance
            )
            flat = list(
                np.array(
                    self._run_circuit(
                        circuit_fn,
                        {"X_train": "encrypted"},
                        [X_array],
                        (X_array,),
                    )
                ).astype(int)
            )
            counts = flat[:n_clusters]
            sums = flat[n_clusters:]
            new_centroids = []
            for cluster in range(n_clusters):
                if counts[cluster] == 0:
                    new_centroids.append(list(centroids[cluster]))
                    continue
                new_centroids.append(
                    [
                        int(
                            _pymath.floor(
                                sums[cluster * n_features + column]
                                / counts[cluster]
                                + 0.5
                            )
                        )
                        for column in range(n_features)
                    ]
                )
            centroids = new_centroids

        return FHEKMeans(centroids, max_distance=max_distance)
