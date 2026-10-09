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
from typing import Any, List, Optional, Union

import numpy as np
import gc

from .._compat import fhe
from .._utils import validate_bounds, validate_integer
from ..math import equal, greater_equal
from .core import euclidean_distance_squared
from .matrix import matrix_multiply, matrix_transpose, matrix_vector_multiply
from ..arrays import make_argmin
from .classes import FHEDecisionTree, FHEKMeans, FHELinearRegression, FHERandomForest
from .utils import _get_progress_bar
from .training import naive_bayes_training


class FHETrainer:
    """Base class for encrypted trainers (aggregate-decrypt pattern).

    Subclasses implement ``fit_encrypted(...)``, using ``_run_circuit`` to
    compile and execute one aggregate circuit over the encrypted inputs.

    Args:
        simulate (bool): When True, run circuits in simulation instead of real
            encrypted execution (fast; for prototyping and tests only).
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Example:
        ```python
        trainer = FHETrainer(simulate=True)
        ```
    """

    def __init__(
        self,
        *,
        simulate: bool = False,
        configuration: Optional[fhe.Configuration] = None,
        verbose: bool = False
    ) -> None:
        """Initialize the FHETrainer base with simulation and configuration options."""
        self.simulate = simulate
        self.configuration = configuration
        self.verbose = verbose
        self.circuit = None

    def _run_circuit(self, function: Any, parameter_encryption: Any, inputset: Any, args: Any) -> Any:
        """Compiles and runs a circuit.
        
        Args:
            function (Any): The circuit function.
            parameter_encryption (Any): The encryption parameter dictionary.
            inputset (Any): The inputset for compilation.
            args (Any): The arguments to run or simulate.
            
        Returns:
            Any: The execution result.
        """
        compiler = fhe.Compiler(function, parameter_encryption)
        if self.configuration is None:
            # Force tight error bounds to avoid FHE noise non-determinism
            self.circuit = compiler.compile(inputset, configuration=fhe.Configuration(global_p_error=0.01))
        else:
            self.circuit = compiler.compile(inputset, configuration=self.configuration)
        assert self.circuit is not None, "Circuit must be compiled first"
        if self.simulate:
            return self.circuit.simulate(*args)
        return self.circuit.encrypt_run_decrypt(*args)

    def fit_encrypted(self, *args, **kwargs) -> Any:
        """Fits the model securely on encrypted training data.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            Any: The trained model.
        """
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
    
    Args:
        X_train (Any): Encrypted design matrix features.
        y_train (Any): Encrypted targets.
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        
    Returns:
        Any: Flattened sufficient statistics array.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.trainers import linear_regression_training
        
        # Calculate sufficient statistics securely
        stats = linear_regression_training(enc_X, enc_y, n_samples=100, n_features=5)
        ```
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
    
    Args:
        weight_scale (int): Scaling factor for weights.
        simulate (bool): Run circuits in simulation.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Example:
        ```python
        trainer = FHELinearRegressionTrainer(weight_scale=100)
        ```
    """

    def __init__(
        self,
        *,
        weight_scale: int = 100,
        simulate: bool = False,
        configuration: Optional[fhe.Configuration] = None,
    ) -> None:
        """Initialize FHELinearRegressionTrainer with the given weight_scale and trainer options."""
        super().__init__(simulate=simulate, configuration=configuration)
        self.weight_scale = validate_integer("weight_scale", weight_scale, minimum=1)

    def fit_encrypted(self, X_train: List[List[int]], y_train: List[int]) -> Any:
        """Fits the model securely on encrypted training data.
        
        Args:
            X_train (List[List[int]]): Encrypted training features.
            y_train (List[int]): Encrypted training targets.
            
        Returns:
            Any: The trained FHELinearRegression model.
        """
        n_samples = len(X_train)
        if n_samples == 0:
            raise ValueError("X_train must contain at least one sample")
        n_features = len(X_train[0])
        if any(len(row) != n_features for row in X_train):
            raise ValueError("every sample must have the same number of features")
        if len(y_train) != n_samples:
            raise ValueError("X_train and y_train must have the same length")

        def training_circuit(X_train: Any, y_train: Any) -> Any:
            """Compile and run the training circuit."""
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
        model = FHELinearRegression(weights, bias, output_scale=self.weight_scale)
        return model


def _gini(class_counts: List[int]) -> float:
    """Calculate the Gini impurity for a set of class counts.
    
    Args:
        class_counts (List[int]): Class counts.
        
    Returns:
        float: Gini impurity score.
        
    Example:
        ```python
        score = _gini([10, 10])
        ```
    """
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
        candidate_thresholds (List[List[int]]): Per feature, the public list of candidate
            thresholds (splits test ``feature >= threshold``). Keep these
            lists short — cost grows with nodes x candidates x samples.
        max_depth (int): Maximum tree depth (levels of splits).
        num_classes (int): Number of label classes (labels are 0..num_classes-1).
        min_samples_leaf (int): A split is rejected when either side would hold
            fewer samples than this.
        simulate (bool): Run circuits in simulation.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.
        
    Example:
        ```python
        trainer = FHEDecisionTreeTrainer(candidate_thresholds=[[1]])
        ```
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
        verbose: bool = False,
    ) -> None:
        if configuration is None:
            configuration = fhe.Configuration(
                global_p_error = 0.01,
                loop_parallelize = True,
                # dataflow_parallelize = True
            )

        super().__init__(simulate=simulate, configuration=configuration, verbose=verbose)
        """Initialize FHEDecisionTreeTrainer with candidate thresholds and tree hyperparameters."""
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

    def _sample_mask(self, X_train: Any, path: Any) -> Any:
        """Compute a per-sample binary mask for the samples that reach the given tree path.

        For each (feature, threshold, side) triple in ``path``, multiplies a comparison
        result into the running mask so that only samples satisfying every split are 1.

        Args:
            X_train (Any): The encrypted training feature matrix.
            path (Any): List of ``(feature_index, threshold, side)`` tuples defining the
                path from root to the current node. ``side`` is ``"ge"`` (left child) or
                ``"lt"`` (right child).

        Returns:
            Any: An encrypted per-sample binary array; element ``i`` is 1 iff sample ``i``
                reaches this node, 0 otherwise.
        """
        mask: Any = 1
        for feature, threshold, side in path:
            comparison = greater_equal(X_train[:, feature], threshold)
            if side == "ge":
                mask = mask * comparison
            else:
                mask = mask * (1 - comparison)
        return mask

    def _level_circuit(self, paths: Any, n_samples: int, with_candidates: bool) -> Any:
        candidates = self.candidate_thresholds
        num_classes = self.num_classes

        def level_counts(X_train: Any, y_train: Any) -> Any:
            """Return the count per level."""
            outputs = []
            all_flag_array = [equal(y_train, label) for label in range(num_classes)]

            for path in paths:
                mask_array = self._sample_mask(X_train, path)
                
                # Precompute mask * flag_array to save FHE multiplications!
                masked_flags = []
                for label in range(num_classes):
                    masked_flag = mask_array * all_flag_array[label]
                    masked_flags.append(masked_flag)
                    total = np.sum(masked_flag)
                    outputs.append(total)
                    
                if with_candidates:
                    for feature, thresholds in enumerate(candidates):
                        for threshold in thresholds:
                            goes_left_array = greater_equal(X_train[:, feature], threshold)
                            for label in range(num_classes):
                                # Instead of mask * flag * left, just do masked_flag * left
                                total = np.sum(masked_flags[label] * goes_left_array)
                                outputs.append(total)
            return fhe.array(outputs)

        return level_counts

    def _choose_split(self, node_counts: Any, candidate_counts: Any) -> Any:
        """Select the best (feature, threshold) split for a node using weighted Gini impurity.

        Args:
            node_counts (Any): Per-class sample counts for this node.
            candidate_counts (Any): List of ``((feature, threshold), left_counts)`` pairs
                produced by ``_level_circuit``.

        Returns:
            Optional[Tuple]: ``(score, feature, threshold)`` for the best valid split,
                or ``None`` if no valid split exists (e.g., all splits violate
                ``min_samples_leaf``).
        """
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

    def fit_encrypted(self, X_train: Union[np.ndarray, List[List[int]]], y_train: Union[np.ndarray,List[int]]) -> Any:
        """Fits the model securely on encrypted training data.
        
        Args:
            X_train (List[List[int]]): Encrypted training features.
            y_train (List[int]): Encrypted training targets.
            
        Returns:
            Any: The trained FHEDecisionTree model.
        """
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

        for depth in _get_progress_bar(range(self.max_depth + 1), "training tree...", self.verbose):
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
                num_samples = np.sum(node_counts)
                node_gini = _gini(node_counts)
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
                    parent[key] = majority  # type: ignore
                    continue

                _, feature, threshold = best
                subtree = {
                    "feature": feature,
                    "threshold": threshold,
                    "left": None,
                    "right": None,
                    "samples": num_samples,
                    "gini": node_gini
                }
                parent[key] = subtree  # type: ignore
                next_frontier.append(
                    {
                        "path": node["path"] + [(feature, threshold, "ge")],  # type: ignore
                        "attach": (subtree, "left"),
                    }
                )
                next_frontier.append(
                    {
                        "path": node["path"] + [(feature, threshold, "lt")],  # type: ignore
                        "attach": (subtree, "right"),
                    }
                )
            frontier = next_frontier
            gc.collect()

        return FHEDecisionTree(container["root"])

class FHERandomForestTrainer(FHETrainer):
    """Hybrid encrypted random forest training.
    
    Trains an ensemble of decision trees where each tree is built on a 
    bootstrap sample of the training data and uses a random subset of features.
    
    Args:
        n_estimators (int): The number of trees in the forest.
        candidate_thresholds (List[List[int]]): Per feature, the public list of candidate thresholds.
        max_depth (int): Maximum tree depth.
        num_classes (int): Number of label classes.
        min_samples_leaf (int): Minimum samples required to be at a leaf node.
        max_features (Optional[int]): Number of features to consider when looking for the best split.
        simulate (bool): Run circuits in simulation.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.
        verbose (bool): If True, outputs a progress bar during training. Defaults to False.
    """
    def __init__(
        self,
        n_estimators: int,
        candidate_thresholds: List[List[int]],
        *,
        max_depth: int = 2,
        num_classes: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        simulate: bool = False,
        configuration: Optional[fhe.Configuration] = None,
        verbose: bool = False,
    ) -> None:
        if configuration is None:
            configuration = fhe.Configuration(
                global_p_error = 0.01,
                loop_parallelize = True,
                # dataflow_parallelize = True
            )

        super().__init__(simulate=simulate, configuration=configuration, verbose=verbose)
        """Initialize FHERandomForestTrainer with the number of estimators and tree hyperparameters."""
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
        self.n_estimators = validate_integer("n_estimators", n_estimators, minimum=1)
        if max_features is not None:
            self.max_features = validate_integer("max_features", max_features, minimum=1)
        else:
            self.max_features = len(self.candidate_thresholds)

    def fit_encrypted(self, X_train: List[List[int]], y_train: List[int]) -> Any:
        """Fit the random forest securely on encrypted training data.

        Trains ``n_estimators`` decision trees, each on a bootstrap sample of the
        training data and using a random subset of ``max_features`` features.

        Args:
            X_train (List[List[int]]): Encrypted training features.
            y_train (List[int]): Encrypted training targets (integer class labels).

        Returns:
            FHERandomForest: The trained random forest model.
        """
        trained_trees = []
        n_samples = len(X_train)
        n_features = len(self.candidate_thresholds)
        
        for i in _get_progress_bar(range(self.n_estimators), "training forest...", self.verbose):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_subset = [X_train[idx] for idx in indices]
            y_subset = [y_train[idx] for idx in indices]

            tree_candidates = []
            if self.max_features < n_features:
                selected_features = np.random.choice(n_features, size=self.max_features, replace=False)
                for f_idx in range(n_features):
                    if f_idx in selected_features:
                        tree_candidates.append(self.candidate_thresholds[f_idx])
                    else:
                        tree_candidates.append([])
            else:
                tree_candidates = self.candidate_thresholds

            decision_tree = FHEDecisionTreeTrainer(candidate_thresholds=tree_candidates,
                                                    max_depth=self.max_depth,
                                                    num_classes=self.num_classes,
                                                    min_samples_leaf=self.min_samples_leaf,
                                                    simulate=self.simulate,
                                                    configuration=self.configuration)
            trained_model = decision_tree.fit_encrypted(X_subset, y_subset)
            trained_trees.append(trained_model.tree)
            print(f"Tree {i+1} structure: {trained_model.tree}\n")
            gc.collect()

        return FHERandomForest(trees=trained_trees)                                            

class FHEKMeansTrainer(FHETrainer):
    """Hybrid encrypted k-means training (fixed iterations).

    Each iteration compiles one circuit with the current public centroids:
    it assigns every encrypted sample to its nearest centroid and returns
    per-cluster sums and counts. Only those aggregates are decrypted; the
    centroid update (division) happens clear-side, and the new centroids
    become the next iteration's public constants.

    Args:
        initial_centroids (List[List[int]]): Public, data-independent starting centroids.
        min_value (int): The lower bound for the encrypted inputs, used to dimension the FHE circuit.
        max_value (int): The upper bound for the encrypted inputs, used to dimension the FHE circuit.
        n_iterations (int): Fixed number of Lloyd iterations.
        simulate (bool): Run circuits in simulation.
        configuration (Optional[fhe.Configuration]): Optional FHE compiler configuration object.

    Example:
        ```python
        trainer = FHEKMeansTrainer(initial_centroids=[[0, 0]], min_value=0, max_value=1)
        ```
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
        """Initialize FHEKMeansTrainer with initial centroids and value bounds."""
        super().__init__(simulate=simulate, configuration=configuration)
        if not initial_centroids:
            raise ValueError("initial_centroids must contain at least one centroid")
        self.initial_centroids = [list(centroid) for centroid in initial_centroids]
        self.minimum, self.maximum = validate_bounds(min_value, max_value)
        self.n_iterations = validate_integer("n_iterations", n_iterations, minimum=1)

    def _step_circuit(self, centroids: Any, n_samples: int, n_features: int, max_distance: int) -> Any:
        n_clusters = len(centroids)
        arg_min = make_argmin(n_clusters, 0, max_distance)

        def assign_and_aggregate(X_train: Any) -> Any:
            """Assign and aggregate values."""
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

    def fit_encrypted(self, X_train: List[List[int]]) -> Any:
        """Fits the model securely on encrypted training data.
        
        Args:
            X_train (List[List[int]]): Encrypted training features.
            
        Returns:
            Any: The trained FHEKMeans model.
        """
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
            gc.collect()

        return FHEKMeans(centroids, max_distance=max_distance)



class FHENaiveBayesTrainer:
    """Trainer for Encrypted Bernoulli Naive Bayes.
    
    Trains a Naive Bayes model over encrypted binary features and encrypted one-hot labels.
    The trainer automatically determines the optimal SCALE for probability quantization
    while maximizing the utilization of the specified `max_bit_width`.

    FHE Optimization (Zero-Centered Shifting):
    Log probabilities are always negative. This wastes half of the available integer 
    space in any bit width (e.g., the positive +1 to +127 range in 8-bit, or higher in 
    16-bit). This class mathematically shifts the log probabilities by finding the 
    `centered_max` and adding it entirely to the class Prior. This centers the final 
    circuit score perfectly in the middle of the available bit-width range, effectively 
    doubling the precision (SCALE) for any chosen bit-width (from 8-bit up to 16-bit).

    Laplace Smoothing:
    To prevent log(0) -infinity errors for features that never appeared in the training 
    set, Laplace smoothing is automatically applied to probability calculations.

    Example:
        ```python
        from concrete_fhe_toolkit.ml.classes import FHENaiveBayesTrainer
        
        trainer = FHENaiveBayesTrainer()
        # X_train must be binary (0 or 1), y_train must be one-hot encoded
        model = trainer.fit_encrypted(X_train, y_train_ohe, max_bit_width=10)
        ```
    """
    def __init__(self):
        """Initializes the FHENaiveBayesTrainer.
        
        Returns:
            None
        """
        self.circuit = None
        self.compiler = None

    def prepare_trainer(self, num_samples: int, num_features: int, num_classes: int, max_bit_width: Any=8, thresholds: Any=None) -> Any:
        """Prepares and compiles the trainer circuit.
        
        Args:
            num_samples (int): Number of training samples.
            num_features (int): Number of features per sample.
            num_classes (int): Number of classes.
            max_bit_width (Any, optional): Maximum bit width for the circuit. Defaults to 8.
            thresholds (Any, optional): Thresholds for training. Defaults to None.
            
        Returns:
            Any: The compiled FHE circuit.
            
        Example:
            ```python
            trainer.prepare_trainer(100, 10, 2)
            ```
        """
        if(max_bit_width > 16):
            raise ValueError("The maximum supported bit width is 16")
        if(max_bit_width > 8):
            warnings.warn("Higher bit widths(>8) may result in longer computation times.", UserWarning, stacklevel=2)
            
        # To prevent 'uint1' inference, use max possible value for bit width
        max_val = (2**(max_bit_width - 1)) - 1
        dummy_X = [[max_val]*num_features for _ in range(num_samples)]
        dummy_y = [[1]*num_classes for _ in range(num_samples)]
        
        if thresholds is not None:
            from .training import make_raw_naive_bayes_training
            circuit_logic = make_raw_naive_bayes_training(thresholds)
            self.compiler = fhe.Compiler(circuit_logic, {"X_train_raw": "encrypted", "y_train_one_hot": "encrypted"})
        else:
            self.compiler = fhe.Compiler(naive_bayes_training, {"X_train": "encrypted", "y_train_one_hot": "encrypted"})
            
        self.circuit = self.compiler.compile([(dummy_X, dummy_y)])
        return self.circuit

    def encrypt_data(self, X_train: Any, y_train: Any) -> Any:
        """Encrypts the training data.
        
        Args:
            X_train (Any): The training features.
            y_train (Any): The training labels.
            
        Returns:
            Any: The encrypted data tuple.
        """
        if self.circuit is None:
            raise ValueError("Circuit is not compiled. Call compile_trainer first.")
        
        return self.circuit.encrypt(X_train, y_train)

    def train_encrypted(self, encrypted_X: Any, encrypted_y: Any) -> Any:
        """Trains the model on the server using encrypted data.
        
        Args:
            encrypted_X (Any): Encrypted training features.
            encrypted_y (Any): Encrypted training labels.
            
        Returns:
            Any: Encrypted results of the training circuit.
        """
        if self.circuit is None:
            raise ValueError("Circuit is not compiled on server.")
            
        return self.circuit.run(encrypted_X, encrypted_y)

    def decrypt_and_finalize_model(self, encrypted_results: Any, max_bit_width: Any=8, * ,epsilon: Any = None) -> Any:
        """Decrypts the results and finalizes the Naive Bayes model.
        
        Args:
            encrypted_results (Any): The encrypted results from the server.
            max_bit_width (Any, optional): Maximum bit width for the circuit. Defaults to 8.
            epsilon (Any, optional): Epsilon for differential privacy. Defaults to None.
            
        Returns:
            Any: The finalized FHENaiveBayes model.
        """
        raw_feature_counts, priors = self.circuit.decrypt(*encrypted_results)
        if epsilon is not None:
            noisy_feature_counts = []
            for row in raw_feature_counts:
                noisy_feature_counts.append(dp_release(row, sensitivity=1, epsilon=epsilon/2))
            priors_noisy = dp_release(priors, sensitivity=1, epsilon=epsilon/2)
            return self._finalize_model(noisy_feature_counts, priors_noisy, max_bit_width)

        return self._finalize_model(raw_feature_counts, priors, max_bit_width)

    @staticmethod
    def _finalize_model(raw_feature_counts: Any, priors: Any, max_bit_width: Any) -> Any:
        formatted_tables = []
        formatted_priors = []
        total_samples = sum(priors)

        max_abs_score = 0.0
        num_features = len(raw_feature_counts[0])
        for prior in priors:
            class_total = int(prior)
            prior_prob = class_total / total_samples
            min_feature_prob = 1 / (class_total + 2)
            
            score_abs = abs(math.log(prior_prob)) + (num_features * abs(math.log(min_feature_prob)))
            max_abs_score = max(max_abs_score, score_abs)

        centered_max = max_abs_score / 2.0
        max_target_int = (2**(max_bit_width - 1)) -1
        
        SCALE = max(1,int(max_target_int / centered_max))

        for c, class_feature_counts in enumerate(raw_feature_counts):
            class_total = int(priors[c])
            
            # Prior Log Prob
            prior_prob = class_total / total_samples
            unscaled_prior = math.log(prior_prob) + centered_max 
            formatted_priors.append(int(round(unscaled_prior * SCALE)))
            
            class_tables = []
            for count_of_ones in class_feature_counts:
                count_of_ones = int(count_of_ones)
                count_of_zeros = class_total - count_of_ones
                
                # Laplace smoothed probabilities: (count + 1) / (class_total + num_classes)
                prob_0 = (count_of_zeros + 1) / (class_total + 2)
                prob_1 = (count_of_ones + 1) / (class_total + 2)
                
                log_prob_0 = int(round(math.log(prob_0) * SCALE))
                log_prob_1 = int(round(math.log(prob_1) * SCALE))
                
                class_tables.append([log_prob_0, log_prob_1])
            formatted_tables.append(class_tables)
            
        model = FHENaiveBayes(formatted_tables, formatted_priors)
        model.scale = SCALE
        return model

    def fit_encrypted(self, X_train: Any, y_train: Any, * ,max_bit_width: Any = 8, thresholds: Any=None, epsilon: Any=None) -> Any:
        """Fits the model securely on encrypted training data.
        
        Args:
            X_train (Any): The training features.
            y_train (Any): The training labels.
            max_bit_width (Any, optional): Maximum bit width for the circuit. Defaults to 8.
            thresholds (Any, optional): Thresholds for training. Defaults to None.
            epsilon (Any, optional): Epsilon for differential privacy. Defaults to None.
            
        Returns:
            Any: The trained FHENaiveBayes model.
        """
        if(max_bit_width > 16):
            raise ValueError("The maximum supported bit width is 16")
        if(max_bit_width > 8):
            warnings.warn("Higher bit widths(>8) may result in longer computation times.", UserWarning, stacklevel=2)
            
        if thresholds is not None:
            from .training import make_raw_naive_bayes_training
            circuit_logic = make_raw_naive_bayes_training(thresholds)
            self.compiler = fhe.Compiler(circuit_logic, {"X_train_raw": "encrypted", "y_train_one_hot": "encrypted"})
        else:
            self.compiler = fhe.Compiler(naive_bayes_training,{"X_train": "encrypted", "y_train_one_hot": "encrypted"})
            
        self.circuit = self.compiler.compile([(X_train, y_train)])

        # The circuit returns raw counts (feature_counts, class_counts)
        raw_feature_counts, priors = self.circuit.encrypt_run_decrypt(X_train, y_train)
        if epsilon is not None:
            noisy_feature_counts = []
            for row in raw_feature_counts:
                noisy_feature_counts.append(dp_release(row, sensitivity=1, epsilon=epsilon/2))
            priors_noisy = dp_release(priors, sensitivity=1, epsilon=epsilon/2)
            return self._finalize_model(noisy_feature_counts, priors_noisy, max_bit_width)

        return self._finalize_model(raw_feature_counts, priors, max_bit_width)
