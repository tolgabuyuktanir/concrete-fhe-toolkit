from .._compat import fhe
import math
from concrete_fhe_toolkit.ml import (
    logistic_regression_inference, linear_regression_inference,
    decision_tree_inference, pca_inference, cnn_inference,
    random_forest_inference, xgboost_inference, svm_inference,
    knn_inference, naive_bayes_inference, mlp_inference, naive_bayes_training
    )
import warnings

class FHEModel:
    """Base class for all FHE machine learning models.
    
    This class handles the boilerplate of compiling the FHE circuit and 
    running predictions. Child classes should override the `_circuit_logic`
    method to provide the specific model's mathematical logic.
    """
    def __init__(self):
        self.circuit = None

    def _circuit_logic(self,features):
        raise NotImplementedError("This function can be used in inherited classes")
        
    def _batch_circuit_logic(self, features_batch):
        return fhe.array([self._circuit_logic(sample) for sample in features_batch])

    def compile(self, inputset, batch_size: int = 16):
        self.batch_size = batch_size
        self.compiler = fhe.Compiler(self._batch_circuit_logic,{"features_batch": "encrypted"})
        self.circuit = self.compiler.compile(inputset)    

    def predict(self, features):
        """Encrypt one sample, run the compiled circuit, and decrypt the result.
        This is a convenience wrapper that calls `predict_many`.

        Example:
            ```python
            model = FHELogisticRegression(weights=[3, 2], bias=-7)
            model.compile(inputset=[[[0, 0]] * 16], batch_size=16)
            print(model.predict([4, 1]))  # 1
            ```
        """
        if self.circuit is None:
            raise ValueError("The model should be compiled before prediction")
        return self.predict_many([features])[0]

    def simulate(self, features):
        """Run one prediction in Concrete's simulator (fast, no key generation).

        Simulation is for prototyping and tests only — inputs are NOT
        protected. The model must be compiled first.

        Example:
            ```python
            model.compile(inputset=[[[0, 0]] * 16], batch_size=16)
            print(model.simulate([4, 1]))  # same output, no keygen
            ```
        """
        if self.circuit is None:
            raise ValueError("The model should be compiled before prediction")
        return self.simulate_many([features])[0]

    def predict_many(self, samples):
        """Predict a batch of samples with one compiled circuit (one key set).

        The circuit is compiled for a specific `batch_size`. If the number of 
        samples is not a multiple of `batch_size`, dummy samples are automatically 
        padded and then removed from the final result.

        Example:
            ```python
            model.compile(inputset=[[[0, 0]] * 16], batch_size=16)
            predictions = model.predict_many([[4, 1], [0, 2], [5, 5]])
            ```
        """
        original_samples_length = len(samples)
        num_of_full_batches = original_samples_length // self.batch_size
        num_of_padding = self.batch_size - original_samples_length % self.batch_size
        
        padded_samples = list(samples)
        if num_of_padding != self.batch_size:
            num_of_full_batches += 1
            for _ in range(num_of_padding):
                padded_samples.append([0] * len(padded_samples[0]))

        predictions = []
        start = 0
        for _ in range(num_of_full_batches):
            batch = padded_samples[start:start+self.batch_size]
            predictions.extend(self.circuit.encrypt_run_decrypt(batch))
            start += self.batch_size

        return predictions[:original_samples_length]    

        

    def simulate_many(self, samples):
        """Simulate a batch of samples (fast counterpart of ``predict_many``)."""
        original_samples_length = len(samples)
        num_of_full_batches = original_samples_length // self.batch_size
        num_of_padding = self.batch_size - original_samples_length % self.batch_size
        
        padded_samples = list(samples)
        if num_of_padding != self.batch_size:
            num_of_full_batches += 1
            for _ in range(num_of_padding):
                padded_samples.append([0] * len(padded_samples[0]))

        predictions = []
        start = 0
        for _ in range(num_of_full_batches):
            batch = padded_samples[start:start+self.batch_size]
            predictions.extend(self.circuit.simulate(batch))
            start += self.batch_size

        return predictions[:original_samples_length]


class FHELogisticRegression(FHEModel):
    """Encrypted Logistic Regression Inference Model.
    
    Evaluates a logistic regression model over encrypted features.
    
    Args:
        weights: The public list of weights for the regression.
        bias: The public bias term.
        
    Example:
        ```python
        model = FHELogisticRegression(weights=[3, 2], bias=-7)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, weights, bias):
        super().__init__()
        self.weights = weights
        self.bias = bias

    def _circuit_logic(self, features):
        return logistic_regression_inference(self.weights, self.bias, features)
        

class FHELinearRegression(FHEModel):
    """Encrypted Linear Regression Inference Model.
    
    Evaluates a linear regression model over encrypted features.
    
    Args:
        weights: The public list of weights for the regression.
        bias: The public bias term.
        
    Example:
        ```python
        model = FHELinearRegression(weights=[10, -5], bias=2)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, weights, bias):
        super().__init__()
        self.weights = weights
        self.bias = bias

    def _circuit_logic(self, features):
        return linear_regression_inference(self.weights, self.bias, features)


class FHEDecisionTree(FHEModel):
    """Encrypted Decision Tree Inference Model.
    
    Evaluates a decision tree over encrypted features.
    
    Args:
        tree: The public tree representation (dict format).
        
    Example:
        ```python
        model = FHEDecisionTree(tree=my_parsed_tree)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self,tree):
        super().__init__()
        self.tree = tree

    def _circuit_logic(self, features):
        return decision_tree_inference(features, self.tree)


class FHEPCA(FHEModel):
    """Encrypted Principal Component Analysis (PCA) Inference Model.
    
    Applies PCA dimensionality reduction to encrypted features.
    
    Args:
        means: The public list of feature means.
        components: The public principal components matrix.
        
    Example:
        ```python
        model = FHEPCA(means=[0.5, 0.5], components=[[1, 0], [0, 1]])
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, means, components):
        super().__init__()
        self.means = means
        self.components = components

    def _circuit_logic(self, features):
        return pca_inference(features, self.means, self.components)


class FHECNN(FHEModel):
    """Encrypted Convolutional Neural Network (CNN) Inference Model.
    
    Applies a 2D convolutional layer to an encrypted image.
    
    Args:
        filters: The public 2D or 3D list of convolutional filters.
        bias: The public list of bias terms for the filters.
        
    Example:
        ```python
        model = FHECNN(filters=my_conv_filters, bias=my_conv_bias)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """        
    def __init__(self, filters, bias):
        super().__init__()
        self.filters = filters
        self.bias = bias

    def _circuit_logic(self, features):
        return cnn_inference(self.filters, self.bias, image = features)


class FHERandomForest(FHEModel):
    """Encrypted Random Forest Inference Model.
    
    Evaluates an ensemble of decision trees over encrypted features.
    
    Args:
        trees: The public list of tree representations (dict format).
        
    Example:
        ```python
        model = FHERandomForest(trees=[tree1, tree2, tree3])
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, trees):
        super().__init__()
        self.trees = trees

    def _circuit_logic(self, features):
        return random_forest_inference(features,self.trees)    


class FHEXGBoost(FHEModel):
    """Encrypted XGBoost Inference Model.
    
    Evaluates a gradient boosting tree ensemble over encrypted features.
    
    Args:
        trees: The public list of tree representations (dict format).
        
    Example:
        ```python
        model = FHEXGBoost(trees=my_xgb_trees)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, trees):
        super().__init__() 
        self.trees = trees   

    def _circuit_logic(self, features):
        return xgboost_inference(features, self.trees)    

class FHESVM(FHEModel):
    """Encrypted Support Vector Machine (SVM) Inference Model.
    
    Evaluates a linear SVM over encrypted features.
    
    Args:
        weights: The public list of support vector weights (dual_coef/coef).
        bias: The public bias or intercept term.
        
    Example:
        ```python
        model = FHESVM(weights=[0.5, -1.2], bias=0.1)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, weights, bias):
        super().__init__()
        self.weights = weights
        self.bias = bias

    def _circuit_logic(self, features):
        return svm_inference(self.weights, self.bias, features)

class FHEKNN(FHEModel):
    """Encrypted K-Nearest Neighbors (KNN) Inference Model.
    
    Evaluates KNN distances between encrypted features and plaintext training data.
    
    Args:
        X_train: The public training dataset features.
        y_train: The public training dataset labels.
        k: The number of nearest neighbors to consider.
        
    Example:
        ```python
        model = FHEKNN(X_train=public_X, y_train=public_y, k=3)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, X_train, y_train, k):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def _circuit_logic(self, features):
        return knn_inference(features, self.X_train, self.y_train, k=self.k)


class FHEMLP(FHEModel):    
    """Encrypted Multi-Layer Perceptron (Dense) Inference Model.
    
    Evaluates a sequence of dense neural network layers over encrypted features.
    
    Args:
        mlp_layers: The public list of layer tuples, where each tuple is `(weights, bias)`.
        
    Example:
        ```python
        model = FHEMLP(mlp_layers=[(w1, b1), (w2, b2)])
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, mlp_layers):
        super().__init__()
        self.mlp_layers = mlp_layers

    def _circuit_logic(self, features):
        return mlp_inference(features,self.mlp_layers)    


class FHENaiveBayes(FHEModel):
    """Encrypted Naive Bayes Inference Model.
    
    Evaluates a Bernoulli Naive Bayes classifier over encrypted binary features.
    The model aggregates log probabilities using FHE lookup tables and returns 
    the class index with the highest score using an encrypted argmax reduction.
    
    Args:
        log_prob_tables: The public quantized feature log probabilities.
        priors: The public quantized class priors.

    Example:
        ```python
        # Assuming `model` was returned by FHENaiveBayesTrainer.fit_encrypted
        model.compile(dummy_inputset, batch_size=1)
        prediction = model.predict([1, 0, 1, 1])
        ```
    """
    def __init__(self, log_prob_tables, priors):
        super().__init__()
        self.log_prob_tables = log_prob_tables
        self.priors = priors

    def _circuit_logic(self, features):
        return naive_bayes_inference(features,self.log_prob_tables,self.priors)


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
        self.circuit = None
        self.compiler = None

    def compile_trainer(self, num_samples: int, num_features: int, num_classes: int, max_bit_width=8, thresholds=None):
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

    def encrypt_data(self, X_train, y_train):
        if self.circuit is None:
            raise ValueError("Circuit is not compiled. Call compile_trainer first.")
        
        return self.circuit.encrypt(X_train, y_train)

    def train_on_server(self, encrypted_X, encrypted_y):
        if self.circuit is None:
            raise ValueError("Circuit is not compiled on server.")
            
        return self.circuit.run(encrypted_X, encrypted_y)

    def decrypt_and_finalize(self, encrypted_results, max_bit_width=8):
        raw_feature_counts, priors = self.circuit.decrypt(*encrypted_results)
        
        formatted_tables = []
        formatted_priors = []
        total_samples = sum(priors)

        max_abs_score = 0
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

    def fit_encrypted(self, X_train, y_train, * ,max_bit_width = 8, thresholds=None):
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
        
        # naive_bayes_inference expects SCALED LOG PROBABILITIES, not raw counts!
        # We will apply Laplace smoothing and dynamically scale the log probabilities.
        formatted_tables = []
        formatted_priors = []
        total_samples = sum(priors)

        max_abs_score = 0
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


class FHEKMeans(FHEModel):
    """Encrypted K-Means Inference Model (cluster assignment).

    Assigns encrypted samples to the nearest public centroid. Train with
    ``FHEKMeansTrainer.fit_encrypted`` or provide centroids directly.

    Args:
        centroids (list): The public list of cluster centroids.
        max_distance (int): Upper bound on the squared distance from any
            sample to any centroid (sizes the argmin reduction).
            
    Example:
        ```python
        model = FHEKMeans(centroids=[[0,0], [10,10]], max_distance=200)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """

    def __init__(self, centroids, *, max_distance):
        super().__init__()
        from .models import nearest_centroid_inference

        self._nearest_centroid_inference = nearest_centroid_inference
        self.centroids = [list(centroid) for centroid in centroids]
        self.max_distance = max_distance

    def _circuit_logic(self, features):
        return self._nearest_centroid_inference(
            features,
            self.centroids,
            max_distance=self.max_distance,
        )

