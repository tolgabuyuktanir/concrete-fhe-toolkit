from .._compat import fhe
import math
from concrete_fhe_toolkit.ml import (
    logistic_regression_inference, linear_regression_inference,
    decision_tree_inference, pca_inference, cnn_inference,
    random_forest_inference, xgboost_inference, svm_inference,
    knn_inference, naive_bayes_inference, mlp_inference, naive_bayes_training
    )

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
    """
    def __init__(self, mlp_layers):
        super().__init__()
        self.mlp_layers = mlp_layers

    def _circuit_logic(self, features):
        return mlp_inference(features,self.mlp_layers)    


class FHENaiveBayes(FHEModel):
    """Encrypted Naive Bayes Inference Model.
    
    Evaluates a Naive Bayes classifier over encrypted features.
    
    Args:
        log_prob_tables: The public feature log probabilities.
        priors: The public class priors.
    """
    def __init__(self, log_prob_tables, priors):
        super().__init__()
        self.log_prob_tables = log_prob_tables
        self.priors = priors

    def _circuit_logic(self, features):
        return naive_bayes_inference(features,self.log_prob_tables,self.priors)


class FHENaiveBayesTrainer:
    """Trainer for Encrypted Naive Bayes.
    
    Trains a Naive Bayes model over encrypted features and encrypted one-hot labels.
    """
    def __init__(self):
        self.circuit = None
        self.compiler = None

    def fit_encrypted(self, X_train, y_train, * ,max_bit_width = 8):
        if(max_bit_width > 16):
            raise ValueError("The maximum supported bit width is 16")
        self.compiler = fhe.Compiler(naive_bayes_training,{"X_train": "encrypted", "y_train_one_hot": "encrypted"})
        self.circuit = self.compiler.compile([(X_train, y_train)])

        # The circuit returns raw counts (feature_counts, class_counts)
        raw_feature_counts, priors = self.circuit.encrypt_run_decrypt(X_train, y_train)
        
        # naive_bayes_inference expects SCALED LOG PROBABILITIES, not raw counts!
        # We will apply Laplace smoothing and scale by 1000.
        formatted_tables = []
        formatted_priors = []
        total_samples = sum(priors)

        max_abs_score = 0
        num_features = len(raw_feature_counts[0]) # Özellik sayısı
        for prior in priors:
            class_total = int(prior) # Sınıfın adeti
            
            # 1. Sınıfın gelme olasılığı (Prior Prob)
            prior_prob = class_total / total_samples
            
            # 2. O sınıf için Laplace'ın verebileceği en düşük olasılık
            min_feature_prob = 1 / (class_total + 2)
            
            # En ekstrem skoru topla ve maksimumla karşılaştır
            score_abs = abs(math.log(prior_prob)) + (num_features * abs(math.log(min_feature_prob)))
            max_abs_score = max(max_abs_score, score_abs)

        max_target_int = (2**(max_bit_width - 1)) -1
        
        SCALE = max(1,int(max_target_int / max_abs_score))

        for c, class_feature_counts in enumerate(raw_feature_counts):
            class_total = int(priors[c])
            
            # Prior Log Prob
            prior_prob = class_total / total_samples
            formatted_priors.append(int(math.log(prior_prob) * SCALE))
            
            class_tables = []
            for count_of_ones in class_feature_counts:
                count_of_ones = int(count_of_ones)
                count_of_zeros = class_total - count_of_ones
                
                # Laplace smoothed probabilities: (count + 1) / (class_total + num_classes)
                prob_0 = (count_of_zeros + 1) / (class_total + 2)
                prob_1 = (count_of_ones + 1) / (class_total + 2)
                
                log_prob_0 = int(math.log(prob_0) * SCALE)
                log_prob_1 = int(math.log(prob_1) * SCALE)
                
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

