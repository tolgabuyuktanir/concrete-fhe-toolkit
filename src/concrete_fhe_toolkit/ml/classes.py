from .._compat import fhe
from concrete_fhe_toolkit.ml import (
    logistic_regression_inference, linear_regression_inference,
    decision_tree_inference, pca_inference, cnn_inference,
    random_forest_inference, xgboost_inference, svm_inference,
    knn_inference, naive_bayes_inference, mlp_inference
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

    def compile(self,inputset):
        self.compiler = fhe.Compiler(self._circuit_logic,{"features": "encrypted"})
        self.circuit = self.compiler.compile(inputset)    

    def predict(self,features):
        if self.circuit is None:
            raise ValueError("The model should be compiled before prediction")
        return self.circuit.encrypt_run_decrypt(features)  


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
        return knn_inference(features, self.X_train, self.y_train, self.k)     


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