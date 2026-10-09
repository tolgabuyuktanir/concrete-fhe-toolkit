"""Core data structures and definitions for encrypted ML."""

from .._compat import fhe
import numpy as np
from typing import Any, Optional
from .._utils import validate_integer
from concrete_fhe_toolkit.ml import (
    logistic_regression_inference, linear_regression_inference,
    decision_tree_inference, pca_inference, cnn_inference,
    random_forest_inference, xgboost_inference, svm_inference,
    knn_inference, naive_bayes_inference, mlp_inference
    )
from .utils import _get_progress_bar

class FHEModel:
    """Base class for models with single-sample or fixed-batch circuits.
    
    Example:
        ```python
        model = FHEModel()
        ```
    """

    def __init__(self):
        """Initializes the FHEModel.
        
        Returns:
            None
        """
        self.circuit = None
        self.batch_size = 1
        self._batched = False
        self._sample_shape = None

    def _repr_html_(self):
        model_name = self.__class__.__name__
        
        # Status configurations
        c_color = "#10B981" if self.circuit is not None else "#EF4444"
        c_text = "Compiled" if self.circuit is not None else "Not Compiled"
        c_icon = "✓" if self.circuit is not None else "✕"
        
        b_color = "#3B82F6" if self._batched else "#6B7280"
        b_text = "Batched" if self._batched else "Not Batched"

        attribute_name = ""
        attribute_value = None
        if hasattr(self, "trees"):
            attribute_name = "Number of Trees"
            attribute_value =  len(self.trees)
        elif hasattr(self, "filters"):
            attribute_name = "Number of Filters"
            attribute_value = len(self.filters)
        elif hasattr(self, "weights"):
            attribute_name = "Number of Weights"
            attribute_value = len(self.weights)
        elif hasattr(self, "k"):
            attribute_name = "Number of Neighbors (k)"
            attribute_value = self.k
        elif hasattr(self, "mlp_layers"):
            attribute_name = "Number of Layers"
            attribute_value = len(self.mlp_layers)
        elif hasattr(self, "centroids"):
            attribute_name = "Number of Centroids"
            attribute_value = len(self.centroids)
        elif hasattr(self, "components"):
            attribute_name = "Number of Components"
            attribute_value = len(self.components) 
        elif hasattr(self, "priors"):
            attribute_name = "Number of Classes"
            attribute_value = len(self.priors)
        elif hasattr(self, "tree"):
            attribute_name = "Tree Depth"

            def tree_depth(node):
                if not isinstance(node, dict):
                    return 0

                depth = 1 + max(tree_depth(node.get("left")), tree_depth(node.get("right")))
                return depth

            attribute_value = tree_depth(self.tree)                          

        extra_html = ""
        if attribute_name != "":
            extra_html = f"""
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px dashed #E5E7EB; padding-top: 12px; margin-top: 4px;">
                    <span style="color: #4B5563; font-size: 14px; font-weight: 500;">{attribute_name}</span>
                    <span style="color: #111827; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 14px; font-weight: 600; background-color: #F3F4F6; padding: 2px 8px; border-radius: 6px;">
                        {attribute_value}
                    </span>
                </div>
            """

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; border: 1px solid #E5E7EB; border-radius: 12px; padding: 20px; background: linear-gradient(to bottom right, #ffffff, #f8fafc); width: 320px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);">
            <div style="display: flex; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #E5E7EB; padding-bottom: 12px;">
                <div style="background-color: #1F2937; color: white; border-radius: 8px; padding: 6px 10px; font-size: 16px; margin-right: 12px;">🤖</div>
                <h3 style="color: #111827; margin: 0; font-size: 18px; font-weight: 600;">{model_name}</h3>
            </div>
            
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #4B5563; font-size: 14px; font-weight: 500;">Compilation</span>
                    <span style="background-color: {c_color}15; color: {c_color}; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; display: flex; align-items: center; gap: 4px;">
                        <span>{c_icon}</span> {c_text}
                    </span>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #4B5563; font-size: 14px; font-weight: 500;">Batch State</span>
                    <span style="background-color: {b_color}15; color: {b_color}; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600;">
                        {b_text}
                    </span>
                </div>
                
                <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px dashed #E5E7EB; padding-top: 12px; margin-top: 4px;">
                    <span style="color: #4B5563; font-size: 14px; font-weight: 500;">Batch Size</span>
                    <span style="color: #111827; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 14px; font-weight: 600; background-color: #F3F4F6; padding: 2px 8px; border-radius: 6px;">
                        {self.batch_size}
                    </span>
                </div>
                {extra_html}
            </div>
        </div>
        """
        return html

    def _circuit_logic(self, features: Any) -> Any:
        """Core circuit logic to be implemented by subclasses.
        
        Args:
            features (Any): The encrypted 2D feature matrix containing input samples.
            
        Returns:
            Any: The result of the circuit computation.
        """
        raise NotImplementedError("Subclasses must implement _circuit_logic")

    def _batch_circuit_logic(self, features_batch: Any) -> Any:
        """Executes the circuit logic on a batch of features.
        
        Args:
            features_batch (Any): A batch of input features.
            
        Returns:
            Any: An array of results from the circuit logic.
        """
        return fhe.array([self._circuit_logic(sample) for sample in features_batch])

    def _single_circuit_logic(self, features: Any) -> Any:
        """Executes the circuit logic on a single sample.
        
        Args:
            features (Any): The encrypted 2D feature matrix containing input samples.
            
        Returns:
            Any: The result of the circuit logic.
        """
        result = self._circuit_logic(features)
        return fhe.array(result) if isinstance(result, (list, tuple)) else result

    @staticmethod
    def _integer_array(value: Any) -> Any:
        """Converts a value to an integer numpy array safely.
        
        Args:
            value (Any): The encrypted value to process or validate.
            
        Returns:
            Any: The resulting integer numpy array.
        """
        array = np.asarray(value)
        if array.size == 0:
            raise ValueError("samples must not be empty")
        if array.dtype.kind not in "iu":
            raise TypeError("samples must contain integers")
        if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
            raise ValueError("samples must fit in signed 64-bit integers")
        return array.astype(np.int64)

    def compile(self, inputset: Any, batch_size: Any=None, *, inputset_is_batched: Any=None,
                configuration: Any=None) -> Any:
        """Compile representative integer samples.

        By default, each inputset item is one sample and the circuit accepts
        one sample, including image/tensor samples. ``predict_many`` reuses it.

        For compatibility, an explicit ``batch_size`` means each inputset
        item is an already assembled batch. Pass ``inputset_is_batched=False``
        with ``batch_size`` to build a batch circuit from individual samples.
        ``inputset_is_batched=True`` can infer batch size from the first item.
        All batches must have the same shape and size. Runtime samples must
        remain within the public bounds represented by the inputset.

        Args:
            inputset (Any): The calibration dataset.
            batch_size (Any, optional): The batch size. Defaults to None.
            inputset_is_batched (Any, optional): Whether the inputset is batched. Defaults to None.
            configuration (Any, optional): Optional FHE compiler configuration object.
            
        Returns:
            Any: The compiled circuit or None.

        Example:
            ```python
            model.compile([[0, 0], [5, 5]])
            model.compile([[0, 0], [5, 5]], batch_size=4,
                          inputset_is_batched=False)
            model.compile([[[0, 0]] * 4, [[5, 5]] * 4], batch_size=4)
            ```
        """
        if inputset_is_batched is not None and not isinstance(inputset_is_batched, bool):
            raise TypeError("inputset_is_batched must be a boolean or None")
        size = None if batch_size is None else validate_integer("batch_size", batch_size, 1)
        prebatched = size is not None if inputset_is_batched is None else inputset_is_batched
        items = [self._integer_array(item) for item in inputset]
        if not items:
            raise ValueError("inputset must contain at least one sample")
        shape = items[0].shape
        if any(item.shape != shape for item in items):
            raise ValueError("all inputset items must have the same shape")
        batched = prebatched or size is not None
        if prebatched:
            if not shape:
                raise ValueError("each inputset item must contain a batch")
            size = shape[0] if size is None else size
            if shape[0] != size:
                raise ValueError("inputset batch dimension must equal batch_size")
            sample_shape = shape[1:]
            calibration = items
        elif batched:
            sample_shape = shape
            # Every lane observes every calibration sample and its bounds.
            assert size is not None
            calibration = [np.stack([item] * size) for item in items]
        else:
            size = 1
            sample_shape = shape
            calibration = items
        function = self._batch_circuit_logic if batched else self._single_circuit_logic
        parameter = "features_batch" if batched else "features"
        compiler = fhe.Compiler(function, {parameter: "encrypted"})
        if configuration is None:
            # Force tight error bounds to avoid FHE noise non-determinism
            circuit = compiler.compile(calibration, configuration=fhe.Configuration(global_p_error=0.01))
        else:
            circuit = compiler.compile(calibration, configuration=configuration)
        # Commit state only after successful compilation.
        self.compiler = compiler
        self.circuit = circuit
        self.batch_size = size
        self._batched = batched
        self._sample_shape = sample_shape
        return self

    def predict(self, features: Any) -> Any:
        """Encrypt one sample, evaluate it, and decrypt its prediction.
        
        Args:
            features (Any): The encrypted 2D feature matrix containing input samples.
            
        Returns:
            Any: The prediction result.
        """
        return self.predict_many([features])[0]

    def simulate(self, features: Any) -> Any:
        """Simulate one sample without encryption; compile the model first.
        
        Args:
            features (Any): The encrypted 2D feature matrix containing input samples.
            
        Returns:
            Any: The simulation result.
        """
        return self.simulate_many([features])[0]

    def _run_many(self, samples: Any, method: Any, verbose: bool=False) -> Any:
        """Helper to run a method over multiple samples.
        
        Args:
            samples (Any): The list of samples.
            method (Any): The name of the method to call on the circuit.
            
        Returns:
            Any: A list of results.
        """
        if self.circuit is None:
            raise ValueError("The model should be compiled before prediction")
        items = [self._integer_array(sample) for sample in samples]
        if not items:
            return []
        expected_shape = self._sample_shape if self._sample_shape is not None else items[0].shape
        if any(item.shape != expected_shape for item in items):
            raise ValueError("sample shape does not match the compiled model")
        run = getattr(self.circuit, method)
        if not self._batched:
            return [run(item) for item in _get_progress_bar(items, "predicting...", verbose)]
        results = []
        for start in _get_progress_bar(range(0, len(items), self.batch_size), "predicting...", verbose):
            batch = items[start:start + self.batch_size]
            count = len(batch)
            # Repeat an in-domain sample, preserving arbitrary tensor shape.
            batch = batch + [batch[-1]] * (self.batch_size - count)
            results.extend(run(np.stack(batch))[:count])
        return results

    def predict_many(self, samples: Any, verbose: bool=False) -> Any:
        """Predict samples using the compiled circuit and its existing keys.

        Partial batches repeat the final sample for padding; padded predictions
        are discarded. Single-sample circuits execute once per input sample.
        
        Args:
            samples (Any): The list of samples to predict on.
            
        Returns:
            Any: The list of predictions.
        """
        return self._run_many(samples, "encrypt_run_decrypt", verbose)

    def simulate_many(self, samples: Any, verbose: bool = False) -> Any:
        """Simulate samples with the same batching rules as ``predict_many``.
        
        Args:
            samples (Any): The list of samples to simulate on.
            
        Returns:
            Any: The list of simulation results.
        """
        return self._run_many(samples, "simulate", verbose)


class FHELogisticRegression(FHEModel):
    """Encrypted Logistic Regression Inference Model.
    
    Evaluates a logistic regression model over encrypted features.
    
    Args:
        weights: The pre-trained cleartext model weights used for inference.
        bias: The pre-trained cleartext model bias used for inference.
        
    Example:
        ```python
        model = FHELogisticRegression(weights=[3, 2], bias=-7)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, weights: Any, bias: Any, *, input_scale: Any = 1, output_scale: Any = 1):
        super().__init__()
        """Initialize the object."""
        self.weights = weights
        self.bias = bias
        self.input_scale = input_scale
        self.output_scale = output_scale

    def _circuit_logic(self, features: Any) -> Any:
        return logistic_regression_inference(self.weights, self.bias, features)
        

class FHELinearRegression(FHEModel):
    """Encrypted Linear Regression Inference Model.
    
    Evaluates a linear regression model over encrypted features.
    
    Args:
        weights: The pre-trained cleartext model weights used for inference.
        bias: The pre-trained cleartext model bias used for inference.
        
    Example:
        ```python
        model = FHELinearRegression(weights=[10, -5], bias=2)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, weights: Any, bias: Any, *, input_scale: Any = 1, output_scale: Any = 1):
        super().__init__()
        """Initialize the object."""
        self.weights = weights
        self.bias = bias
        self.input_scale = input_scale
        self.output_scale = output_scale

    def _circuit_logic(self, features: Any) -> Any:
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
    def __init__(self,tree: Any) -> None:
        """Initialize the encrypted decision tree model.
        
        Args:
            tree (Any): The dictionary representation of the trained tree.
        """
        super().__init__()
        self.tree = tree

    def _tree_depth(self, node):
        if not isinstance(node, dict):
            return 0 #leaf
        left_depth = self._tree_depth(node["left"])
        right_depth = self._tree_depth(node["right"])
        return 1 + max(left_depth, right_depth)

    def _flatten_tree(self):
        depth_tree = self._tree_depth(self.tree)
        enc_thresholds = [0,] * (2**depth_tree-1)
        enc_feature_indices = [0,] * (2**depth_tree-1)
        enc_leaf_values = [0,] * (2**depth_tree)
        
        leaf_start_idx = 2**depth_tree-1
        def recurse(node, idx):
            if isinstance(node, dict):
                enc_thresholds[idx] = node["threshold"]
                enc_feature_indices[idx] = node["feature"]
                recurse(node["left"], 2 * idx + 1)
                recurse(node["right"], 2 * idx + 2)
            else:
                if idx >= leaf_start_idx:
                    enc_leaf_values[idx - leaf_start_idx] = node
                else:
                    enc_thresholds[idx] = 0
                    enc_feature_indices[idx] = 0
                    recurse(node, 2 * idx + 1)
                    recurse(node, 2 * idx + 2) 
        recurse(self.tree, 0)

        return enc_thresholds, enc_feature_indices, enc_leaf_values

    def compile(self, inputset: Any, batch_size: Any=None, *, inputset_is_batched: Any=None, configuration: Any=None) -> Any:
        if inputset_is_batched is not None and not isinstance(inputset_is_batched, bool):
            raise TypeError("inputset_is_batched must be a boolean or None")
        size = None if batch_size is None else validate_integer("batch_size", batch_size, 1)
        prebatched = size is not None if inputset_is_batched is None else inputset_is_batched
        
        items = [self._integer_array(item) for item in inputset]
        if not items:
            raise ValueError("inputset must contain at least one sample")
        shape = items[0].shape
        if any(item.shape != shape for item in items):
            raise ValueError("all inputset items must have the same shape")
            
        batched = prebatched or size is not None
        
        enc_t, enc_f, enc_l = self._flatten_tree()
        self._enc_t = np.array(enc_t, dtype=np.int64)
        self._enc_f = np.array(enc_f, dtype=np.int64)
        self._enc_l = np.array(enc_l, dtype=np.int64)
        
        if prebatched:
            size = shape[0] if size is None else size
            sample_shape = shape[1:]
            calibration = [(item, self._enc_t, self._enc_f, self._enc_l) for item in items]
        elif batched:
            sample_shape = shape
            assert size is not None
            calibration = [(np.stack([item] * size), self._enc_t, self._enc_f, self._enc_l) for item in items]
        else:
            size = 1
            sample_shape = shape
            calibration = [(item, self._enc_t, self._enc_f, self._enc_l) for item in items]

        num_feat = sample_shape[0]
            
        def _single_logic(features, enc_t, enc_f, enc_l):
            return decision_tree_inference(features, enc_t, enc_f, enc_l, num_feat)
            
        def _batch_logic(features_batch, enc_t, enc_f, enc_l):
            return fhe.array([_single_logic(sample, enc_t, enc_f, enc_l) for sample in features_batch])
            
        function = _batch_logic if batched else _single_logic
        parameter = "features_batch" if batched else "features"
        
        compiler = fhe.Compiler(
            function, 
            {
                parameter: "encrypted",
                "enc_t": "encrypted",
                "enc_f": "encrypted",
                "enc_l": "encrypted"
            }
        )
        
        if configuration is None:
            circuit = compiler.compile(calibration, configuration=fhe.Configuration(global_p_error=0.01))
        else:
            circuit = compiler.compile(calibration, configuration=configuration)
            
        self.compiler = compiler
        self.circuit = circuit
        self.batch_size = size
        self._batched = batched
        self._sample_shape = sample_shape
        return self

    def _run_many(self, samples: Any, method: Any, verbose: bool=False) -> Any:
        if self.circuit is None:
            raise ValueError("The model should be compiled before prediction")
        items = [self._integer_array(sample) for sample in samples]
        if not items:
            return []
        expected_shape = self._sample_shape if self._sample_shape is not None else items[0].shape
        if any(item.shape != expected_shape for item in items):
            raise ValueError("sample shape does not match the compiled model")
            
        run = getattr(self.circuit, method)
        if not self._batched:
            return [run(item, self._enc_t, self._enc_f, self._enc_l) for item in _get_progress_bar(items, "predicting...", verbose)]
            
        results = []
        for start in _get_progress_bar(range(0, len(items), self.batch_size), "tree prediction...", verbose):
            batch = items[start:start + self.batch_size]
            count = len(batch)
            batch = batch + [batch[-1]] * (self.batch_size - count)
            results.extend(run(np.stack(batch), self._enc_t, self._enc_f, self._enc_l)[:count])
        return results

    def export_graphviz(self, feature_names = None, class_names = None):
        try:
            import graphviz
        except ImportError as e:
            raise ImportError("Please install graphviz to use this feature.") from e
            
        node_id = 0
        tree_graph = graphviz.Digraph(node_attr={'shape': 'box', 'style': 'filled, rounded', 'fontname': 'helvetica', 'fillcolor': 'white'})
        
        def recurse(node):
            nonlocal node_id
            node_id_str = str(node_id)
            node_id += 1

            if isinstance(node, dict):
                feat_name = feature_names[node.get('feature')] if feature_names else f"Feature {node.get('feature')}"
                node_text = f"{feat_name} >= {node.get('threshold')}"
                if "gini" in node and "samples" in node:
                    node_text += f"\nGini: {node.get('gini'):.3f}"
                    node_text += f"\nSamples: {node.get('samples')}"

                tree_graph.node(node_id_str, label=node_text)
                left_child_id = recurse(node["left"])
                tree_graph.edge(node_id_str, left_child_id)
                right_child_id = recurse(node["right"])
                tree_graph.edge(node_id_str, right_child_id)
                return node_id_str
            else:
                class_value = class_names[node] if class_names else node
                tree_graph.node(node_id_str, label=f"Class: {class_value}") 
            return node_id_str

        recurse(self.tree)
        return tree_graph       

    def show_tree(self) -> None:
        tree_graph = self.export_graphviz()
        
        try:
            tree_graph.view(cleanup=True)
        except Exception as e:
            print(f"Tree could not plotted, there can be an error with Graphviz or GUI in your system. Error:{e}")    

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
    def __init__(self, means: Any, components: Any):
        super().__init__()
        """Initialize the object."""
        self.means = means
        self.components = components

    def _circuit_logic(self, features: Any) -> Any:
        return pca_inference(features, self.means, self.components)


class FHECNN(FHEModel):
    """Encrypted Convolutional Neural Network (CNN) Inference Model.
    
    Applies a 2D convolutional layer to an encrypted image.
    
    Args:
        filters: The public 2D or 3D list of convolutional filters.
        bias: The pre-trained cleartext model bias used for inference.
        
    Example:
        ```python
        model = FHECNN(filters=my_conv_filters, bias=my_conv_bias)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """        
    def __init__(self, filters: Any, bias: Any):
        super().__init__()
        """Initialize the object."""
        self.filters = filters
        self.bias = bias

    def _circuit_logic(self, features: Any) -> Any:
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
    def __init__(self, trees: Any) -> None:
        """Initialize the encrypted random forest model.
        
        Args:
            trees (Any): The list of dictionary representations of the trained trees.
        """
        super().__init__()
        self.trees = trees

    def _flatten_forest(self):
        enc_thresholds_list = []
        enc_feature_indices_list = []
        enc_leaf_values_list = []
        
        def tree_depth(node):
            if not isinstance(node, dict):
                return 0
            return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))
            
        def recurse(node, idx, enc_t, enc_f, enc_l, leaf_start_idx):
            if isinstance(node, dict):
                enc_t[idx] = node["threshold"]
                enc_f[idx] = node["feature"]
                recurse(node["left"], 2 * idx + 1, enc_t, enc_f, enc_l, leaf_start_idx)
                recurse(node["right"], 2 * idx + 2, enc_t, enc_f, enc_l, leaf_start_idx)
            else:
                if idx >= leaf_start_idx:
                    enc_l[idx - leaf_start_idx] = node
                else:
                    enc_t[idx] = 0
                    enc_f[idx] = 0
                    recurse(node, 2 * idx + 1, enc_t, enc_f, enc_l, leaf_start_idx)
                    recurse(node, 2 * idx + 2, enc_t, enc_f, enc_l, leaf_start_idx)
                    
        max_d = max([tree_depth(t) for t in self.trees]) if self.trees else 0
        leaf_start_idx = 2**max_d - 1
        
        for tree in self.trees:
            enc_thresholds = [0] * leaf_start_idx
            enc_feature_indices = [0] * leaf_start_idx
            enc_leaf_values = [0] * (2**max_d)
            
            recurse(tree, 0, enc_thresholds, enc_feature_indices, enc_leaf_values, leaf_start_idx)

            enc_thresholds_list.append(enc_thresholds)
            enc_feature_indices_list.append(enc_feature_indices)
            enc_leaf_values_list.append(enc_leaf_values)
            
        return enc_thresholds_list, enc_feature_indices_list, enc_leaf_values_list

    def compile(self, inputset: Any, batch_size: Any=None, *, inputset_is_batched: Any=None, configuration: Any=None) -> Any:
        if not self.trees:
            raise ValueError("The forest must contain at least one tree")
        if inputset_is_batched is not None and not isinstance(inputset_is_batched, bool):
            raise TypeError("inputset_is_batched must be a boolean or None")
        size = None if batch_size is None else validate_integer("batch_size", batch_size, 1)
        prebatched = size is not None if inputset_is_batched is None else inputset_is_batched
        
        items = [self._integer_array(item) for item in inputset]
        if not items:
            raise ValueError("inputset must contain at least one sample")
        shape = items[0].shape
        if any(item.shape != shape for item in items):
            raise ValueError("all inputset items must have the same shape")
            
        batched = prebatched or size is not None
        
        enc_t_list, enc_f_list, enc_l_list = self._flatten_forest()
        self._enc_t = np.array(enc_t_list, dtype=np.int64)
        self._enc_f = np.array(enc_f_list, dtype=np.int64)
        self._enc_l = np.array(enc_l_list, dtype=np.int64)
        
        if prebatched:
            size = shape[0] if size is None else size
            sample_shape = shape[1:]
            calibration = [(item, self._enc_t, self._enc_f, self._enc_l) for item in items]
        elif batched:
            sample_shape = shape
            assert size is not None
            calibration = [(np.stack([item] * size), self._enc_t, self._enc_f, self._enc_l) for item in items]
        else:
            size = 1
            sample_shape = shape
            calibration = [(item, self._enc_t, self._enc_f, self._enc_l) for item in items]

        num_feat = sample_shape[0]
            
        def _single_logic(features, enc_t, enc_f, enc_l):
            return random_forest_inference(features, enc_t, enc_f, enc_l, num_feat)
            
        def _batch_logic(features_batch, enc_t, enc_f, enc_l):
            return fhe.array([_single_logic(sample, enc_t, enc_f, enc_l) for sample in features_batch])
            
        function = _batch_logic if batched else _single_logic
        parameter = "features_batch" if batched else "features"
        
        compiler = fhe.Compiler(
            function, 
            {
                parameter: "encrypted",
                "enc_t": "encrypted",
                "enc_f": "encrypted",
                "enc_l": "encrypted"
            }
        )
        
        if configuration is None:
            circuit = compiler.compile(calibration, configuration=fhe.Configuration(global_p_error=0.01))
        else:
            circuit = compiler.compile(calibration, configuration=configuration)
            
        self.compiler = compiler
        self.circuit = circuit
        self.batch_size = size
        self._batched = batched
        self._sample_shape = sample_shape
        return self

    def _run_many(self, samples: Any, method: Any, verbose: bool=False) -> Any:
        if self.circuit is None:
            raise ValueError("The model should be compiled before prediction")
        items = [self._integer_array(sample) for sample in samples]
        if not items:
            return []
        expected_shape = self._sample_shape if self._sample_shape is not None else items[0].shape
        if any(item.shape != expected_shape for item in items):
            raise ValueError("sample shape does not match the compiled model")
            
        run = getattr(self.circuit, method)
        if not self._batched:
            return [run(item, self._enc_t, self._enc_f, self._enc_l) for item in _get_progress_bar(items, "predicting...", verbose)]
            
        results = []
        for start in _get_progress_bar(range(0, len(items), self.batch_size), "forest prediction...", verbose):
            batch = items[start:start + self.batch_size]
            count = len(batch)
            batch = batch + [batch[-1]] * (self.batch_size - count)
            results.extend(run(np.stack(batch), self._enc_t, self._enc_f, self._enc_l)[:count])
        return results


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
    def __init__(self, trees: Any):
        super().__init__() 
        """Initialize the object."""
        self.trees = trees   

    def _flatten_forest(self):
        enc_thresholds_list = []
        enc_feature_indices_list = []
        enc_leaf_values_list = []
        
        def tree_depth(node):
            if not isinstance(node, dict):
                return 0
            return 1 + max(tree_depth(node["left"]), tree_depth(node["right"]))
            
        def recurse(node, idx, enc_t, enc_f, enc_l, leaf_start_idx):
            if isinstance(node, dict):
                enc_t[idx] = node["threshold"]
                enc_f[idx] = node["feature"]
                recurse(node["left"], 2 * idx + 1, enc_t, enc_f, enc_l, leaf_start_idx)
                recurse(node["right"], 2 * idx + 2, enc_t, enc_f, enc_l, leaf_start_idx)
            else:
                if idx >= leaf_start_idx:
                    enc_l[idx - leaf_start_idx] = node
                else:
                    enc_t[idx] = 0
                    enc_f[idx] = 0
                    recurse(node, 2 * idx + 1, enc_t, enc_f, enc_l, leaf_start_idx)
                    recurse(node, 2 * idx + 2, enc_t, enc_f, enc_l, leaf_start_idx)
                    
        max_d = max([tree_depth(t) for t in self.trees]) if self.trees else 0
        leaf_start_idx = 2**max_d - 1
        
        for tree in self.trees:
            enc_thresholds = [0] * leaf_start_idx
            enc_feature_indices = [0] * leaf_start_idx
            enc_leaf_values = [0] * (2**max_d)
            
            recurse(tree, 0, enc_thresholds, enc_feature_indices, enc_leaf_values, leaf_start_idx)

            enc_thresholds_list.append(enc_thresholds)
            enc_feature_indices_list.append(enc_feature_indices)
            enc_leaf_values_list.append(enc_leaf_values)
            
        return enc_thresholds_list, enc_feature_indices_list, enc_leaf_values_list

    def compile(self, inputset: Any, batch_size: Any=None, *, inputset_is_batched: Any=None, configuration: Any=None) -> Any:
        if not self.trees:
            raise ValueError("The xgboost ensemble must contain at least one tree")
        if inputset_is_batched is not None and not isinstance(inputset_is_batched, bool):
            raise TypeError("inputset_is_batched must be a boolean or None")
        size = None if batch_size is None else validate_integer("batch_size", batch_size, 1)
        prebatched = size is not None if inputset_is_batched is None else inputset_is_batched
        
        items = [self._integer_array(item) for item in inputset]
        if not items:
            raise ValueError("inputset must contain at least one sample")
        shape = items[0].shape
        if any(item.shape != shape for item in items):
            raise ValueError("all inputset items must have the same shape")
            
        batched = prebatched or size is not None
        
        enc_t_list, enc_f_list, enc_l_list = self._flatten_forest()
        self._enc_t = np.array(enc_t_list, dtype=np.int64)
        self._enc_f = np.array(enc_f_list, dtype=np.int64)
        self._enc_l = np.array(enc_l_list, dtype=np.int64)
        
        if prebatched:
            size = shape[0] if size is None else size
            sample_shape = shape[1:]
            calibration = [(item, self._enc_t, self._enc_f, self._enc_l) for item in items]
        elif batched:
            sample_shape = shape
            assert size is not None
            calibration = [(np.stack([item] * size), self._enc_t, self._enc_f, self._enc_l) for item in items]
        else:
            size = 1
            sample_shape = shape
            calibration = [(item, self._enc_t, self._enc_f, self._enc_l) for item in items]

        num_feat = sample_shape[0]
            
        def _single_logic(features, enc_t, enc_f, enc_l):
            return xgboost_inference(features, enc_t, enc_f, enc_l, num_feat)
            
        def _batch_logic(features_batch, enc_t, enc_f, enc_l):
            return fhe.array([_single_logic(sample, enc_t, enc_f, enc_l) for sample in features_batch])
            
        function = _batch_logic if batched else _single_logic
        parameter = "features_batch" if batched else "features"
        
        compiler = fhe.Compiler(
            function, 
            {
                parameter: "encrypted",
                "enc_t": "encrypted",
                "enc_f": "encrypted",
                "enc_l": "encrypted"
            }
        )
        
        if configuration is None:
            circuit = compiler.compile(calibration, configuration=fhe.Configuration(global_p_error=0.01))
        else:
            circuit = compiler.compile(calibration, configuration=configuration)
            
        self.compiler = compiler
        self.circuit = circuit
        self.batch_size = size
        self._batched = batched
        self._sample_shape = sample_shape
        return self

    def _run_many(self, samples: Any, method: Any, verbose: bool=False) -> Any:
        if self.circuit is None:
            raise ValueError("The model should be compiled before prediction")
        items = [self._integer_array(sample) for sample in samples]
        if not items:
            return []
        expected_shape = self._sample_shape if self._sample_shape is not None else items[0].shape
        if any(item.shape != expected_shape for item in items):
            raise ValueError("sample shape does not match the compiled model")
            
        run = getattr(self.circuit, method)
        if not self._batched:
            return [run(item, self._enc_t, self._enc_f, self._enc_l) for item in _get_progress_bar(items, "predicting...", verbose)]
            
        results = []
        for start in _get_progress_bar(range(0, len(items), self.batch_size), "xgboost prediction...", verbose):
            batch = items[start:start + self.batch_size]
            count = len(batch)
            batch = batch + [batch[-1]] * (self.batch_size - count)
            results.extend(run(np.stack(batch), self._enc_t, self._enc_f, self._enc_l)[:count])
        return results

class FHESVM(FHEModel):
    """Encrypted Support Vector Machine (SVM) Inference Model.
    
    Evaluates a linear SVM over encrypted features.
    
    Args:
        weights: The pre-trained cleartext model weights used for inference.
        bias: The pre-trained cleartext model bias used for inference.
        
    Example:
        ```python
        model = FHESVM(weights=[0.5, -1.2], bias=0.1)
        model.compile(dummy_inputset, batch_size=1)
        ```
    """
    def __init__(self, weights: Any, bias: Any):
        super().__init__()
        """Initialize the object."""
        self.weights = weights
        self.bias = bias

    def _circuit_logic(self, features: Any) -> Any:
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
    def __init__(self, X_train: Any, y_train: Any, k: Any):
        super().__init__()
        """Initialize the object."""
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def _circuit_logic(self, features: Any) -> Any:
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
    def __init__(self, mlp_layers: Any):
        super().__init__()
        """Initialize the object."""
        self.mlp_layers = mlp_layers

    def _circuit_logic(self, features: Any) -> Any:
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
    def __init__(self, log_prob_tables: Any, priors: Any):
        super().__init__()
        """Initialize the object."""
        self.log_prob_tables = log_prob_tables
        self.priors = priors
        self.scale: Optional[int] = None

    def _circuit_logic(self, features: Any) -> Any:
        return naive_bayes_inference(features,self.log_prob_tables,self.priors)


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

    def __init__(self, centroids: Any, *, max_distance: Any):
        super().__init__()
        """Initialize the object."""
        from .models import nearest_centroid_inference

        self._nearest_centroid_inference = nearest_centroid_inference
        self.centroids = [list(centroid) for centroid in centroids]
        self.max_distance = max_distance

    def _circuit_logic(self, features: Any) -> Any:
        return self._nearest_centroid_inference(
            features,
            self.centroids,
            max_distance=self.max_distance,
        )

