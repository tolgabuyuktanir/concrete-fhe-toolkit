"""Activation helpers for encrypted ML circuits."""

from typing import Any, Optional, List, Callable

from .._compat import fhe

from concrete_fhe_toolkit._utils import compile_function, validate_bounds
from concrete_fhe_toolkit.math import greater_equal, make_exp
from concrete_fhe_toolkit.math.basic import maximum
from concrete_fhe_toolkit.arithmetic import make_floor_divide


def relu(value: Any) -> Any:
    """Return max(0, value).
    
    Useful for introducing non-linearity in neural networks while preserving 
    positive values exactly.

    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import relu
        
        # Inside an FHE circuit (e.g. after a linear layer)
        # result = relu(encrypted_tensor_element)
        ```
    """
    return maximum(0, value)


def leaky_relu(value: Any, alpha: float = 0.01) -> Any:
    """Return value when positive, otherwise alpha * value truncated toward zero.

    ``alpha`` must be the reciprocal of a positive integer (for example 0.01,
    0.1, or 0.5) because the slope is applied with integer division.
    
    Unlike standard ReLU, this avoids "dead neurons" by allowing a small,
    non-zero gradient when the unit is not active.

    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import leaky_relu
        
        # Inside an FHE circuit
        # result = leaky_relu(encrypted_val, alpha=0.1)
        ```
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")
    divisor = round(1 / alpha)
    if abs(1 / alpha - divisor) > 1e-9:
        raise ValueError("alpha must be the reciprocal of a positive integer")
    # -((-value) // divisor) truncates toward zero, so small negative inputs
    # scale toward 0 instead of sticking at -1.
    scaled = -((-value) // divisor)
    return maximum(scaled, value)


def unit_step(value: Any) -> Any:
    """Return the Heaviside step of a number: 0 when value < 0, otherwise 1.
    
    Often used in simple perceptrons or binary classification layers.

    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import unit_step
        
        # Inside an FHE circuit
        # result = unit_step(encrypted_val)
        ```
    """
    return greater_equal(value, 0)


def threshold_activation(value: Any, threshold: Any) -> Any:
    """Return 1 when value >= threshold, otherwise 0.
    
    Useful for custom decision boundaries in threshold-based models.

    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import threshold_activation
        
        # Inside an FHE circuit
        # result = threshold_activation(encrypted_val, threshold=5)
        ```
    """
    return greater_equal(value, threshold)

def make_softmax(
    min_input: int = -127, 
    max_input: int = 127,
    *,
    input_scale: int = 10,
    output_scale: int = 100,
    probability_scale: int = 100,
) -> Callable[[List[Any]], List[Any]]:
    """Create a scaled softmax function for a list of encrypted scores.
    
    This function uses an exponential approximation and floor division to calculate
    probabilities as integer percentages. The output is scaled by `probability_scale`.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import make_softmax
        
        # Create a softmax function for inputs scaled by 10
        softmax = make_softmax(min_input=-50, max_input=50, input_scale=10)
        
        # Inside an FHE circuit
        # enc_probs = softmax(enc_scores)
        ```
    """
    exp_func = make_exp(min_input, max_input, input_scale=input_scale, output_scale=output_scale)
    div_func = make_floor_divide(zero_result=0)
    
    def softmax(values: List[Any]) -> List[Any]:
        total: Any = 0
        exp_values = []
        for value in values:
            # make_exp already handles dividing by input_scale and multiplying by output_scale internally
            exp_value = exp_func(value)
            total = total + exp_value
            exp_values.append(exp_value)

        probabilities = []
        for value in exp_values:
            # We multiply by probability_scale before dividing so the output is an integer percentage
            probabilities.append(div_func(value * probability_scale, total))  

        return probabilities

    return softmax     

def compile_relu(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted ReLU.
    
    Compiles the ReLU function into a concrete-python FHE circuit.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import compile_relu
        
        circuit = compile_relu(min_value=-10, max_value=10)
        # circuit.encrypt_run_decrypt(-5) -> 0
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        relu,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def compile_leaky_relu(
    min_value: int = -15,
    max_value: int = 15,
    *,
    alpha: float = 0.01,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted leaky ReLU.
    
    Compiles the Leaky ReLU function into an FHE circuit, keeping a non-zero
    slope for negative inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import compile_leaky_relu
        
        circuit = compile_leaky_relu(min_value=-10, max_value=10, alpha=0.1)
        # circuit.encrypt_run_decrypt(-10) -> -1
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)

    def bound_leaky_relu(value: Any) -> Any:
        return leaky_relu(value, alpha)

    return compile_function(
        bound_leaky_relu,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def compile_unit_step(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted unit step.
    
    Compiles the Heaviside step function into an FHE circuit.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import compile_unit_step
        
        circuit = compile_unit_step(min_value=-10, max_value=10)
        # circuit.encrypt_run_decrypt(5) -> 1
        # circuit.encrypt_run_decrypt(-3) -> 0
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        unit_step,
        {"value": "encrypted"},
        [minimum, maximum],
        configuration,
    )


def compile_threshold_activation(
    min_value: int = -15,
    max_value: int = 15,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile encrypted threshold activation over two encrypted inputs.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import compile_threshold_activation
        
        circuit = compile_threshold_activation(min_value=-10, max_value=10)
        # Evaluates (value >= threshold) fully obliviously
        ```
    """
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        threshold_activation,
        {"value": "encrypted", "threshold": "encrypted"},
        [
            (minimum, minimum),
            (minimum, maximum),
            (maximum, minimum),
            (maximum, maximum),
        ],
        configuration,
    )

def compile_softmax(
    size: int,
    min_value: int = -127,
    max_value: int = 127,
    *,
    configuration: Optional[fhe.Configuration] = None,
) -> fhe.Circuit:
    """Compile an FHE circuit for the softmax function over an array of fixed size.
    
    Since FHE circuits require fixed dimensions, the `size` of the input array 
    must be specified at compile time.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.activations import compile_softmax
        
        # Compile softmax for a 3-class model
        circuit = compile_softmax(size=3, min_value=-50, max_value=50)
        
        # Run it (inputs should be scaled!)
        # probs = circuit.encrypt_run_decrypt([20, -10, 5])
        ```
    """
    function = make_softmax(min_value,max_value)
    minimum, maximum = validate_bounds(min_value, max_value)
    return compile_function(
        function,
        {"values": "encrypted"},
        [
            [minimum] * size,
            [maximum] * size,
            [0] * size
        ],
        configuration,
    )
