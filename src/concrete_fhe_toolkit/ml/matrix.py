from typing import Any, List
import numpy as np

from concrete_fhe_toolkit.arrays import array_sum
from concrete_fhe_toolkit._utils import validate_integer
from concrete_fhe_toolkit._compat import fhe


def _matrix_shape(matrix):
    """Validate public matrix dimensions without inspecting encrypted values."""
    rows = len(matrix)
    if rows == 0:
        return 0, 0
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix must be rectangular")
    return rows, columns


def _same_matrix_shape(matrix1, matrix2):
    shape = _matrix_shape(matrix1)
    if shape != _matrix_shape(matrix2):
        raise ValueError("Matrix sizes should be equal")
    return shape


def matrix_transpose(matrix: List[List[Any]]) -> List[List[Any]]:
    """Transpose an encrypted 2D matrix (swap rows and columns).
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_transpose
        
        # Inside an FHE circuit
        # A_T = matrix_transpose(enc_matrix_A)
        ```
    """
    _matrix_shape(matrix)
    return [list(row) for row in zip(*matrix)]

def dot_product(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the dot product of two encrypted arrays (vectors).
    
    This is the fundamental operation for linear layers and convolution.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import dot_product
        
        # Inside an FHE circuit
        # score = dot_product(enc_weights, enc_features)
        ```
    """
    if(len(array1) != len(array2)):
        raise ValueError("Array sizes must be equal to perform dot product")

    product_list = [x*y for x,y in zip(array1,array2)]
    result = array_sum(product_list)
    return result

def matrix_add(matrix1: List[List[Any]], matrix2: List[List[Any]]) -> List[List[Any]]:
    """Perform element-wise addition of two encrypted matrices of the same dimensions.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_add
        
        # Inside an FHE circuit
        # C = matrix_add(enc_matrix_A, enc_matrix_B)
        ```
    """
    rows, columns = _same_matrix_shape(matrix1, matrix2)
    return [[matrix1[i][j] + matrix2[i][j] for j in range(columns)] for i in range(rows)]

def matrix_subtract(matrix1: List[List[Any]], matrix2: List[List[Any]]) -> List[List[Any]]:
    """Perform element-wise subtraction of two encrypted matrices of the same dimensions.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_subtract
        
        # Inside an FHE circuit
        # C = matrix_subtract(enc_matrix_A, enc_matrix_B)
        ```
    """
    rows, columns = _same_matrix_shape(matrix1, matrix2)
    return [[matrix1[i][j] - matrix2[i][j] for j in range(columns)] for i in range(rows)]

def matrix_multiply(matrix1: List[List[Any]], matrix2: List[List[Any]]) -> List[List[Any]]:
    """Perform matrix multiplication (dot product) of two encrypted matrices.
    
    Note: Matrix multiplication involves many multiplications and additions, 
    so the resulting FHE circuit may be deep. Use with appropriately scaled values.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_multiply
        
        # Inside an FHE circuit
        # C = matrix_multiply(enc_matrix_A, enc_matrix_B)
        ```
    """
    left_rows, left_columns = _matrix_shape(matrix1)
    right_rows, _ = _matrix_shape(matrix2)
    if left_rows == 0 or right_rows == 0:
        return []
    if left_columns != right_rows:
        raise ValueError("Matrix dimensions are incompatible for multiplication")
    transposed = matrix_transpose(matrix2)
    return [[dot_product(row, column) for column in transposed] for row in matrix1]

def matrix_elementwise_multiply(matrix1: List[List[Any]], matrix2: List[List[Any]]) -> List[List[Any]]:
    """Perform Hadamard (element-wise) multiplication of two encrypted matrices.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_elementwise_multiply
        
        # Inside an FHE circuit
        # C = matrix_elementwise_multiply(enc_matrix_A, enc_matrix_B)
        ```
    """
    rows, columns = _same_matrix_shape(matrix1, matrix2)
    return [[matrix1[i][j] * matrix2[i][j] for j in range(columns)] for i in range(rows)]


def matrix_vector_multiply(matrix: List[List[Any]], array: List[Any]) -> List[Any]:
    """Multiply an encrypted matrix by an encrypted vector.
    
    Often used to evaluate a linear layer: output = W * input.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_vector_multiply
        
        # Inside an FHE circuit
        # output_vec = matrix_vector_multiply(enc_weight_matrix, enc_input_vec)
        ```
    """
    rows, columns = _matrix_shape(matrix)
    if rows and columns != len(array):
        raise ValueError("Matrix and vector dimensions are incompatible")
    return [dot_product(row, array) for row in matrix]

def matrix_exp(matrix: List[List[Any]], exponent: int) -> List[List[Any]]:
    """Calculate the power of an encrypted square matrix.
    
    This uses the highly efficient Square-and-Multiply (Binary Exponentiation)
    algorithm to reduce the number of FHE matrix multiplications required.
    
    Args:
        matrix: The square encrypted matrix.
        exponent: The public integer exponent to raise the matrix to.
        
    Returns:
        The exponentiated encrypted matrix.
        
    Raises:
        ValueError: If the matrix is not square or the exponent is negative.
        
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_exp
        
        # Inside an FHE circuit
        # M_cubed = matrix_exp(enc_matrix, 3)
        ```
    """
    size, columns = _matrix_shape(matrix)
    if size == 0 or columns != size:
        raise ValueError("Matrix should be a nonempty square matrix for exponentiation")
    exponent = validate_integer("exponent", exponent, minimum=0)
    result = [[int(i == j) for j in range(size)] for i in range(size)]
    base = matrix
    while exponent:
        if exponent % 2:
            result = matrix_multiply(result, base)
        exponent //= 2
        if exponent:
            base = matrix_multiply(base, base)
    return result
 
def covariance_matrix(matrix: List[List[Any]]) -> List[List[Any]]:
    """Calculate the covariance matrix of an encrypted 2D dataset.
    
    The dataset should have observations as rows and features as columns.
    Returns the sample covariance (denominator N - 1), floored to integers.
    Means are not rounded: all arithmetic remains integer until the final division.
    
    Args:
        matrix: The encrypted 2D dataset (N observations x M features).
        
    Returns:
        The M x M encrypted covariance matrix.
        
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import covariance_matrix
        
        # Inside an FHE circuit
        # cov = covariance_matrix(enc_dataset)
        ```
    """
    n_samples, n_features = _matrix_shape(matrix)
    if n_samples < 2 or n_features == 0:
        raise ValueError("covariance requires at least two samples and one feature")
    data = fhe.array(matrix)
    totals = np.sum(data, axis=0)
    denominator = n_samples * (n_samples - 1)
    # Compute exact sample covariance before the single final floor division.
    # n * sum(x*y) - sum(x)*sum(y) avoids rounding the means prematurely.
    products = np.reshape(data, (n_samples, n_features, 1)) * np.reshape(
        data, (n_samples, 1, n_features)
    )
    numerators = n_samples * np.sum(products, axis=0) - np.reshape(
        totals, (n_features, 1)
    ) * np.reshape(totals, (1, n_features))
    # Tensor arithmetic keeps diagonal and off-diagonal entries at a common
    # signed integer width in Concrete, including tiny covariances.
    covariance = numerators // denominator
    return [[covariance[i][j] for j in range(n_features)] for i in range(n_features)]


def matrix_flatten(matrix: List[List[Any]]) -> List[Any]: 
    """Flatten a 2D encrypted matrix into a 1D encrypted array.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_flatten
        
        # Inside an FHE circuit
        # vec = matrix_flatten(enc_matrix)
        ```
    """
    flatten_list = []

    for row in matrix:
        for value in row:
            flatten_list.append(value)

    return flatten_list        

def tensor_flatten(tensor: List[List[List[Any]]]) -> List[Any]:
    """Flatten a 3D encrypted tensor into a 1D encrypted array.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import tensor_flatten
        
        # Inside an FHE circuit
        # vec = tensor_flatten(enc_3d_tensor)
        ```
    """
    flatten_list = []
    
    for channel in tensor:
        for row in channel:
            for value in row:
                flatten_list.append(value)

    return flatten_list
