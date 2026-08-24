from typing import Any, List

from concrete_fhe_toolkit.arrays import array_sum, array_multiply
import math as python_math


def matrix_transpose(matrix: List[List[Any]]) -> List[List[Any]]:
    """Transpose an encrypted 2D matrix (swap rows and columns).
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_transpose
        
        # Inside an FHE circuit
        # A_T = matrix_transpose(enc_matrix_A)
        ```
    """
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
    result_matrix = []

    for i in range(len(matrix1)):
        if(len(matrix1[i]) != len(matrix2[i])):
            raise ValueError("Matrix sizes should be equal")

        row = []
        for j in range(len(matrix1[i])):
            row.append(matrix1[i][j] + matrix2[i][j])

        result_matrix.append(row)

    return result_matrix

def matrix_subtract(matrix1: List[List[Any]], matrix2: List[List[Any]]) -> List[List[Any]]:
    """Perform element-wise subtraction of two encrypted matrices of the same dimensions.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_subtract
        
        # Inside an FHE circuit
        # C = matrix_subtract(enc_matrix_A, enc_matrix_B)
        ```
    """
    result_matrix = []

    for i in range(len(matrix1)):
        if(len(matrix1[i]) != len(matrix2[i])):
            raise ValueError("Matrix sizes should be equal")

        row = []
        for j in range(len(matrix1[i])):
            row.append(matrix1[i][j] - matrix2[i][j])

        result_matrix.append(row)

    return result_matrix

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
    result_matrix = []
    matrix2_transpose = matrix_transpose(matrix2)

    if len(matrix1) == 0 or len(matrix2) == 0:
        return []

    for i in range(len(matrix1)):
        if len(matrix1[i]) != len(matrix2):
            raise ValueError("Matrix dimensions are incompatible for multiplication")
            
        row = []
        for j in range(len(matrix2_transpose)):
            row.append(dot_product(matrix1[i],matrix2_transpose[j]))

        result_matrix.append(row)
        
    return result_matrix

def matrix_elementwise_multiply(matrix1: List[List[Any]], matrix2: List[List[Any]]) -> List[List[Any]]:
    """Perform Hadamard (element-wise) multiplication of two encrypted matrices.
    
    Example:
        ```python
        from concrete_fhe_toolkit.ml.matrix import matrix_elementwise_multiply
        
        # Inside an FHE circuit
        # C = matrix_elementwise_multiply(enc_matrix_A, enc_matrix_B)
        ```
    """
    result_matrix = []
    if len(matrix1) == 0 or len(matrix2) == 0:
        return []

    for i in range(len(matrix1)):
        if len(matrix1[i]) != len(matrix2[i]):
            raise ValueError("Matrix dimensions are incompatible for multiplication")

        result_matrix.append(array_multiply(matrix1[i],matrix2[i]))    

    return result_matrix


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
    result_vector = []

    for i in range(len(matrix)):
        result_vector.append(dot_product(matrix[i],array))

    return result_vector       

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
    if len(matrix) != len(matrix[0]):
        raise ValueError("Matrix should be square matrix for the exponentiation")

    result_matrix = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            row.append(1 if i==j else 0)
        result_matrix.append(row)

    if exponent < 0:
        raise ValueError("The exponent cannot be negative")
    
    elif exponent == 0:
        return result_matrix   
    
    base = matrix
    while exponent > 0:
        if exponent % 2 == 1:
            result_matrix = matrix_multiply(result_matrix, base)
        base = matrix_multiply(base, base)
        exponent = exponent // 2
        
    return result_matrix
 
def covariance_matrix(matrix: List[List[Any]]) -> List[List[Any]]:
    """Calculate the covariance matrix of an encrypted 2D dataset.
    
    The dataset should have observations as rows and features as columns.
    Uses integer division to center the data, maintaining the FHE pipeline.
    
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
    transpose_matrix = matrix_transpose(matrix)
    means = [array_sum(row) // len(row) for row in transpose_matrix]

    cov_matrix = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            row.append(matrix[i][j] - means[j])
        cov_matrix.append(row)

    cov_matrix = matrix_multiply(transpose_matrix,cov_matrix)

    for i in range(len(cov_matrix)):
        for j in range(len(cov_matrix[0])):
            cov_matrix[i][j] = cov_matrix[i][j] // (len(matrix) - 1)
   
    return cov_matrix            


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