from typing import Any, List

from concrete_fhe_toolkit.arrays import array_sum, array_multiply


def matrix_transpose(matrix: List[List[Any]]) -> List[List[Any]]:
    """Transpose an encrypted 2D matrix (swap rows and columns)."""
    return [list(row) for row in zip(*matrix)]

def dot_product(array1: List[Any], array2: List[Any]) -> Any:
    """Calculate the dot product of two encrypted arrays (vectors)."""
    if(len(array1) != len(array2)):
        raise ValueError("Array sizes must be equal to perform dot product")

    product_list = [x*y for x,y in zip(array1,array2)]
    result = array_sum(product_list)
    return result

def matrix_add(matrix1: List[List[Any]], matrix2: List[List[Any]]) -> List[List[Any]]:
    """Perform element-wise addition of two encrypted matrices of the same dimensions."""
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
    """Perform element-wise subtraction of two encrypted matrices of the same dimensions."""
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
    """Perform matrix multiplication (dot product) of two encrypted matrices."""
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
    result_matrix = []
    if len(matrix1) == 0 or len(matrix2) == 0:
        return []

    for i in range(len(matrix1)):
        if len(matrix1[i]) != len(matrix2[i]):
            raise ValueError("Matrix dimensions are incompatible for multiplication")

        result_matrix.append(array_multiply(matrix1[i],matrix2[i]))    

    return result_matrix


def matrix_vector_multiply(matrix: List[List[Any]], array: List[Any]) -> List[Any]:
    """Multiply an encrypted matrix by an encrypted vector."""
    result_vector = []

    for i in range(len(matrix)):
        result_vector.append(dot_product(matrix[i],array))

    return result_vector        

def matrix_flatten(matrix: List[List[Any]]) -> List[Any]: 
    flatten_list = []

    for row in matrix:
        for value in row:
            flatten_list.append(value)

    return flatten_list        

def tensor_flatten(tensor: List[List[List[Any]]]) -> List[Any]:
    flatten_list = []
    
    for channel in tensor:
        for row in channel:
            for value in row:
                flatten_list.append(value)

    return flatten_list 