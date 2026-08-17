from typing import Any, List

from concrete_fhe_toolkit.arrays import array_sum


def matrix_transpose(matrix: List[List[Any]]) -> List[List[Any]]:
    return [list(row) for row in zip(*matrix)]

def dot_product(array1: List[Any], array2: List[Any]) -> Any:
    if(len(array1) != len(array2)):
        raise ValueError("Array sizes must be equal to perform dot product")

    product_list = [x*y for x,y in zip(array1,array2)]
    result = array_sum(product_list)
    return result

def matrix_add(matrix1: List[List[Any]], matrix2: List[List[Any]]) -> List[List[Any]]:
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

def matrix_vector_multiply(matrix: List[List[Any]], array: List[Any]) -> List[Any]:
    result_vector = []

    for i in range(len(matrix)):
        result_vector.append(dot_product(matrix[i],array))

    return result_vector        