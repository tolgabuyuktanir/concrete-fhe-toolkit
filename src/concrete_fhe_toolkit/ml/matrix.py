from typing import Any, List

from concrete_fhe_toolkit.arrays import array_sum


def matrix_transpose(matrix: List[List[Any]]) -> List[List[Any]]:
    return [list(row) for row in zip(*matrix)]

def dot_product(array1: List[Any], array2: List[Any]) -> Any:
    list1 = list(array1)
    list2 = list(array2)
    if(len(list1) != len(list2)):
        raise ValueError("Array sizes must be equal to perform dot product")

    product_list = [x*y for x,y in zip(list1,list2)]
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