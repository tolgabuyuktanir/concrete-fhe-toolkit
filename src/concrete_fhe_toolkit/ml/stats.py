from typing import Any,List
from concrete_fhe_toolkit.arrays import array_sum
from concrete_fhe_toolkit.math import maximum, minimum, greater 


def array_mean(array: List[Any]) -> Any:
    """Calculate the mean of an encrypted array."""
    return array_sum(array) // len(array)


def array_variance(array: List[Any]) -> Any:
    """Calculate the variance of an encrypted array."""
    mean_val = array_mean(array)
    diff_sum = 0

    for item in array:
        diff_sum += (item-mean_val)*(item-mean_val)

    return diff_sum // len(array)

def array_max(elements: List[Any]) -> Any:
    """Find the maximum value in an encrypted array using a tournament reduction."""
    current_round = list(elements)
    if(len(current_round) == 0):
        return 0

    while len(current_round) > 1:
        next_round = []
        for i in range(0,len(current_round)-1,2):
            comparison_result = maximum(current_round[i], current_round[i+1])
            next_round.append(comparison_result)

        if(len(current_round)%2 == 1): #odd number of element check
            next_round.append(current_round[-1])

        current_round = next_round    
    
    return current_round[0]

def array_min(elements: List[Any]) -> Any:
    """Find the minimum value in an encrypted array using a tournament reduction."""
    current_round = list(elements)
    if(len(current_round) == 0):
        return 0

    while len(current_round) > 1:
        next_round = []
        for i in range(0,len(current_round)-1,2):
            comparison_result = minimum(current_round[i], current_round[i+1])
            next_round.append(comparison_result)

        if(len(current_round)%2 == 1): #odd number of element check
            next_round.append(current_round[-1])

        current_round = next_round    
    
    return current_round[0]

def array_range(array: List[Any]) -> Any:
    """Calculate the range (max - min) of an encrypted array."""
    min_val = array_min(array)
    max_val = array_max(array)

    return max_val - min_val


def array_count_greater(array: List[Any], threshold: Any) -> Any:
    """Count how many elements in an encrypted array are strictly greater than a threshold."""
    num_of_greater = 0
    for item in array:
        num_of_greater += greater(item,threshold)

    return num_of_greater
