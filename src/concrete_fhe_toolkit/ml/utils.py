from typing import List,Any
from concrete_fhe_toolkit.math import equal

def one_hot_encode(label: Any, num_classes: int) -> List[Any]:
    encoded_list = []
    for item in range(num_classes):
        encoded_list.append(equal(label, item))

    return encoded_list


