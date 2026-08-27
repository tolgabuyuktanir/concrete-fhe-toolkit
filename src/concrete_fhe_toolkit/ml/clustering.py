"""Clustering task namespace: models, trainers, and metrics.

sklearn-style entry point for unsupervised grouping of encrypted samples.
"""

from typing import Any, List

from ..arrays import make_minimum
from .classes import FHEKMeans
from .core import euclidean_distance_squared
from .models import nearest_centroid_inference
from .trainers import FHEKMeansTrainer


def inertia(
    samples: List[List[Any]],
    centroids: List[List[int]],
    *,
    max_distance: int,
) -> Any:
    """Sum of squared distances from each sample to its nearest centroid.

    The standard k-means quality metric (lower is better). ``max_distance``
    must bound the squared distance from any sample to any centroid.
    """
    if not centroids:
        raise ValueError("centroids must contain at least one centroid")
    nearest = make_minimum(len(centroids), 0, max_distance)
    total: Any = 0
    for sample in samples:
        distances = [
            euclidean_distance_squared(centroid, sample)
            for centroid in centroids
        ]
        total = total + nearest(distances)
    return total


__all__ = [
    "FHEKMeans",
    "FHEKMeansTrainer",
    "inertia",
    "nearest_centroid_inference",
]
