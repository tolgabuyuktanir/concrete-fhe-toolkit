"""Encrypted customer segmentation with k-means — trained on encrypted data.

The server clusters customers it never sees in the clear: each training
iteration runs one circuit that assigns encrypted samples to centroids and
returns only per-cluster sums and counts (the aggregate-decrypt pattern).
Optionally, the released centroids get differential-privacy noise.

Run:  python examples/kmeans_segmentation.py
"""

import numpy as np

from concrete_fhe_toolkit import privacy
from concrete_fhe_toolkit.ml.clustering import FHEKMeansTrainer, inertia


def main() -> None:
    rng = np.random.default_rng(7)
    low_spenders = rng.integers(0, 4, size=(6, 2))
    high_spenders = rng.integers(7, 10, size=(6, 2))
    customers = np.concatenate([low_spenders, high_spenders]).tolist()

    trainer = FHEKMeansTrainer(
        initial_centroids=[[2, 2], [7, 7]],
        min_value=0,
        max_value=9,
        n_iterations=3,
        simulate=True,  # prototyping mode; drop for real encrypted training
    )
    model = trainer.fit_encrypted(customers)
    print("centroids:", model.centroids)
    print("inertia:", int(inertia(customers, model.centroids,
                                  max_distance=model.max_distance)))

    # Release the centroids with a formal privacy guarantee.
    noised = privacy.dp_release(
        [value for centroid in model.centroids for value in centroid],
        sensitivity=1,
        epsilon=8.0,
        rng=np.random.default_rng(0),
    )
    print("dp-released centroids:", [noised[:2], noised[2:]])


if __name__ == "__main__":
    main()
