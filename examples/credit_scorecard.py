"""Encrypted credit scorecard — the fintech story, end to end.

A bank scores loan applicants without ever seeing their raw data: age and
monthly income arrive encrypted, are binned into scorecard buckets inside
the circuit, weighted, and thresholded — one compiled circuit, one
encrypted round trip per applicant.

Run:  python examples/credit_scorecard.py
"""

import numpy as np

from concrete_fhe_toolkit.ml import FHELogisticRegression
from concrete_fhe_toolkit.ml.estimation import estimate_model_cost
from concrete_fhe_toolkit.ml.pipeline import FHEPipeline
from concrete_fhe_toolkit.ml.preprocessing import FHEBinner


def main() -> None:
    # Public scorecard: age buckets and income buckets (thousands/month),
    # with weights fitted offline by the risk team.
    scorecard = FHEPipeline(
        [
            FHEBinner([[21, 30, 45, 60], [10, 25, 60]]),
            FHELogisticRegression(weights=[2, 3], bias=-8),
        ]
    )

    report = estimate_model_cost(scorecard, min_feature=0, max_feature=90)
    print(f"cost level: {report.level} "
          f"({report.comparisons} comparisons, {report.lookups} lookups)")

    inputset = [
        np.array([18, 0], dtype=np.int64),
        np.array([80, 90], dtype=np.int64),
        np.array([35, 20], dtype=np.int64),
    ]
    scorecard.compile(inputset)

    applicants = {
        "genc, dusuk gelir": [22, 8],
        "orta yas, orta gelir": [38, 30],
        "orta yas, yuksek gelir": [50, 70],
    }
    for label, features in applicants.items():
        decision = int(scorecard.simulate(np.array(features, dtype=np.int64)))
        print(f"{label:>24}: {'ONAY' if decision else 'RED'}")


if __name__ == "__main__":
    main()
