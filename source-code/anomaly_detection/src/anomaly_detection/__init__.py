from beartype.claw import beartype_this_package

beartype_this_package()

from anomaly_detection.detectors import (  # noqa: E402
    GaussianAnomalyDetector,
    IsolationForestDetector,
    evaluate,
)

__all__ = [
    "GaussianAnomalyDetector",
    "IsolationForestDetector",
    "evaluate",
]
