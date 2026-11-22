from .classical import IForestDetector, LOFDetector, OCSVMDetector, PCADetector, RareEventRule
from .deep import AutoEncoder, DeepLog

DETECTORS = {
    "RareEvent": RareEventRule,
    "PCA": PCADetector,
    "IsolationForest": IForestDetector,
    "OneClassSVM": OCSVMDetector,
    "LOF": LOFDetector,
    "AutoEncoder": AutoEncoder,
    "DeepLog": DeepLog,
}

__all__ = ["DETECTORS", "RareEventRule", "PCADetector", "IForestDetector", "OCSVMDetector", "LOFDetector",
           "AutoEncoder", "DeepLog"]
