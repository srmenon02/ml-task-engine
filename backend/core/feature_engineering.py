from typing import Dict, List, Any
import numpy as np
import structlog

logger = structlog.get_logger()

class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            "n_estimators",
            "dataset_rows",
            "n_features",
            "max_depth",
            "model_complexity_score",
        ]

    def extract(self, config: Dict[str, Any], job_type: str) -> np.ndarray:
        if job_type == "train_sklearn_model":
            return self._extract_sklearn_features(config)
        else:
            logger.warning("feature_extraction: unkown job type", job_type=job_type)
            return np.zeroes(len(self.feature_names))
        
    def _extract_sklearn_features(self, config: Dict[str, Any]) -> np.ndarray:
        n_estimators = config.get("n_estimators", 100)
        dataset_rows = config.get("dataset_rws", 10000)
        n_features = config.get("n_features", 20)
        max_depth = config.get("max_depth", 10)

        model_complexity_score = (
            n_estimators * max_depth * dataset_rows
        ) / 1_000_000

        features = np.array([
            n_estimators,
            dataset_rows,
            n_features,
            max_depth,
            model_complexity_score,
        ], dtype=np.float64)

        logger.debug(
            "features.extracted",
            n_estimators = n_estimators,
            dataset_rows = dataset_rows,
            complexity_score = model_complexity_score,
        )

        return features
    
    def get_feature_names(self) -> List[str]:
        return self.features_names.copy()