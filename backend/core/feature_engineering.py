from typing import Dict, List, Any
import numpy as np
import structlog
from numbers import Number as num
logger = structlog.get_logger()

MODEL_COMPLEXITY = {
    "LogisticRegression": 1,
    "DecisionTree": 2,
    "KNeighbors": 3,
    "RandomForest": 5,
    "GradientBoosting": 7,
    "SVC": 4,
}
class FeatureExtractor:
    def __init__(self):
        self.feature_names = [
            "n_estimators",
            "dataset_rows",
            "n_features",
            "max_depth",
            "model_complexity_score",
            "model_complexity_rank",
        ]

    def extract(self, config: Dict[str, Any], job_type: str) -> np.ndarray:
        if job_type == "train_sklearn_model":
            return self._extract_sklearn_features(config)
        else:
            logger.warning("feature_extraction: unkown job type", job_type=job_type)
            return np.zeros(len(self.feature_names))
        
    def _extract_sklearn_features(self, config: Dict[str, Any]) -> np.ndarray:
        n_estimators = validate_config_input(config.get("n_estimators", 100))
        dataset_rows = validate_config_input(config.get("dataset_rows", 10000))
        n_features = validate_config_input(config.get("n_features", 20))
        max_depth = validate_config_input(config.get("max_depth", 10))

        model_name = config.get("model", "RandomForest")
        model_complexity_rank = MODEL_COMPLEXITY.get(model_name, 5)

        model_complexity_score = (n_estimators * max_depth * dataset_rows) / 1_000_000 if isinstance(n_estimators,num) and isinstance(max_depth, num) and isinstance(dataset_rows, num) else 0
        adjusted_complexity_score = model_complexity_score * (model_complexity_rank / 5)

        features = np.array([
            n_estimators,
            dataset_rows,
            n_features,
            max_depth,
            adjusted_complexity_score,
            float(model_complexity_rank),
        ], dtype=np.float64)

        logger.debug(
            "features.extracted",
            n_estimators = n_estimators,
            dataset_rows = dataset_rows,
            complexity_score = adjusted_complexity_score,
            model_rank = model_complexity_rank,
        )

        return features
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()
    
def validate_config_input(value):
    if isinstance(value, num) and value >= 0 and value != float('inf'):
        return value
    else:
        return -1