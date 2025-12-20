from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib
import structlog
from pathlib import Path
import os

from core.feature_engineering import FeatureExtractor
from models import local_session, ResourceProfile

logger = structlog.get_logger()
_predictor = None

class ResourcePredictor:

    def __init__(self, model_path: Optional[str] = None):
        self.feature_extractor = FeatureExtractor()
        self.memory_model = Ridge(alpha=1.0)
        self.cpu_model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_samples = 0

        if model_path:
            self.model_path = Path(model_path)
            self.model_path.mkdir(parents=True, exist_ok=True)
        else:
            self.model_path = Path(__file__).parent / "saved_models"
            self.model_path.mkdir(parents=True, exist_ok=True)
        
        self._load_models()


    def predict(
            self,
            job_config: Dict,
            job_type: str,
    ) -> Tuple[float, float]:
    
        features = self.feature_extractor.extract(job_config, job_type)
        features = features.reshape(1, -1)

        if not self.is_trained:
            logger.warning(
                "predictor untrained",
                message = "Model not trained, using conservative estimates"
            )
            return self._get_conservative_estimate(job_config)
        
        try:
            features_scaled = self.scaler.transform(features)

            memory_mb = self.memory_model.predict(features_scaled)[0]
            cpu_percent = self.cpu_model.predict(features_scaled)[0]

            logger.info(
                "predictor predicted",
                memory_mb = memory_mb,
                cpu_percent = cpu_percent,
                training_samples = self.training_samples,
            )

            return memory_mb, cpu_percent

        except Exception as e:
            logger.error(f"prediction.prediction failed {e}")
            return self._get_conservative_estimate(job_config)
        
    def train(self, min_samples: int = 10) -> bool:
        logger.info("prediction training started")

        db = local_session()
        try:
            profiles = db.query(ResourceProfile).all()

            if len(profiles) < min_samples:
                logger.warning(
                    "predictor insufficient data",
                    available = len(profiles),
                    required = min_samples,
                )
                return False
            X = []
            y_memory = []
            y_cpu = []

            for profile in profiles:
                features = self.feature_extractor.extract(
                    profile.config,
                    profile.job_type
                )
                X.append(features)
                y_memory.append(profile.memory_mb)
                y_cpu.append(profile.cpu_percent)

            X = np.array(X)
            y_memory = np.array(y_memory)
            y_cpu = np.array(y_cpu)

            X_scaled = self.scaler.fit_transform(X)

            self.memory_model.fit(X_scaled, y_memory)
            self.cpu_model.fit(X_scaled, y_cpu)

            self.is_trained = True
            self.training_samples = len(profiles)

            self._save_models()

            logger.info(
                "predictor training completed",
                samples = len(profiles),
                memory_score = self.memory_model.score(X_scaled, y_memory),
                cpu_score = self.cpu_model.score(X_scaled, y_cpu),
            )

            return True
        
        except Exception as e:
            logger.error(f"prediction training failed {e}")
            return False
        
        finally:
            db.close()

    def evaluate(self) -> Dict[str, float]:
        if not self.is_trained:
            return {"error": "model cannot be evaluated prior to training"}
        
        db = local_session()

        try:
            profiles = db.query(ResourceProfile).all()

            if len(profiles) == 0:
                return {"error": "No data avialble for evaluation"}
            
            X = []
            y_memory_true = []
            y_cpu_true = []

            for profile in profiles:
                features = self.feature_extractor.extract(
                    profile.config,
                    profile.job_type
                )

                X.append(features)
                y_memory_true.append(profile.memory_mb)
                y_cpu_true.append(profile.cpu_percent)

            X = np.array(X)
            y_memory_true = np.array(y_memory_true)
            y_cpu_true = np.array(y_cpu_true)

            X_scaled = self.scaler.transform(X)
            y_memory_pred = self.memory_model.predict(X_scaled)
            y_cpu_pred = self.cpu_model.predict(X_scaled)

            memory_mae = np.mean(np.abs(y_memory_true - y_memory_pred))
            memory_mape = np.mean(np.abs((y_memory_true - y_memory_pred) / y_memory_true)) * 100

            cpu_mae = np.mean(np.abs(y_cpu_true - y_cpu_pred))
            cpu_mape = np.mean(np.abs((y_cpu_true - y_cpu_pred) / y_cpu_true)) * 100

            return {
                "memory_mae_mb" : float(memory_mae),
                "memory_mape_percent": float(memory_mape),
                "cpu_mae_percent": float(cpu_mae),
                "cpu_mape_percent": float(cpu_mape),
                "samples": len(profiles),
            }

        except Exception as e:
            logger.error(f"predictor evalutor failed: {e}")
            return {"error": str(e)}
        
        finally: 
            db.close()

    def _get_conservative_estimate(self, job_config: Dict) -> Tuple[float, float]:
        dataset_rows = job_config.get("dataset_rows", 10000)
        n_estimators = job_config.get("n_estimators", 100)

        memory_mb = (dataset_rows * 0.01) + (n_estimators * 1.0)
        memory_mb = max(memory_mb, 100.0)

        cpu_percent = min(20 + (n_estimators / 10), 80.0)

        logger.info(
            "predictor conservative estimate",
            memory_mb = memory_mb,
            cpu_percent = cpu_percent,
        )

        return memory_mb, cpu_percent
    
    def _save_models(self):
        try:
            joblib.dump(self.memory_model, self.model_path / "memory_model.pkl")
            joblib.dump(self.cpu_model, self.model_path / "cpu_model.pkl")
            joblib.dump(self.scaler, self.model_path / "scaler.pkl")

            logger.info("predictor moodels saved")

        except Exception as e:
            logger.error(f"predictor save failed {e}")

    def _load_models(self):
        try:
            memory_path = self.model_path / "memory_model.pkl"
            cpu_path = self.model_path / "cpu_model.pkl"
            scaler_path = self.model_path / "scaler.pkl"

            if memory_path.exists() and cpu_path.exists() and scaler_path.exists():
                self.memory_model = joblib.load(memory_path)
                self.cpu_model = joblib.load(cpu_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True

                logger.info("predictor models_loaded", path=str(self.model_path))
            else:
                logger.info("no saved models found", 
                    memory_exists=memory_path.exists(),
                    cpu_exists=cpu_path.exists(),
                    scaler_exists=scaler_path.exists()
                )

        except Exception as e:
            logger.error(f"predictor load models failed {e}")

def get_predictor() -> ResourcePredictor:
    global _predictor
    if _predictor is None:
        _predictor = ResourcePredictor()
    return _predictor

