import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path

from core.predictor import ResourcePredictor, get_predictor
from models import ResourceProfile

class TestResourcePredictor:

    @pytest.fixture
    def predictor(self, tmp_path):
        return ResourcePredictor(model_path = str(tmp_path))
    
    @pytest.fixture
    def sample_profiles(self, test_db):
        profiles = []
        for i in range(20):
            profile = ResourceProfile(
                job_type = "train_sklearn_model",
                config = {
                    "n_estimators": 100 + i * 10,
                    "dataset_rows": 10000 + i * 1000,
                    "n_features": 20,
                    "max_depth": 10
                },
                memory_mb = 500 + i * 50,
                cpu_percent = 30 + i * 2,
                execution_time = 60 + i * 5
            )
            test_db.add(profile)
            profiles.append(profile)

        test_db.commit()
        return profiles
    
    def test_predictor_initialization(self, predictor):
        assert predictor.is_trained is False
        assert predictor.training_samples == 0
        assert predictor.memory_model is not None
        assert predictor.cpu_model is not None

    def test_predict_untrained_uses_conservative_estimate(self, predictor):
        config = {
            "n_estimators": 100,
            "dataset_rows": 10000
        }

        memory_mb, cpu_percent = predictor.predict(config, "train_sklearn_model")

        assert memory_mb >= 100.0
        assert cpu_percent >= 20.0
        assert cpu_percent <= 80.0

    @patch('core.predictor.local_session')
    def test_train_with_insufficient_data(self, mock_session, predictor):
        mock_db = Mock()
        mock_db.query().all.return_value = [Mock()] * 3
        mock_session.return_value = mock_db

        success = predictor.train(min_samples = 10)

        assert success is False
        assert predictor.is_trained is False

    @patch('core.predicotr.local_session')
    def test_train_with_sufficient_data(self, mock_session, predictor, sample_profiles):
        mock_db = Mock()
        mock_db.query().all.return_value = sample_profiles
        mock_session.return_value = mock_db

        success = predictor.train(min_samples = 5)

        assert success is True
        assert predictor.is_trained is True
        assert predictor.training_samples == len(sample_profiles)

    def test_predict_after_training(self, predictor, sample_profiles, test_db):
        with patch('core.predictor.local_session', return_value = test_db):
            predictor.train(min_samples = 5)

        config = {
            "n_estimators": 150,
            "dataset_rows": 15000,
            "n_features": 20,
            "max_depth": 10
        }

        memory_mb, cpu_percent = predictor.predict(config, "train_sklearn_model")

        assert 100 < memory_mb < 5000
        assert 10 < cpu_percent < 100

    @pytest.mark.parametrize("n_estimators, dataset_rows, expected_min_memory", [
        (50, 5000, 50),
        (200, 20000, 200),
        (500, 50000, 500),
    ])
    def test_conservative_estimates_scale_with_config(
        self, predictor, n_estimators, dataset_rows, expected_min_memory
    ):
        config = {
            "n_estimators": n_estimators,
            "dataset_rows": dataset_rows
        }

        memory_mb, cpu_percent = predictor._get_conservative_estimate(config)

        assert memory_mb >= expected_min_memory



