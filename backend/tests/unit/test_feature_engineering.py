import pytest
import numpy as np
from core.feature_engineering import FeatureExtractor

class TestFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    def test_initialization(self, extractor):
        assert len(extractor.feature_names) > 0
        assert "n_estimators" in extractor.feature_names
        assert "dataset_rows" in extractor.feature_names

    def test_extract_sklearn_features(self, extractor):
        config = {
            "n_estimators": 100,
            "dataset_rows": 10000,
            "n_features": 20,
            "max_depth": 10,
        }
        features = extractor.extract(config, "train_sklearn_model")
        assert len(features) == len(extractor.feature_names)
        assert isinstance(features, np.ndarray)
        assert features[0] == 100
        assert features[1] == 10000

    def test_extract_with_missing_values(self, extractor):
        config = {
            "n_estimators": 50
        }

        features = extractor.extract(config, "train_sklearn_model")

        assert isinstance(features, np.ndarray)
        assert features[0] == 50
        assert all(f >= 0 for f in features)

    def test_complexity_score_calculation(self, extractor):
        simple_config = {
            "n_estimators": 10,
            "dataset_rows": 100,
            "max_depth": 2
        }

        complex_config = {
            "n_estimators": 500,
            "dataset_rows": 1000000,
            "max_depth": 20
        }

        simple_features = extractor.extract(simple_config, "train_sklearn_model")
        complex_features = extractor.extract(complex_config, "train_sklearn_model")

        assert simple_features[4] < complex_features[4]

    def test_feature_scaling(self, extractor):
        config = {
            "n_estimators": 200,
            "dataset_rows": 50000,
            "n_features": 30,
            "max_depth": 15,
        }
        features = extractor.extract(config, "train_sklearn_model")
        assert np.all(np.isfinite(features))
        assert np.all(features >= 0)

    def test_unknown_job_type(self, extractor):
        config = {"param": "value"}

        features = extractor.extract(config, "unknown_job_type")
        assert isinstance(features, np.ndarray)
        assert len(features) == len(extractor.feature_names)

    def test_get_feature_names(self, extractor):
        names = extractor.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)

    @pytest.mark.parametrize("n_estimators,dataset_rows,expected_complexity_order", [
        (10, 100, "low"),
        (100, 10000, "medium"),
        (1000, 100000, "high"),
    ])
    def test_complexity_score_ordering(self, extractor, n_estimators, dataset_rows, expected_complexity_order):
        config = {
            "n_estimators": n_estimators,
            "dataset_rows": dataset_rows,
            "max_depth": 10,
        }
        features = extractor.extract(config, "train_sklearn_model")
        complexity = features[4]

        if expected_complexity_order == "low":
            assert complexity < 10
        elif expected_complexity_order == "medium":
            assert 10 <= complexity < 1000
        elif expected_complexity_order == "high":
            assert complexity >= 1000

    @pytest.mark.security
    def test_handles_invalid_numeric_values(self, extractor):
        config = {
            "n_estimators": -100,
            "dataset_rows": 0,
            "n_features": float('inf'),
            "max_depth": None
        }

        features = extractor.extract(config, "train_sklearn_model")

        print("Features: ", type(features), features)
        assert isinstance(features, np.ndarray)
        assert np.all(np.isfinite(features))

    @pytest.mark.security
    def test_handles_string_injection_attempts(self, extractor):
        config = {
            "n_estimators": "'; DROP TABLE jobs; --",
            "dataset_rows": "<script>alert('xss')</script>",
            "n_features": "../../etc/passwd",
        }
        
        features = extractor.extract(config, "train_sklearn_model")

        assert isinstance(features, np.ndarray)
        assert np.all(np.isfinite(features))

class TestModelComplexityRank:
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()

    def test_feature_vector_is_six_elements(self, extractor):
        config = {"model": "RandomForest", "n_estimators": 100, "dataset_rows": 10000}
        features = extractor.extract(config, "train_sklearn_model")
        assert len(features) == 6

    @pytest.mark.parametrize("lighter,heavier", [
        ("LogisticRegression", "RandomForest"),
        ("RandomForest", "GradientBoosting"),
        ("DecisionTree", "SVC"),
    ])
    def test_heavier_models_produce_higher_rank(self, extractor, lighter, heavier):
        config = {"n_estimators": 100, "dataset_rows": 10000, "max_depth": 10}

        light_features = extractor.extract({**config, "model": lighter}, "train_sklearn_model")
        heavy_features = extractor.extract({**config, "model": heavier}, "train_sklearn_model")

        assert light_features[5] < heavy_features[5]

    def test_unknown_model_gets_default_rank(self, extractor):
        config = {"model": "SomeNewModel", "n_estimators": 100, "dataset_rows": 10000, "max_depth": 10}
        features = extractor.extract(config, "train_sklearn_model")
        assert features[5] == 5.0
        assert np.all(np.isfinite(features))

    def test_adjusted_complexity_scales_with_model(self, extractor):
        config = {"n_estimators": 100, "dataset_rows": 10000, "max_depth": 10}

        lr_features = extractor.extract({**config, "model": "LogisticRegression"}, "train_sklearn_model")
        gb_features = extractor.extract({**config, "model": "GradientBoosting"}, "train_sklearn_model")

        assert lr_features[4] < gb_features[4]
