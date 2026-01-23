import pytest
from fastapi import status
import time
from unittest.mock import patch, MagicMock
from core.rate_limiter import TieredRateLimiter, UserTier

@pytest.mark.security
class TestUserRateLimiting:
    def test_rate_limit_headers_present(self, client, auth_headers, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 50, True, True]
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 100
                },
                "priority": 5
            },
            headers = auth_headers
        )
        assert "X-RateLimit-Limit" in response.headers or response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    
    def test_rate_limit_retry_after_header(self, client, auth_headers, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 100, True, True]
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 100
                },
                "priority": 5
            },
            headers = auth_headers
        )
        if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            error_detail = response.json()
            assert "retry_after" in error_detail or "Retry-After" in response.headers

@pytest.mark.security
class TestIPBasedRateLimiting:

    def test_distributed_attack_from_same_ip(self, client, mock_redis):
        pipeline = mock_redis.pipeline.return_value

        for i in range(5):
            headers = {f"Authorization": "Bearer test_api_key_user{i}"}

            if i < 3:
                pipeline.execute.return_value = [0, 50 + i * 10, True, True]
                response = client.get("/health", headers = headers)
                assert response.status_code == status.HTTP_200_OK
            else:
                pipeline.execute.side_effect = [
                    [0, 10, True, True],
                    [0, 300, True, True]
                ]

                response = client.get("/health", headers = headers)

@pytest.mark.security
class TestBurstProtection:
    def test_rapid_requests_blocked(self, client, auth_headers, mock_redis):
        pipeline = mock_redis.pipeline.return_value

        request_count = 0
        for i in range(10):
            if i < 5:
                pipeline.execute.return_value = [0, i * 20, True, True]
                response = client.get("/health", headers = auth_headers)
                if response.status_code == status.HTTP_200_OK:
                    request_count += 1

            else:
                pipeline.execute.return_value = [0, 100, True, True]
                response = client.get("/health", headers = auth_headers)
                if response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                    break

        assert request_count < 10

    def test_sliding_window_rate_limit(self, client, auth_headers, mock_redis):
        pipeline = mock_redis.pipeline.return_value

        pipeline.execute.return_value = [0, 80, True, True]
        response1 = client.get("/health", headers = auth_headers)

        pipeline.execute.return_value = [0, 40, True, True]
        response2 = client.get("/health", headers = auth_headers)

        assert response2.status_code == status.HTTP_200_OK

@pytest.mark.security
class TestBulkOperationLimits:
    def test_bulk_submission_rate_limited(self, client, auth_headers, mock_redis):
        pipeline = mock_redis.pipeline.return_value

        pipeline.execute.return_value = [0, 90, True, True]
        jobs = [
            {
                "job_type": "train_sklearn_model",
                "config": {"n_estimators": 100},
                "priority": 5
            }
        ] * 20
        
        response = client.post(
            "/api/v1/bulk/jobs",
            json={"jobs": jobs},
            headers=auth_headers
        )
        

        assert response.status_code in [status.HTTP_201_CREATED, status.HTTP_429_TOO_MANY_REQUESTS]

    def test_bulk_size_limited(self, client, auth_headers):
        jobs = [
            {
                "job_type": "train_sklearn_model",
                "config": {"n_estimators": 100},
                "priority": 5
            }
        ] * 150  # Exceeds max of 100
        
        response = client.post(
            "/api/v1/bulk/jobs",
            json={"jobs": jobs},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST

@pytest.mark.security
class TestResourceExhaustionPrevention:
    def test_job_parameter_limits_enforced(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 1000000,
                    "dataset_rows": 1000000000
                },
                "priority": 5
            },
            headers = auth_headers
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_max_memory_limit_enforced(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "job_type": "train_sklearn_model",
                    "config": {"n_estimators": 100},
                    "max_memory_mb": 1_000_000
                },
                "priority": 5
            },
            headers = auth_headers
        )

        if response.status_code == status.HTTP_201_CREATED:
            job_data = response.json()
            assert job_data["max_memory_mb"] < 100_000

    def test_execution_time_limit_enforced(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json={
                "job_type": "train_sklearn_model",
                "config": {"n_estimators": 100},
                "max_execution_time_sec": 86400 * 365  
            },
            headers=auth_headers
        )

        if response.status_code == status.HTTP_201_CREATED:
            job_data = response.json()
            assert job_data["max_execution_time_sec"] <= 86400

@pytest.mark.security
class TestTieredRateLimiting:
    
    def test_free_tier_lowest_limits(self, client, mock_redis):        
        limiter = TieredRateLimiter(redis_client=mock_redis)
        pipeline = mock_redis.pipeline.return_value
        
        pipeline.execute.return_value = [0, 100, True, True]
        allowed, info = limiter.is_allowed("free_user", user_tier=UserTier.FREE)
        
        assert allowed is False
        assert info["tier"] == "free"
    
    def test_pro_tier_higher_limits(self, client, mock_redis):
        from core.rate_limiter import TieredRateLimiter, UserTier
        
        limiter = TieredRateLimiter(redis_client=mock_redis)
        pipeline = mock_redis.pipeline.return_value
        
        pipeline.execute.return_value = [0, 150, True, True]
        allowed, info = limiter.is_allowed("pro_user", user_tier=UserTier.PRO)
        
        assert allowed is True
        assert info["tier"] == "pro"
    
    def test_enterprise_tier_highest_limits(self, client, mock_redis):
        """Enterprise tier should have highest limits."""
        from core.rate_limiter import TieredRateLimiter, UserTier
        
        limiter = TieredRateLimiter(redis_client=mock_redis)
        pipeline = mock_redis.pipeline.return_value
        
        pipeline.execute.return_value = [0, 5000, True, True]
        allowed, info = limiter.is_allowed("enterprise_user", user_tier=UserTier.ENTERPRISE)
        
        assert allowed is True
        assert info["tier"] == "enterprise"