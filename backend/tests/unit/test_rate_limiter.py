import pytest
from datetime import datetime
from core.rate_limiter import TieredRateLimiter, UserTier
from core.security import RedisRateLimiter
import redis as redis_lib
from unittest.mock import MagicMock
@pytest.fixture
def redis_rate_limiter(mock_redis):
    return RedisRateLimiter(redis_client=mock_redis)
@pytest.mark.unit
class TestRedisRateLimiter:
    
    def test_allows_requests_within_limit(self, redis_rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 50, True, True]
        
        allowed, info = redis_rate_limiter.is_allowed("user123")
        
        assert allowed is True
        assert info["allowed"] is True

    def test_blocks_request_exceeding_limit(self, redis_rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 100, True, True]
        
        allowed, info = redis_rate_limiter.is_allowed("user123")
        
        assert allowed is False
        assert "limit_type" in info
        assert info["limit"] == 100

    def test_separate_limits_per_user(self, redis_rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value

        pipeline.execute.return_value = [0, 100, True, True]
        allowed1, _ = redis_rate_limiter.is_allowed("user1")

        pipeline.execute.return_value = [0, 10, True, True]
        allowed2, _ = redis_rate_limiter.is_allowed("user2")

        assert allowed1 is False
        assert allowed2 is True

    def test_ip_based_rate_limiting(self, redis_rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 50, 0, 501, 0, 50]

        allowed, info = redis_rate_limiter.is_allowed("user123", ip_address="192.168.1.1")

        assert allowed is False
        assert info["limit_type"] == "ip"

    def test_global_rate_limit(self, redis_rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 50, 0, 1001]

        allowed, info = redis_rate_limiter.is_allowed("user123")

        assert allowed is False
        assert info["limit_type"] == "global"

    def test_get_usage_statistics(self, redis_rate_limiter, mock_redis):
        mock_redis.zcard.return_value = 75

        usage = redis_rate_limiter.get_usage("user123")

        assert usage["user_id"] == "user123"
        assert usage["requests_used"] == 75
        assert usage["requests_remaining"] == 25
        assert usage["requests_limit"] == 100

    def test_reset_rate_limit(self, redis_rate_limiter, mock_redis):
        mock_redis.delete.return_value = 1

        success = redis_rate_limiter.reset("user123")

        assert success is True
        mock_redis.delete.assert_called()

    def test_fallback_on_redis_error(self, redis_rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.side_effect = redis_lib.RedisError("Connection Failed")

        allowed, info = redis_rate_limiter.is_allowed("user123")
        assert allowed is True

@pytest.fixture
def rate_limiter(mock_redis):
    return TieredRateLimiter(redis_client=mock_redis)
@pytest.mark.unit
class TestTieredRateLimiter:
    def test_free_tier_limits(self, rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 50, True, True]
        
        allowed, info = rate_limiter.is_allowed("user123", user_tier = UserTier.FREE)
        
        assert allowed is True
        assert info["tier"] == "free"

    def test_pro_tier_higher_limits(self, rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 150, True, True]
        
        allowed, info = rate_limiter.is_allowed("user123", user_tier = UserTier.PRO)
        
        assert allowed is True
        assert info["tier"] == "pro"

    def test_enterprise_tier_higher_limits(self, rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 5000, True, True]
        
        allowed, info = rate_limiter.is_allowed("user123", user_tier = UserTier.ENTERPRISE)
        
        assert allowed is True
        assert info["tier"] == "enterprise"

    def test_cost_based_tier_limiting(self, rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value

        pipeline.execute.return_value = [0, 90, True, True]
        allowed, _ = rate_limiter.is_allowed("user123", user_tier = UserTier.FREE, cost = 1)
        assert allowed is True

        pipeline.execute.return_value = [0, 100, True, True]
        allowed, _ = rate_limiter.is_allowed("user123", user_tier = UserTier.FREE, cost = 20)
        assert allowed is False

    @pytest.mark.security
    def test_prevents_tier_escalation(self, rate_limiter, mock_redis):
        pipeline = mock_redis.pipeline.return_value

        pipeline.execute.return_value = [0, 100, True, True]
        allowed, info = rate_limiter.is_allowed("user123", user_tier = UserTier.FREE)
        assert allowed is False




