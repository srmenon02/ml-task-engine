from enum import Enum
from core.security import RedisRateLimiter
import os
from typing import Dict, Optional, Any
from datetime import datetime
import redis
import structlog

logger = structlog.get_logger()
class UserTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class TieredRateLimiter(RedisRateLimiter):
    TIER_LIMITS = {
        UserTier.FREE: {"requests": 100, "window": 60},
        UserTier.PRO: {"requests": 2000, "window": 60},
        UserTier.ENTERPRISE: {"requests": 10000, "window": 60},
    }

    def __init__(self, redis_client = None, redis_url: str = os.getenv("REDIS_URL")):
        super().__init__(redis_client, redis_url)
        self.limits["user"] = self.TIER_LIMITS[UserTier.FREE]

    def is_allowed(
            self,
            user_id: str,
            ip_address: Optional[str] = None,
            user_tier: UserTier = UserTier.FREE,
            cost: int = 1
    ) -> tuple[bool, Dict[str, Any]]:
        user_limit = self.TIER_LIMITS[user_tier]

        now = datetime.now().timestamp()

        allowed, remaining = self._check_limit(
            "user",
            user_id,
            now,
            user_limit["requests"],
            user_limit["window"],
        )

        if not allowed:
            return False, {
                "limit_type": "user",
                "limit": user_limit["requests"],
                "window": user_limit["window"],
                "retry_after": user_limit["window"],
                "tier": user_tier.value
            }
        
        return True, {"allowed": True, "remaining": remaining, "tier": user_tier.value}
    def _check_limit(
            self,
            limit_type: str,
            identifier: str,
            timestamp: float,
            max_requests: int,
            window_seconds: int,
            cost: int = 1
        ) -> tuple[bool, int]:
            key = self._get_key(limit_type, identifier)
            window_start = timestamp - window_seconds

            try:
                pipe = self.redis.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)

                for i in range(cost):
                    pipe.zadd(key, {f"{timestamp}: {i}": timestamp})
                pipe.expire(key, window_seconds * 2)
                results = pipe.execute()

                current_count = results[1]
                print("COUNT:", current_count, "LIMIT:", max_requests)
                if current_count + cost > max_requests:
                    for i in range(cost):
                        self.redis.zrem(key, f"{timestamp}: {i}")
                    return False, 0
                
                remaining = max_requests - current_count - cost
                return True, remaining
            
            except redis.RedisError as r:
                logger.error(f"RedisRateLimiter Redis error: {r}")
                return True, max_requests
                