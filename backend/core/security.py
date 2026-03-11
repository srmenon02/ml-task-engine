from typing import Dict, Any, List, Optional
import os
import re
import redis
import structlog
import hashlib
from datetime import datetime, timezone
import urllib.parse
logger = structlog.get_logger()

def _fully_decode(value: str, max_rounds: int = 2) -> str:
    prev = value
    for _ in range(max_rounds):
        decoded = urllib.parse.unquote(prev)
        if decoded == prev:
            break
        prev = decoded
    return prev

def _decode_unicode_escapes(value: str) -> str:
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except Exception:
        return value
    
class SecurityValidator:
    ALLOWED_JOB_TYPES = {
        "train_sklearn_model",
        "sleep",
    }

    MAX_N_ESTIMATORS = 100000
    MAX_DATASET_ROWS = 1000000
    MAX_SLEEP_SECONDS = 60 * 60 * 24

    DANGEROUS_PATTERNS = [
        r"__import__",           
        r"eval\s*\(",            
        r"exec\s*\(",           
        r"compile\s*\(",
        r"open\s*\(",
        r"\.system\s*\(",
        r"subprocess",
        r"os\.",
        r"sys\.",
        r"\.\./",                
        r"<script",             
        r"javascript:",
        r";\s*DROP",            
        r";\s*DELETE",
        r";\s*UPDATE",
        r"\sOR\s+\d+\s*=\s*\d+", 
        r"--\s*$",             
        r"/\*",                
        r"\$\{",                 
        r"\$\(",
        r"['\"]\s*or\s*['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?", 
        r"\.\.[\\/]",
        r"windows[\\/]+system32",
        r"system32[\\/]+config",
        r"^/etc/passwd",
        r"^/etc/shadow",
        r"^/proc/",
        r"^/root/",
        r"/etc/",
        r"<script",
        r"<\s*img",
        r"<\s*svg",
        r"<\s*iframe",
        r"<\s*object",
        r"<\s*embed",
        r"onerror\s*=",
        r"onload\s*=",
        r"onclick\s*=",
        r"onmouseover\s*=",
        r"alert\s*\(",
        r"prompt\s*\(",
        r"confirm\s*\(",
        r"string\.fromcharcode",
        r"document\.",
        r"window\.",
        r"\{\{.*?\}\}",
        r"#\{.*?\}",
        r"<%=?\s*.*?\s*%>",
        r"__\w+__",
        r"\bconfig\.\w+\s*\(",
        r"\brequest\.\w+\s*\(",
        r";\s*\w+",        
        r"&\s*\w+",       
        r"\|\s*\w+",    
        r"\|\|\s*\w+",   
        r"&&\s*\w+",     
        r"\b(ls|whoami|id|pwd|cat|uname|ps|env)\b",
        r"\sAND\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"]",
        r"\sAND\s+\d+\s*=\s*\d+",
        r"\sOR\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"]",
        r"\beval\b",
        r"\bexec\b",
        r"\bcompile\b",
        r"(?:^|['\"\s])AND\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"]",
        r"AND\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"]",
        r"'\s*AND\s+'[^']*'\s*=\s*'[^']*'",
        r"['\"]\s*AND\s+['\"]\w+['\"]\s*=\s*['\"]\w+['\"]",
        r"\b(AND|OR)\b\s+['\"].+?['\"]\s*=\s*['\"].+?['\"]",
        "' AND 'x'='x"
    ]

    EXTERNAL_URL_PATTERN = re.compile(
        r"(https?:\/\/[^\s]+)",
        re.IGNORECASE
    )   

    ALLOWED_DOMAINS = ["localhost", "127.0.0.1", "mycompany.internal"]

    SUPPORTED_MODELS = {
            "RandomForest", "LogisticRegression", "SVC", 
            "GradientBoosting", "DecisionTree", "KNeighbors"
    }

    @classmethod
    def validate_job(cls, job_type: str, config: Dict[str, Any]) -> tuple[bool, str]:
        if job_type not in cls.ALLOWED_JOB_TYPES:
            return False, f"Job Type {job_type} is not allowed."
        if not isinstance(config, dict):
            return False, f"Job config must be dictionary"
        
        for key,value in config.items():
            if isinstance(value, str):
                decoded_value = _decode_unicode_escapes(_fully_decode(value))
                normalized = decoded_value.strip()
                for pattern in cls.DANGEROUS_PATTERNS:
                    if re.search(pattern, normalized, re.IGNORECASE):
                        logger.warning(
                            "SecurityValidator dangerous pattern detected",
                            key=key,
                            pattern=pattern,
                            value=normalized,
                        )

                        return False, f"Dangerous Pattern detected in config: {pattern}"
                    
                    if re.search(pattern, decoded_value, re.IGNORECASE):
                        logger.warning(
                            "SecurityValidator dangerous pattern detected",
                            key=key,
                            pattern=pattern,
                            value=decoded_value,
                        )

                        return False, f"Dangerous Pattern detected in config: {pattern}"
                    
                if cls.EXTERNAL_URL_PATTERN.search(decoded_value):
                    domain = re.findall(r"https?://([^/]+)", decoded_value)[0]
                    if domain not in cls.ALLOWED_DOMAINS:
                        return False, f"External URLs are not allowed in config: {value}"

        if job_type == "train_sklearn_model":
            return cls._validate_sklearn_job(config)
        elif job_type == "sleep":
            return cls._validate_sleep_job(config)
        
        return True, ""
    
    @classmethod
    def _validate_sklearn_job(cls, config: Dict) -> tuple[bool, str]:
        n_estimators = config.get("n_estimators", 100)
        if not isinstance(n_estimators, (int, float)):
            return False, "n_estimators must be a number"
        if n_estimators < 1 or n_estimators > cls.MAX_N_ESTIMATORS:
            return False, f"n_estimators must be between 1-{cls.MAX_N_ESTIMATORS}"
        
        dataset_rows = config.get("dataset_rows", 1000)
        if not isinstance(dataset_rows, (int, float)):
            return False, "dataset_rows must be a number"
        if dataset_rows < 1 or dataset_rows > cls.MAX_DATASET_ROWS:
            return False, f"dataset_rows must be between 1-{cls.MAX_DATASET_ROWS}"
        
        model = config.get("model", "RandomForest")
        if model not in cls.SUPPORTED_MODELS:
            return False, f"Model type {model} is not supported"
        
        return True, ""
        
    @classmethod
    def _validate_sleep_job(cls, config: Dict) -> tuple[bool, str]:
        sleep_seconds = config.get("sleep_seconds", 10)
        if not isinstance(sleep_seconds, (int, float)):
            return False, "sleep_seconds must be a number"
        if sleep_seconds < 1 or sleep_seconds > cls.MAX_SLEEP_SECONDS:
            return False, f"sleep_seconds must be between 1-{cls.MAX_SLEEP_SECONDS}"
        
        return True, " "
    
class RedisRateLimiter:
    def __init__(
            self,
            redis_client: Optional[redis.Redis] = None,
            redis_url: str = "redis://localhost:6379/0"
    ):
        
        if redis_client:
            self.redis = redis_client
        else:
            self.redis = redis.from_url(
                redis_url,
                decode_responses = True,
                socket_timeout = 5,
                socket_connect_timeout = 5,
            )

        self.limits = {
            "user": {"requests": 100, "window": 60},
            "ip": {"requests": 300, "window": 60},
            "global": {"requests": 1000, "window": 60},
        }

    def _get_key(self, limit_type: str, identifier: str) -> str:
        hashed = hashlib.sha256(identifier.encode()).hexdigest()[:16]
        return f"ratelimit:{limit_type}:{hashed}"
    
    def is_allowed(
            self,
            user_id: str,
            ip_address: Optional[str] = None,
    ) -> tuple[bool, Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        timestamp = now.timestamp()

        checks = [
            ("user", user_id, self.limits["user"]),
        ]

        if ip_address:
            checks.append(("ip", ip_address, self.limits["ip"]))

        checks.append(("global", "global", self.limits["global"]))

        for limit_type, identifier, config in checks:
            allowed, remaining = self._check_limit(
                limit_type,
                identifier,
                timestamp,
                config["requests"],
                config["window"],
            )

            if not allowed:
                logger.warning(
                    "RedisRateLimiter limit exceeded",
                    limit_type = limit_type,
                    identifier = identifier if limit_type == "global" else "redacted",
                    limit = config["requests"],
                    window = config["window"],
                )

                return False, {
                    "limit_type": limit_type,
                    "limit": config["requests"],
                    "window": config["window"],
                    "retry_after": config["window"],
                }
        return True, {"allowed": True}
    
    def _check_limit(
            self,
            limit_type: str,
            identifier: str,
            timestamp: float,
            max_requests: int,
            window_seconds: int,
    ) -> tuple[bool, int]:
        key = self._get_key(limit_type, identifier)
        window_start = timestamp - window_seconds

        try:
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {f"{timestamp}": timestamp})
            pipe.expire(key, window_seconds * 2)
            results = pipe.execute()
            current_count = results[1]

            if current_count >= max_requests:
                self.redis.zrem(key, f"{timestamp}")
                return False, 0
            
            remaining = max_requests - current_count - 1
            return True, remaining
        
        except redis.RedisError as r:
            logger.error(
                f"RedisRateLimiter Redis error: {r}",
                fallback = "allowing request"
            )

            return True, max_requests
        
    def get_usage(self, user_id: str) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).timestamp()
        user_key = self._get_key("user", user_id)

        try:
            window_start = now - self.limits["user"]["window"]

            self.redis.zremrangebyscore(user_key, 0, window_start)

            current_count = self.redis.zcard(user_key)

            return {
                "user_id": user_id,
                "requests_used": current_count,
                "requests_limit": self.limits["user"]["requests"],
                "window_seconds": self.limits["user"]["window"],
                "requests_remaining": self.limits["user"]["requests"] - current_count,
            }
        
        except redis.RedisError as r:
            logger.error(f"RedisRateLimiter get_usage error: {r}")
            return {"error": "UNable to fetch usage"}
        
    def reset(self, user_id: str) -> bool:
        try:
            key = self._get_key("user", user_id)
            self.redis.delete(key)
            logger.info("RedisRateLimiter reset", user_id = user_id)
            return True
        except redis.RedisError as r:
            logger.error(f"RedisRateLimiter reset error: {r}")
            return False
        
_validator = SecurityValidator()
_rate_limiter = None

def get_validator() -> SecurityValidator:
    return _validator

def get_rate_limiter() -> RedisRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _rate_limiter = RedisRateLimiter(redis_url = redis_url)
    return _rate_limiter

def get_rate_limiter_dep() -> RedisRateLimiter:
    return get_rate_limiter()

