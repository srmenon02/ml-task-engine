from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo
from typing import Optional, List
from pathlib import Path
import os

class Settings(BaseSettings):
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env = "DEBUG")

    API_TITLE: str = "ML Task Engine API"
    API_VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "http://localhost:5173"],
        env = "CORS_ORIGIN"
    )

    DB_URL: str = Field(..., env = "DB_URL")
    DB_POOL_SIZE: int = Field(default = 20, env = "DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default = 40, env = "DB_MAX_OVERFLOW")

    REDIS_URL: Optional[str] = Field(..., env = "REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default = None, env = "REDIS_PASSWORD")

    CELERY_BROKER_URL: Optional[str] = Field(..., env = "CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: Optional[str] = Field(..., env = "CELERY_RESULT_BACKEND")
    CELERY_TASK_TIME_LIMIT: int = Field(default = 3600, env = "CELERY_TASK_TIME_LIMIT")

    API_KEYS: Optional[str] = Field(..., env = "API_KEYS")
    SECRET_KEY: Optional[str] = Field(..., env = "SECRET_KEY")
    RATE_LIMIT_ENABLED: bool = Field(default = True, env = "RATE_LIMIT_ENABLED")

    DEFAULT_USER_TIER: str = Field(default = "free", env = "DEFAULT_USER_TIER")
    FREE_TIER_LIMIT: int = Field(default = 100, env = "FREE_TIER_LIMIT")
    PRO_TIER_LIMIT: int = Field(default = 2000, env = "PRO_TIER_LIMIT")
    ENTERPRISE_TIER_LIMIT: int = Field(default = 10000, env = "ENTERPRISE_TIER_LIMIT")
    

    MAX_PAGE_SIZE: int = Field(default = 100, env = "MAX_PAGE_SIZE")
    MAX_BULK_JOBS: int = Field(default = 100, env = "MAX_BULK_JOBS")
    MAX_MEMORY_MB: int = Field(default = 10000, env = "MAX_MEMORY_MB")
    MAX_EXECUTION_TIME_SEC: int = Field(default = 86400, env = "MAX_EXECUTION_TIME_SEC")

    LOG_LEVEL: str = Field(default = "INFO", env = "LOG_LEVEL")
    JSON_LOGS: bool = Field(default = False, env = "JSON_LOGS")

    SENTRY_DSN: Optional[str] = Field(default = None, env = "SENTRY_DSN")
    PROMETHEUS_ENABLED: bool = Field(default = True, env = "PROMETHEUS_ENABLED")
    

    UVICORN_HOST: str = Field(default = "127.0.0.1", env = "UVICORN_HOST")
    UVICORN_PORT: int = Field(default = 8000, env = "UVICORN_PORT")
    @field_validator("ENVIRONMENT")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production", "ci", "test"]
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v
    
    @field_validator("CORS_ORIGINS", mode="before")
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
    
    @field_validator("DB_URL", mode="before")
    def validate_db_url(cls, v, info: ValidationInfo):
        env = info.data.get("ENVIRONMENT")
        if env == "production" and "sqlite" in v.lower():
            raise ValueError("SQLite is not allowed in production environment")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

_settings: Optional[Settings] = None

def reset_settings():
    global _settings
    _settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        env = os.getenv("ENVIRONMENT", "development")
        env_file = Path(f".env.{env}")
        if env_file.exists():
            _settings = Settings(_env_file=str(env_file))
        else:
            _settings = Settings()
    return _settings

def is_production() -> bool:
    return get_settings().ENVIRONMENT == "production"

def is_development() -> bool:
    return get_settings().ENVIRONMENT == "development"
