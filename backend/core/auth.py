from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict
import json
import secrets
import hashlib
import structlog
import os
from dotenv import load_dotenv
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

if not ENV_PATH.exists():
    raise RuntimeError(f".env file not found at {ENV_PATH}")

load_dotenv(ENV_PATH)

logger = structlog.get_logger()

security = HTTPBearer()

def get_api_keys() -> Dict:
    return json.loads(os.getenv("API_KEYS"))

VALID_API_KEYS = get_api_keys()

def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    api_key = credentials.credentials

    if api_key not in VALID_API_KEYS:
        logger.warning("Auth.Invalid API Key Attempt", key_prefix=api_key[:8])
        raise HTTPException(
            status_code = 401,
            detail = "Invalid API Key"
        )
    
    logger.info("Authorized API Key", user_id = VALID_API_KEYS[api_key]["user_id"])
    return VALID_API_KEYS[api_key]
