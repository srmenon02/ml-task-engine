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

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(".env not found, using OS environment variables")

logger = structlog.get_logger()

security = HTTPBearer()

def load_api_keys() -> Dict:
    keys_load = os.getenv("API_KEYS")
    keys = {}
    for key in keys_load.split(","):
        key = key.strip()
        if key:
            user_id = key.split("_")[-1] if "_" in key else "default_user"
            keys[key] = {
                "user_id": user_id,
                "permissions": ["read", "write"],
            }
            
    return keys

VALID_API_KEYS = load_api_keys()

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
    
    user_info = VALID_API_KEYS[api_key]
    logger.info("Authorized API Key", user_id = user_info["user_id"])
    return user_info
