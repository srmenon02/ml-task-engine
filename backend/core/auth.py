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
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(".env not found, using OS environment variables")

logger = structlog.get_logger()

security = HTTPBearer(auto_error=False)

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
    if credentials is None:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid auth scheme")

    api_key = credentials.credentials

    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return VALID_API_KEYS[api_key]

@dataclass
class CurrentUser:
    user_id: str
    is_admin: bool = False

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> CurrentUser:
    user_info = verify_api_key(credentials)

    is_admin = user_info["user_id"].startswith("admin")

    return CurrentUser(
        user_id = user_info["user_id"],
        is_admin = is_admin
    )
