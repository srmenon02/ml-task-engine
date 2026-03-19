import httpx
from jose import jwt, JWTError
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import lru_cache
import time
from models import local_session
from models.user import User
from datetime import datetime, timezone

security = HTTPBearer(auto_error=False)

CLERK_JWKS_URL = "https://creative-amoeba-82.clerk.accounts.dev/.well-known/jwks.json"

_jwks_cache = {"keys": None, "fetched_at": 0}

async def get_jwks():
    now = time.time()
    # Refresh every 60 minutes
    if _jwks_cache["keys"] and now - _jwks_cache["fetched_at"] < 3600:
        return _jwks_cache["keys"]

    async with httpx.AsyncClient() as client:
        response = await client.get(CLERK_JWKS_URL)
        _jwks_cache["keys"] = response.json()
        _jwks_cache["fetched_at"] = now
        return _jwks_cache["keys"]
    
def get_or_create_user(clerk_id: str, email: str) -> dict:
    db = local_session()
    try:
        user = db.query(User).filter(User.clerk_id == clerk_id).first()

        if not user:
            user = User(
                clerk_id=clerk_id,
                email=email if email else None, 
                tier="free",
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            if email and not user.email:
                user.email = email
            user.last_seen_at = datetime.now(timezone.utc)
            db.commit()

        return {
            "id": user.id,
            "clerk_id": user.clerk_id,
            "email": user.email,
            "tier": user.tier,
        }
    finally:
        db.close()

async def verify_clerk_token(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    token = credentials.credentials

    try:
        jwks = await get_jwks()
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={"verify_aud": False}
        )

        user = get_or_create_user(
            clerk_id=payload["sub"],
            email=payload.get("email", ""),
        )

        return {
            "user_id": user["id"],
            "clerk_id": payload["sub"],
            "email": user["email"],
            "tier": user["tier"],
            "permissions": ["read", "write"],
        }

    except JWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token")