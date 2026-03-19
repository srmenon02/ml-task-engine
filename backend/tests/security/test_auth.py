import pytest
from fastapi import status
from unittest.mock import Mock, patch, AsyncMock
from fastapi.security import HTTPAuthorizationCredentials
import base64
import time
from models import Job, JobStatus
from core.auth import verify_clerk_token
from api.main import app

MOCK_CLERK_PAYLOAD = {
    "sub": "user_clerk123",
    "email": "test@example.com",
    "iat": int(time.time()),
    "exp": int(time.time()) + 3600,
}

MOCK_USER_DICT = {
    "id": "internal-uuid-123",
    "clerk_id": "user_clerk123",
    "email": "test@example.com",
    "tier": "free",
}

MOCK_AUTH = {
    "user_id": MOCK_USER_DICT["id"],
    "clerk_id": MOCK_CLERK_PAYLOAD["sub"],
    "email": MOCK_USER_DICT["email"],
    "tier": "free",
    "permissions": ["read", "write"],
}


@pytest.mark.security
class TestClerkTokenVerification:

    @pytest.mark.asyncio
    async def test_valid_token_accepted(self):
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = "valid.jwt.token"

        with patch("core.auth.get_jwks", new_callable=AsyncMock) as mock_jwks, \
             patch("core.auth.jwt.decode", return_value=MOCK_CLERK_PAYLOAD), \
             patch("core.auth.get_or_create_user", return_value=MOCK_USER_DICT):

            mock_jwks.return_value = {"keys": []}
            from core.auth import verify_clerk_token as real_verify
            result = await real_verify(credentials)

        assert result is not None
        assert "user_id" in result
        assert result["user_id"] == MOCK_USER_DICT["id"]

    @pytest.mark.asyncio
    async def test_missing_credentials_rejected(self):
        with patch("core.auth.get_jwks", new_callable=AsyncMock):
            from core.auth import verify_clerk_token as real_verify
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await real_verify(None)
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self):
        from jose import JWTError
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = "invalid.jwt.token"

        with patch("core.auth.get_jwks", new_callable=AsyncMock) as mock_jwks, \
             patch("core.auth.jwt.decode", side_effect=JWTError("bad token")):

            mock_jwks.return_value = {"keys": []}
            from core.auth import verify_clerk_token as real_verify
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await real_verify(credentials)
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_token_returns_correct_shape(self):
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = "valid.jwt.token"

        with patch("core.auth.get_jwks", new_callable=AsyncMock) as mock_jwks, \
             patch("core.auth.jwt.decode", return_value=MOCK_CLERK_PAYLOAD), \
             patch("core.auth.get_or_create_user", return_value=MOCK_USER_DICT):

            mock_jwks.return_value = {"keys": []}
            from core.auth import verify_clerk_token as real_verify
            result = await real_verify(credentials)

        assert "user_id" in result
        assert "clerk_id" in result
        assert "email" in result
        assert "tier" in result
        assert "permissions" in result
        assert isinstance(result["permissions"], list)

    @pytest.mark.asyncio
    async def test_token_includes_read_write_permissions(self):
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = "valid.jwt.token"

        with patch("core.auth.get_jwks", new_callable=AsyncMock) as mock_jwks, \
             patch("core.auth.jwt.decode", return_value=MOCK_CLERK_PAYLOAD), \
             patch("core.auth.get_or_create_user", return_value=MOCK_USER_DICT):

            mock_jwks.return_value = {"keys": []}
            from core.auth import verify_clerk_token as real_verify
            result = await real_verify(credentials)

        assert "read" in result["permissions"]
        assert "write" in result["permissions"]


@pytest.mark.security
class TestBearerTokenFormat:

    @pytest.fixture(autouse=True)
    def remove_auth_override(self):
        app.dependency_overrides.pop(verify_clerk_token, None)
        yield

    def test_accepts_valid_bearer_token(self, client):
        async def override(credentials=None):
            return MOCK_AUTH
        app.dependency_overrides[verify_clerk_token] = override

        response = client.get(
            "/health",
            headers={"Authorization": "Bearer valid.jwt.token"}
        )
        assert response.status_code == status.HTTP_200_OK

    def test_rejects_basic_auth(self, client):
        creds = base64.b64encode(b"user:pass").decode()
        response = client.get(
            "/jobs",
            headers={"Authorization": f"Basic {creds}"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_rejects_missing_auth_header(self, client):
        response = client.get("/jobs")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.security
class TestAuthorizationChecks:

    def test_user_can_only_see_own_jobs(self, client, test_db):
        job_user1 = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100},
            priority=5,
            user_id="internal-uuid-user1",
        )
        job_user2 = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100},
            priority=5,
            user_id="internal-uuid-user2",
        )
        test_db.add_all([job_user1, job_user2])
        test_db.commit()

        auth_user1 = {
            "user_id": "internal-uuid-user1",
            "clerk_id": "user_clerk_1",
            "email": "user1@example.com",
            "tier": "free",
            "permissions": ["read", "write"],
        }

        async def override_as_user1(credentials=None):
            return auth_user1

        app.dependency_overrides[verify_clerk_token] = override_as_user1

        try:
            response = client.get(
                f"/jobs/{job_user1.id}",
                headers={"Authorization": "Bearer valid.jwt.token"}
            )
            assert response.status_code == status.HTTP_200_OK

            response = client.get(
                f"/jobs/{job_user2.id}",
                headers={"Authorization": "Bearer valid.jwt.token"}
            )
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_401_UNAUTHORIZED,
            ]
        finally:
            app.dependency_overrides.pop(verify_clerk_token, None)

    def test_cannot_impersonate_other_users(self, client):
        response = client.post(
            "/jobs",
            json={
                "job_type": "train_sklearn_model",
                "config": {"n_estimators": 100},
                "user_id": "victim-uuid",
            },
            headers={"Authorization": "Bearer valid.jwt.token"}
        )

        if response.status_code == status.HTTP_201_CREATED:
            assert response.json().get("user_id") != "victim-uuid"

@pytest.mark.security
class TestAPIKeySecurity:

    @pytest.fixture(autouse=True)
    def remove_auth_override(self):
        app.dependency_overrides.pop(verify_clerk_token, None)
        yield

    def test_api_key_not_accepted_as_clerk_token(self, client):
        response = client.get(
            "/jobs",
            headers={"Authorization": "Bearer test_api_key_user123"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_api_key_not_in_url(self, client):
        response = client.get("/jobs?api_key=test_api_key_user123")
        assert response.status_code in [
            status.HTTP_403_FORBIDDEN,
            status.HTTP_401_UNAUTHORIZED,
        ]

    @pytest.mark.parametrize("suspicious_token", [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "__import__('os').system('ls')",
        "",
        "Bearer",
        "null",
    ])
    @pytest.mark.asyncio
    async def test_malicious_token_rejected(self, suspicious_token):
        from jose import JWTError
        credentials = Mock(spec=HTTPAuthorizationCredentials)
        credentials.credentials = suspicious_token

        with patch("core.auth.get_jwks", new_callable=AsyncMock) as mock_jwks, \
             patch("core.auth.jwt.decode", side_effect=JWTError("invalid")):

            mock_jwks.return_value = {"keys": []}
            from core.auth import verify_clerk_token as real_verify
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await real_verify(credentials)
            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.security
class TestSessionManagement:

    def test_correlation_id_preserved_across_requests(self, client):
        correlation_id = "test-correlation-123"
        response = client.get(
            "/health",
            headers={
                "Authorization": "Bearer valid.jwt.token",
                "X-Correlation-ID": correlation_id,
            }
        )
        assert response.headers.get("X-Correlation-ID") == correlation_id

    def test_multiple_requests_with_same_token(self, client):
        for _ in range(5):
            response = client.get(
                "/health",
                headers={"Authorization": "Bearer valid.jwt.token"}
            )
            assert response.status_code == status.HTTP_200_OK

@pytest.mark.security
class TestUserProvisioning:

    def test_new_user_created_on_first_login(self):
        from core.auth import get_or_create_user
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with patch("core.auth.local_session", return_value=mock_db):
            mock_user = MagicMock()
            mock_user.id = "new-uuid"
            mock_user.clerk_id = "user_new123"
            mock_user.email = "new@example.com"
            mock_user.tier = "free"
            mock_db.refresh.side_effect = lambda u: None

            with patch("core.auth.User", return_value=mock_user):
                result = get_or_create_user("user_new123", "new@example.com")

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called()

    def test_existing_user_last_seen_updated(self):
        from core.auth import get_or_create_user
        from unittest.mock import MagicMock

        mock_existing_user = MagicMock()
        mock_existing_user.id = "existing-uuid"
        mock_existing_user.clerk_id = "user_existing123"
        mock_existing_user.email = "existing@example.com"
        mock_existing_user.tier = "free"
        mock_existing_user.last_seen_at = None

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_existing_user

        with patch("core.auth.local_session", return_value=mock_db):
            get_or_create_user("user_existing123", "existing@example.com")

        assert mock_existing_user.last_seen_at is not None
        mock_db.commit.assert_called()