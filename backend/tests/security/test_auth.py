import pytest
from fastapi import status, HTTPException
from core.auth import verify_api_key, load_api_keys, hash_api_key
from fastapi.security import HTTPAuthorizationCredentials
from unittest.mock import Mock
import time
from models import Job, JobStatus
import base64
@pytest.mark.security
class TestAPIKeyValidation:
    def test_valid_api_key_accepted(self):
        credentials = Mock(spec = HTTPAuthorizationCredentials)
        credentials.scheme = "Bearer"
        credentials.credentials = "test_api_key_user123"

        result = verify_api_key(credentials)

        assert result is not None
        assert "user_id" in result

    def test_invailid_api_key_rejected(self):
        credentials = Mock(spec = HTTPAuthorizationCredentials)
        credentials.scheme = "Bearer"
        credentials.credentials = "invalid_api_key"

        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_missing_credentials_rejected(self):
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(None)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_empty_api_key_rejected(self):
        credentials = Mock(spec = HTTPAuthorizationCredentials)
        credentials.scheme = "Bearer"
        credentials.credentials = ""

        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    def test_api_key_extracts_user_id(self):
        credentials = Mock(spec = HTTPAuthorizationCredentials)
        credentials.scheme = "Bearer"
        credentials.credentials = "test_api_key_user123"

        result = verify_api_key(credentials)

        assert result is not None
        assert "user_id" in result
        assert result["user_id"] == "user123"

    def test_api_key_includes_permissions(self):
        credentials = Mock(spec = HTTPAuthorizationCredentials)
        credentials.scheme = "Bearer"
        credentials.credentials = "test_api_key_user123"

        result = verify_api_key(credentials)

        assert result is not None
        assert "permissions" in result
        assert isinstance(result["permissions"], list)

@pytest.mark.security
class TestBearerTokenFormat:
    def test_accepts_bearer_token(self, client):
        response = client.get(
            "/health",
            headers = {"Authorization": "Bearer test_api_key_user123"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_rejects_basic_auth(self, client):
        creds = base64.b64encode(b"user:pass").decode()
        response = client.get(
            "/jobs",
            headers = {"Authorization": f"Basic {creds}"}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_rejects_malformed_bearer(self, client):
        malformed_tokens = [
            "Bearer",
            "Bearer ",
            "Bearertest_api_key_user123",
            "Bearer test_api_key_user123 extra"
        ]

        for token in malformed_tokens:
            response = client.get(
                "/jobs",
                headers = {"Authorization": token}
            )

            assert response.status_code == status.HTTP_401_UNAUTHORIZED

@pytest.mark.security
class TestAPIKeySecurity:
    def test_api_key_not_in_url(self, client):
        response = client.get("/jobs?api_key=test_api_key_user123")
        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_401_UNAUTHORIZED]

    def test_api_key_not_in_query_string(self, client, auth_headers):
        response = client.get("/jobs?page-1", headers=auth_headers)
        assert "api_key" not in str(response.url).lower()

    @pytest.mark.parametrize("suspicious_key", [
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "__import__('os').system('ls')",
    ])
    def test_api_key_injection_attempts_blocked(self, suspicious_key):
        credentials = Mock(spec = HTTPAuthorizationCredentials)
        credentials.scheme = "Bearer"
        credentials.credentials = suspicious_key

        with pytest.raises(HTTPException) as exc_info:
            verify_api_key(credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_api_key_timing_attack_resistance(self):
        credentials_valid = Mock(spec = HTTPAuthorizationCredentials)
        credentials_valid.scheme = "Bearer"
        credentials_valid.credentials = "test_api_key_user123"

        credentials_invalid = Mock(spec = HTTPAuthorizationCredentials)
        credentials_invalid.scheme = "Bearer"
        credentials_invalid.credentials = "invalid_api_key"

        start_time_valid = time.time()
        try:
            verify_api_key(credentials_valid)
        except:
            pass
        verify_api_key(credentials_valid)
        valid_time = time.time() - start_time_valid

        start_time_invalid = time.time()
        try:
            verify_api_key(credentials_invalid)
        except:
            pass
        invalid_time = time.time() - start_time_invalid

        assert abs(invalid_time - valid_time) < 0.1

@pytest.mark.security
class TestAuthorizationChecks:
    def test_user_can_only_see_own_jobs(self, client, test_db):
        job_user1 = Job(
            job_type = "train_sklearn_model",
            config = {
                "n_estimators": 100,
            },
            priority = 5,
            user_id = "user1"
        )

        job_user2 = Job(
            job_type = "train_sklearn_model",
            config = {
                "n_estimators": 100,
            },
            priority = 5,
            user_id = "user2"
        )

        test_db.add_all([job_user1, job_user2])
        test_db.commit()

        headers_user1 = {"Authorization": "Bearer test_api_key_user1"}
        response = client.get(f"/jobs/{job_user1.id}", headers=headers_user1)
        assert response.status_code == status.HTTP_200_OK

        response = client.get(f"/jobs/{job_user2.id}", headers=headers_user1)
        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_401_UNAUTHORIZED]

    def test_permission_based_access(self):
        credentials = Mock(spec = HTTPAuthorizationCredentials)
        credentials.scheme = "Bearer"
        credentials.credentials = "test_api_key_user123"

        result = verify_api_key(credentials)

        assert "read" in result["permissions"]
        assert "write" in result["permissions"]

    def test_cannot_impersonate_other_users(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 100,
                },
                "user_id": "victim_user"
            },
            headers = auth_headers
        )

        if response.status_code == status.HTTP_201_CREATED:
            job_data = response.json()
            assert "user_id" not in job_data

@pytest.mark.security
class TestSessionManagement:
    def test_api_key_does_not_expire_during_request(self, client, auth_headers):
        for _ in range(5):
            response = client.get("/health", headers = auth_headers)
            assert "X-Correlation-ID" in response.headers

    def test_correlation_id_preservd_across_requests(self, client, auth_headers):
        correlation_id = "test-correlation-123"
        headers = {
            **auth_headers,
            "X-Correlation-ID": correlation_id
        }

        response = client.get("/health", headers = headers)

        assert response.headers.get("X-Correlation-ID") == correlation_id

@pytest.mark.security
class TestAPIKeyManagement:
    def test_load_api_keys_from_env(self, monkeypatch):
        monkeypatch.setenv("API_KEYS", "key1_user1, key2_user2, key3_user3")

        keys = load_api_keys()

        assert len(keys) == 3
        assert "key1_user1" in keys
        assert "key2_user2" in keys
        assert "key3_user3" in keys

        assert keys["key1_user1"]["user_id"] == "user1"
        assert keys["key2_user2"]["user_id"] == "user2" 
        assert keys["key3_user3"]["user_id"] == "user3"

    def test_api_key_hash_consistent(self):
        key = "test_api_key_user123"

        hash1 = hash_api_key(key)
        hash2 = hash_api_key(key)

        assert hash1 == hash2

    def test_api_key_hash_unique(self):
        key1 = "test_api_key_user123"
        key2 = "test_api_key_user456"

        hash1 = hash_api_key(key1)
        hash2 = hash_api_key(key2)

        assert hash1 != hash2
