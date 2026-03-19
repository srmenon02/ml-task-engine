import pytest
from fastapi import status
from unittest.mock import patch, AsyncMock
from models import Job, JobStatus
from core.auth import verify_clerk_token
from api.main import app

MOCK_AUTH = {
    "user_id": "test-internal-uuid",
    "clerk_id": "user_test123",
    "email": "test@example.com",
    "tier": "free",
    "permissions": ["read", "write"],
}


class TestJobAPI:
    def test_create_job_authenticated(self, client, auth_headers, test_db):
        response = client.post(
            "/jobs",
            json={
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "RandomForest",
                    "n_estimators": 100,
                    "dataset_rows": 10000,
                },
                "priority": 10,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["job_type"] == "train_sklearn_model"
        assert data["status"] == JobStatus.PENDING
        job = test_db.query(Job).filter(Job.id == data["id"]).first()
        assert job is not None

    def test_create_job_unauthenticated(self, client):
        app.dependency_overrides.pop(verify_clerk_token, None)
        try:
            response = client.post(
                "/jobs",
                json={
                    "job_type": "train_sklearn_model",
                    "config": {"n_estimators": 100},
                },
            )
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_403_FORBIDDEN,
            ]
        finally:
            # Restore the override so other tests are not affected
            async def override_verify_clerk_token(credentials=None):
                return MOCK_AUTH
            app.dependency_overrides[verify_clerk_token] = override_verify_clerk_token

    @pytest.mark.security
    def test_create_job_with_malicious_config(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json={
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "'; DROP TABLE jobs; --",
                    "n_estimators": 100,
                },
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.parametrize("model,extra_config", [
        ("RandomForest",       {"n_estimators": 10}),
        ("GradientBoosting",   {"n_estimators": 10}),
        ("LogisticRegression", {"max_iter": 100}),
        ("SVC",                {"C": 1.0}),
        ("DecisionTree",       {"max_depth": 3}),
        ("KNeighbors",         {"n_neighbors": 3}),
    ])
    def test_all_model_types_accepted(self, client, auth_headers, model, extra_config):
        response = client.post(
            "/jobs",
            json={
                "job_type": "train_sklearn_model",
                "config": {
                    "model": model,
                    "dataset_rows": 100,
                    **extra_config,
                },
                "priority": 5,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["status"] == "pending"