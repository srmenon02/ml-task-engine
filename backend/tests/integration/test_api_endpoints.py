import pytest
from fastapi import status
from models import Job, JobStatus

class TestJobAPI:
    def test_create_job_authenticated(self, client, auth_headers, test_db):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "RandomForest",
                    "n_estimators": 100,
                    "dataset_rows": 10000
                },
                "priority": 10
            },
            headers = auth_headers
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["job_type"] == "train_sklearn_model"
        assert data["status"] == JobStatus.PENDING

        job = test_db.query(Job).filter(Job.id == data["id"]).first()
        assert job is not None

    def test_create_job_unauthenticated(self, client):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {"n_estimators": 100}
            }
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.security
    def test_create_job_with_malicious_config(self, client, auth_headers):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "'; DROP TABLE jobs; --",
                    "n_estimators": 100
                }
            },
            headers = auth_headers
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST