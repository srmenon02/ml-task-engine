import pytest
from datetime import datetime, timedelta, timezone
from models import JobStatus

MOCK_USER_ID = "test-internal-uuid"


@pytest.mark.integration
class TestFiltering:

    def test_filter_by_status(self, client, auth_headers, job_factory):
        job_factory.create_batch(3, user_id=MOCK_USER_ID, status=JobStatus.COMPLETED)
        job_factory.create_batch(2, user_id=MOCK_USER_ID, status=JobStatus.FAILED)

        response = client.get(
            "/api/v1/jobs?status=completed",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3

    def test_filter_by_mutiple_status(self, client, auth_headers, job_factory):
        job_factory.create_batch(3, user_id=MOCK_USER_ID, status=JobStatus.COMPLETED)
        job_factory.create_batch(2, user_id=MOCK_USER_ID, status=JobStatus.FAILED)
        job_factory.create_batch(1, user_id=MOCK_USER_ID, status=JobStatus.PENDING)

        response = client.get(
            "/api/v1/jobs?status=completed&status=failed",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5

    def test_filter_by_date_range(self, client, auth_headers, job_factory):
        recent_date = datetime.now(timezone.utc) - timedelta(hours=2)
        recent_job = job_factory.create_batch(1, user_id=MOCK_USER_ID, created_at=recent_date)

        response = client.get(
            "/api/v1/jobs?date_range=last_24h",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == recent_job[0].id