import pytest
from datetime import datetime, timedelta
from models import JobStatus

@pytest.mark.integration
class TestFiltering:

    def test_filter_by_status(self, client, auth_headers, job_factory):
        job_factory.create_batch(3, user_id = "user123", status = JobStatus.COMPLETED)
        job_factory.create_batch(2, user_id = "user123", status = JobStatus.FAILED)
        #test_db.commit()

        response = client.get(
            "/api/v1/jobs?status=completed",
            headers = auth_headers
        )

        assert response.status_code == 200
        data =response.json()
        assert data["total"] == 3

    def test_filter_by_mutiple_status(self, client, auth_headers, job_factory):
        job_factory.create_batch(3, user_id = "user123", status = JobStatus.COMPLETED)
        job_factory.create_batch(2, user_id = "user123", status = JobStatus.FAILED)
        job_factory.create_batch(1, user_id = "user123", status = JobStatus.PENDING)
        #test_db.commit()

        response = client.get(
            "/api/v1/jobs?status=completed&status=failed",
            headers = auth_headers
        )

        assert response.status_code == 200
        data =response.json()
        assert data["total"] == 5

    def test_filter_by_date_range(self, client, auth_headers, job_factory):
        old_date = datetime.now() - timedelta(days = 10)
        recent_date = datetime.now() - timedelta(hours = 2)

        old_job = job_factory.create_batch(1, user_id = "user123", created_at = old_date)
        recent_job = job_factory.create_batch(1, user_id = "user123", created_at = recent_date)
        #test_db.commit()

        response = client.get(
            f"/api/v1/jobs?date_range=last_24h",
            headers = auth_headers
        )

        assert response.status_code == 200
        data =response.json()
        assert data["total"] == 2
        assert data["items"][0]["id"] == recent_job[0].id


