import pytest
from models import JobStatus
from factories.job_factory import JobFactory

@pytest.mark.integration
class TestBulkOperations:
    def test_bulk_submit_success(self, client, auth_headers, test_db):
        response = client.post(
            "/api/v1/bulk/jobs",
            json = {
                "jobs": [
                    {
                        "jobs_type": "train_sklearn_model",
                        "config": {"n_estimators": 100, "dataset_rows": 1000},
                        "priority": 10
                    },
                    {
                        "jobs_type": "train_sklearn_model",
                        "config": {"n_estimators": 200, "dataset_rows": 2000}, 
                        "priority": 5
                    }
                ]
            },
            headers = auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_submitted"] == 2
        assert data["total_failed"] == 0
        assert len(data["successful_job_ids"]) == 2

    def test_bulk_submit_partial_failure(self, client, auth_headers, test_db):
        response = client.post(
            "/api/v1/bulk/jobs",
            json = {
                "jobs": [
                    {
                        "jobs_type": "train_sklearn_model",
                        "config": {"n_estimators": 100, "dataset_rows": 1000},
                        "priority": 10
                    },
                    {
                        "jobs_type": "invalid_type",
                        "config": {"n_estimators": 200, "dataset_rows": 2000}, 
                        "priority": 5
                    }
                ]
            },
            headers = auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["total_submitted"] == 1
        assert data["total_failed"] == 1

    def test_bulk_cancel(self, client, auth_headers, test_db):
        jobs = JobFactory.create_batch(5, user_id = "user123", status = JobStatus.PENDING)
        test_db.add_all(jobs)
        test_db.commit()

        job_ids = [job.id for job in jobs]

        response = client.delete(
            "/api/v1/bulk/jobs",
            json = {
                "job_ids": job_ids
            },
            headers = auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_canclled"] == 5
