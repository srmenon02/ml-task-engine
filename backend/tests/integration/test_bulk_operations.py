import pytest
from models import JobStatus
from tests.factories.job_factory import JobFactory
from models import Job, JobStatus

@pytest.mark.integration
class TestBulkOperations:
    def test_bulk_submit_success(self, client, auth_headers, test_db):
        response = client.post(
            "/api/v1/bulk/jobs",
            json = {
                "jobs": [
                    {
                        "job_type": "train_sklearn_model",
                        "config": {"n_estimators": 100, "dataset_rows": 1000},
                        "priority": 10
                    },
                    {
                        "job_type": "train_sklearn_model",
                        "config": {"n_estimators": 200, "dataset_rows": 2000}, 
                        "priority": 5
                    }
                ]
            },
            headers = auth_headers
        )

        assert response.status_code == 201
        data = response.json()
        assert data["total_submitted"] == 2
        assert data["total_failed"] == 0
        assert len(data["sucessful_job_ids"]) == 2

    def test_bulk_submit_partial_failure(self, client, auth_headers, test_db):
        response = client.post(
            "/api/v1/bulk/jobs",
            json = {
                "jobs": [
                    {
                        "job_type": "train_sklearn_model",
                        "config": {"n_estimators": 100, "dataset_rows": 1000},
                        "priority": 10
                    },
                    {
                        "job_type": "invalid_type",
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

    def test_bulk_cancel(self, client, auth_headers, job_factory, test_db):
        jobs = [
        Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100, "dataset_rows": 1000},
            user_id="user123",
            status=JobStatus.PENDING,
            priority=5
        )
        for _ in range(5)
    ]
    
        test_db.add_all(jobs)
        test_db.commit()

        
        job_ids = [j.id for j in jobs]
        for job in jobs:
            print(f"Job {job.id} status after creation: {job.status}")
        
        # Refresh from DB to see current state
        test_db.expire_all()
        for job_id in job_ids:
            job = test_db.query(Job).filter(Job.id == job_id).first()
            print(f"Job {job_id} status from DB: {job.status}")
    

        response = client.post(
            "/api/v1/bulk/jobs/cancel",
            json = {
                "job_ids": job_ids
            },
            headers = auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_cancelled"] == 5
