import pytest
import time
from models import Job, JobStatus, Execution

@pytest.mark.integration
class TestJobLifecycle:
    def test_job_pending_to_completed(self, client, auth_headers, test_db, celery_worker):
        response = client.post(
            "/jobs",
            json = {
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "RandomForest",
                    "n_estimators": 10,
                    "dataset_rows": 100
                }
            },
            headers = auth_headers
        )

        job_id = response.json()["id"]

        time.sleep(2)

        job = test_db.query(Job).filter(Job.id == job_id).first()
        assert job.status == JobStatus.COMPLETED
        assert job.started_at is not None
        assert job.completed_at is not None
        assert job.results is not None