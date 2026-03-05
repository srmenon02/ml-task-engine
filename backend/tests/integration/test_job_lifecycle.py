import pytest
from models import Job, JobStatus, Execution
import structlog
from unittest.mock import patch
from workers.tasks import execute_job
from workers.tasks import _execute_job_with_limits, _store_resource_profile
import psutil
from datetime import datetime, timezone
import time
from unittest.mock import patch
logger = structlog.get_logger()

@pytest.mark.integration
class TestJobLifecycle:    
    def test_job_pending_to_completed(self, client, auth_headers, test_db):
        from unittest.mock import patch
        
        queued_jobs = []
        
        def mock_delay(job_id):
            queued_jobs.append(job_id)
            job = test_db.query(Job).filter(Job.id == job_id).first()
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            test_db.commit()
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.results = {
                "model_type": "RandomForest",
                "accuracy": 0.95,
                "training_time": 1.5
            }
            test_db.commit()
        
        with patch('workers.tasks.execute_job.delay', side_effect=mock_delay):
            response = client.post(
                "/jobs",
                json={
                    "job_type": "train_sklearn_model",
                    "config": {
                        "model": "RandomForest",
                        "n_estimators": 10,
                        "dataset_rows": 100,
                        "n_features": 5
                    }
                },
                headers=auth_headers
            )
            
            assert response.status_code == 201
            job_id = response.json()["id"]
            
            assert job_id in queued_jobs
        
        test_db.expire_all()
        job = test_db.query(Job).filter(Job.id == job_id).first()
        
        assert job.status == JobStatus.COMPLETED
        assert job.started_at is not None
        assert job.completed_at is not None
        assert job.results is not None