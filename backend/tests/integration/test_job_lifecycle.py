import pytest
from models import Job, JobStatus, Execution
import structlog
from unittest.mock import patch
from workers.tasks import execute_job
from workers.tasks import _execute_job_with_limits, _store_resource_profile
import psutil
from datetime import datetime

logger = structlog.get_logger()

@pytest.mark.integration
class TestJobLifecycle:
    def test_job_pending_to_completed(self, client, auth_headers, test_db):        
        def mock_execute_job(job_id):
            job = test_db.query(Job).filter(Job.id == job_id).first()
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            test_db.commit()
            
            process = psutil.Process()
            cpu_samples = []
            memory_samples = []
            
            job_snapshot = {
                "id": job.id,
                "job_type": job.job_type,
                "config": job.config,
                "max_memory_mb": job.max_memory_mb,
                "max_execution_time_sec": job.max_execution_time_sec
            }
            
            result = _execute_job_with_limits(
                job_snapshot, process, cpu_samples, memory_samples, "test-worker", test_db
            )
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.results = result
            test_db.commit()
            
            return {"status": "success", "job_id": job_id, "result": result}
        
        with patch('workers.tasks.execute_job.delay', side_effect=lambda job_id: mock_execute_job(job_id)):
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
        
        test_db.expire_all()
        job = test_db.query(Job).filter(Job.id == job_id).first()
        
        assert job.status == JobStatus.COMPLETED
        assert job.results is not None