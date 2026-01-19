import pytest
from unittest.mock import patch, Mock
from datetime import datetime
from models import Job, JobStatus, JobPriority
from core.scheduler import JobScheduler, get_scheduler

@pytest.mark.unit
class TestJobScheduler:
    @pytest.fixture
    def scheduler(self):
        return JobScheduler()
    
    @pytest.fixture
    def sample_job(self, test_db):
        job = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100},
            user_id="test_user",
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL.value,
        )

        test_db.add(job)
        test_db.commit()
        test_db.refresh(job)
        return job
    
    @patch('core.scheduler.local_session')
    @patch('workers.tasks.execute_job.apply_async')
    def test_submit_job_success(self, mock_apply_async, mock_session, scheduler, sample_job, test_db):
        mock_session.return_value = test_db
        success = scheduler.submit_job(sample_job.id)
        
        assert success is True
        mock_apply_async.assert_called_once()
        call_args = mock_apply_async.call_args
        assert call_args[1]['args'] == [sample_job.id]

    @patch('core.scheduler.local_session')
    def test_submit_nonexistent_job(self, mock_session, scheduler, test_db):
        mock_session.return_value = test_db
        success = scheduler.submit_job(9999)
        
        assert success is False

    @patch('workers.tasks.execute_job.apply_async')
    @patch('core.scheduler.local_session')
    def test_submit_job_with_custom_priority(self, mock_session, mock_apply_async, scheduler, sample_job, test_db):
        mock_session.return_value = test_db

        custom_priority = 15
        success = scheduler.submit_job(sample_job.id, priority = custom_priority)
        
        assert success is True
        call_args = mock_apply_async.call_args
        assert call_args[1]['priority'] == custom_priority

    @patch('core.scheduler.local_session')
    def test_cancel_pending_job(self, mock_session, scheduler, sample_job, test_db):
        mock_session.return_value = test_db
        success = scheduler.cancel_job(sample_job.id, cancelled_by = "admin")
        
        assert success is True
        updated_job = test_db.get(Job, sample_job.id)
        assert updated_job.status == JobStatus.CANCELED
        assert updated_job.cancelled_by == "admin"
        assert updated_job.cancelled_at is not None

    @patch('core.scheduler.local_session')
    def test_cannot_cancel_completed_job(self, mock_session, scheduler, test_db):
        mock_session.return_value = test_db
        completed_job = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100},
            user_id="test_user",
            status=JobStatus.COMPLETED,
            priority=JobPriority.NORMAL.value,
        )

        test_db.add(completed_job)
        test_db.commit()

        success = scheduler.cancel_job(completed_job.id)
        
        assert success is False
        assert completed_job.status == JobStatus.COMPLETED

    @patch('core.scheduler.local_session')
    def test_cannot_cancel_already_cancelled_job(self, mock_session, scheduler, test_db):
        mock_session.return_value = test_db

        cancelled_job = Job(
            job_type="train_sklearn_model",
            config={"n_estimators": 100},
            user_id="test_user",
            status=JobStatus.CANCELED,
            priority=JobPriority.NORMAL.value,
            cancelled_by="user",
            cancelled_at=datetime.now(),
        )

        test_db.add(cancelled_job)
        test_db.commit()

        success = scheduler.cancel_job(cancelled_job.id)

        assert success is False

    @patch('core.scheduler.local_session')
    def test_cancel_nonexistent_job(self, mock_session, scheduler, test_db):
        mock_session.return_value = test_db
        success = scheduler.cancel_job(job_id = 9999)
        
        assert success is False

    @pytest.mark.security
    @patch('core.scheduler.local_session')
    def test_cancel_tracks_who_cancelled(self, mock_session, scheduler, sample_job, test_db):
        mock_session.return_value = test_db

        scheduler.cancel_job(sample_job.id, cancelled_by = "admin")

        updated_job = test_db.get(Job, sample_job.id)
        assert updated_job.cancelled_by == "admin"
        assert updated_job.cancelled_at is not None

@pytest.mark.unit
class TestSchedulerSiingleton:
    def test_get_scheduler_returns_same_instance(self):
        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()

        assert scheduler1 is scheduler2

    def test_scheduler_persists_across_calls(self):
        scheduler = get_scheduler()
        scheduler.test_attribute = "test_value"

        scheduler2 = get_scheduler()
        assert scheduler2.test_attribute == "test_value"
   