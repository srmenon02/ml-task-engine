from typing import Optional
import structlog
from datetime import datetime

from models import local_session, Job, JobStatus, JobPriority
from workers.tasks import execute_job

from celery.result import AsyncResult


logger = structlog.get_logger()

class JobScheduler:
    def submit_job(self, job_id: int, priority: Optional[int] = None) -> bool:
        db = local_session()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                logger.error("Scheduler.job not found", job_id = job_id)
                return False
            
            task_priority = priority if priority is not None else job.priority
            execute_job.apply_async(
                args = [job_id],
                priority = task_priority,
            )

            logger.info(
                "Scheduler job submitted",
                job_id = job_id,
                priority = task_priority
            )

            return True
        except Exception as e:
            logger.error(f"Scheduler submit job failedL {e}", job_id = job_id)
            return False
        finally:
            db.close()

    def cancel_job(self, job_id: int, cancelled_by: str = "system") -> bool:
        db = local_session()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                logger.error("Scheduler.job not found", job_id = job_id)
                return False
            if job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
                logger.warning(
                    "Cannot cancel job",
                    job_id = job.id,
                    job_status = job.status,
                )

                return False
            
            job.status = JobStatus.CANCELLED
            job.cancelled_by = cancelled_by
            job.cancceled_at = datetime.now()
            db.commit()

            if job.status == JobStatus.RUNNING:
                pass

            logger.info(
                "scheduler job successfuly cancelled",
                job_id = job.id,
                cancelled_by = cancelled_by,
            )

            return True
        except Exception as e:
            logger.error(f"Scheduler cancel job failed: {e}")
            return False
        
        finally:
            db.close()

_scheduler = None

def _get_scheduler() -> JobScheduler:
    global _scheduler
    if not _scheduler:
        _scheduler = JobScheduler()
    return _scheduler




