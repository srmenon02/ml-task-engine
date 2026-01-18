from typing import List, Dict, Any
from sqlalchemy.orm import Session
import structlog

from models import Job, JobStatus
from core.predictor import get_predictor
from core.security import get_validator
from core.scheduler import get_scheduler
from workers.tasks import execute_job
from pydantic import BaseModel

logger = structlog.get_logger()

class BulkJobResult(BaseModel):    
    total_submitted: int
    total_failed: int
    sucessful_job_ids: List[int]
    failed_jobs: List[Dict[str, Any]]

def bulk_submit_jobs(
        job_configs: List[Dict[str, Any]],
        user_id: str,
        db: Session
) -> BulkJobResult:
    validator = get_validator()
    predictor = get_predictor()

    successful_ids = []
    failed_jobs = []

    for idx, config in enumerate(job_configs):
        try:
            job_type = config.get("job_type")
            job_config = config.get("config", {})

            is_valid, error_msg = validator.validate_job(job_type, job_config)
            if not is_valid:
                failed_jobs.append({
                    "index": idx,
                    "config": config,
                    "error": error_msg
                })
                continue

            predicted_memory, predicted_cpu = predictor.predict(job_config, job_type)

            job = Job(
                job_type = job_type,
                config = job_config,
                user_id = user_id,
                status = JobStatus.PENDING,
                predicted_cpu_percent = predicted_cpu,
                predicted_memory_db = predicted_memory,
                priority = config.get("priority", 5),
                max_memory_mb = config.get("max_memory_mb", predicted_memory * 2),
                max_execution_time_sec = config.get("max_execution_time_sec", 3600)
            )

            db.add(job)
            db.flush()

            successful_ids.append(job.id)
        except Exception as e:
            logger.error(f"Bulk submit error for job {idx}: {e}")
            failed_jobs.append({
                "index": idx,
                "config": config,
                "error": e
            })

    db.commit()

    for job_id in successful_ids:
        execute_job.delay(job_id)

    logger.info(
        "Bulk Job submitted",
        total = len(job_configs),
        successful = len(successful_ids),
        failed = len(failed_jobs),
        user_id = user_id
    )

    return BulkJobResult(
        total_submitted = len(successful_ids),
        total_failed = len(failed_jobs),
        sucessful_job_ids = successful_ids,
        failed_jobs = failed_jobs
    )

def bulk_cancel_jobs(
        job_ids: List[int],
        user_id: str,
        cancelled_by: str,
        db: Session
) -> Dict[str, Any]:
    scheduler = get_scheduler()

    cancelled = []
    failed = []

    for job_id in job_ids:
        job = db.query(Job).filter(
            Job.id == job_id,
            Job.user_id == user_id
        ).first()

        if not job:
            failed.append({
                "job_id": job_id,
                "error": "Job not found or unauthorized"
            })
            continue
        
        success = scheduler.cancel_job(job_id, cancelled_by = cancelled_by)

        if success:
            cancelled.append(job_id)
        else:
            failed.append({
                "job_id": job_id,
                "error": "Failed to cancel job (already completed/cancelled)"
            })

    logger.info(
        "Bulk Jobs cancelled",
        total = len(job_ids),
        cancelled = len(cancelled),
        failed = len(failed),
        user_id = user_id
    )

    return {
        "total_cancelled": len(cancelled),
        "total_failed": len(failed),
        "cancelled_job_ids": cancelled,
        "failed_cancellations": failed
    }


    
