from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import structlog
import sys
from pathlib import Path
from datetime import datetime

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from workers.tasks import execute_job
from workers.celery_app import celery_app
from models import get_db, Job, JobStatus
from models import JobPriority
from pydantic import BaseModel, Field
from core.predictor import get_predictor
from core.accuracy_tracker import calculate_prediction_accuracy
from core.scheduler import get_scheduler
from core.worker_health import get_health_monitor

logger = structlog.get_logger()

app = FastAPI(
    title="Machine-Learning Task Engine API",
    description="Distributed Task Engine with ML-based Resource Prediction",
    version="0.2.0",
)

class JobCreate(BaseModel):
    job_type: str
    config: dict
    user_id: str = "default_user"
    priority: int = Field(defualt=JobPriority.NORMAL.value, ge=0, le=20)
    max_memory_mb: Optional[float] = None
    max_execution_time_sec: int = Field(default=3600, ge=60, le=86400)


    class Config:
        json_schema_extra = {
            "example": {
                "job_type": "train_sklearn_model",
                "config": {
                    "model": "RandomForest",
                    "n_estimators": 100,
                    "dataset_rows": 10000
                }
            }
        }

class JobResponse(BaseModel):
    id: int
    job_type: str
    status: JobStatus
    config: dict
    created_at: datetime
    results: Optional[dict] = None
    predicted_memory_db: Optional[float] = None
    predicted_cpu_percent: Optional[float] = None
    priority: int
    max_memory_mb: Optional[float] = None
    max_execution_time_sec: int
    cancelled_by: Optional[str]
    cancelled_at: Optional[datetime] = None
    error_message: str = None

    class Config:
        from_attributes = True

@app.get("/health")
def health_check():
    return {"status": "healthy"} 

@app.post("/jobs", response_model=JobResponse, status_code=201)
def create_job(job_data: JobCreate, db: Session = Depends(get_db)):
    logger.info("job.create requested", job_type=job_data.job_type, user_id=job_data.user_id)

    predictor = get_predictor()
    predicted_memory, predicted_cpu = predictor.predict(
        job_data.config,
        job_data.job_type
    )

    max_memory = job_data.max_memory_mb
    if max_memory is None:
        max_memory = predicted_memory * 2.0

    logger.info(
        "job resources predicted",
        memory_mb = predicted_memory,
        cpu_percent = predicted_cpu,
    )

    job = Job(
        job_type=job_data.job_type,
        config=job_data.config,
        user_id=job_data.user_id,
        status=JobStatus.PENDING,
        predicted_cpu_percent=predicted_cpu,
        predicted_memory_db=predicted_memory,
        priority=job_data.priority,
        max_memory_mb=max_memory,
        max_execution_time_sec=job_data.max_execution_time_sec,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    logger.info("job.created", job_id=job.id, predicted_memory_db=job.predicted_memory_db, predicted_cpu_percent=job.predicted_cpu_percent)
    execute_job.delay(job.id)

    logger.info("job queued", job_id = job.id)

    return job

@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job


@app.get("/jobs", response_model=List[JobResponse])
def list_job(
    status: JobStatus = None,
    jobs_limit: int = 100,
    db: Session = Depends(get_db)
):
    
    query = db.query(Job)

    if status:
        query = query.filter(Job.status == status)

    jobs = query.order_by(Job.created_at.desc()).limit(jobs_limit).all()

    return jobs

@app.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: int,
    cancelled_by: str = "api_user",
    db: Session = Depends(get_db)
):
    scheduler = get_scheduler()
    success = scheduler.cancel_job(job_id, cancelled_by=cancelled_by)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel job")
    
    return {"status": "cancelled", "job_id": job_id}

@app.get("/workers/health")
def get_workers_health():
    inspect = celery_app.control.inspect()

    stats = inspect.stats()
    active = inspect.active()
    registered = inspect.registered()

    if not stats:
        return {
            "total_workers": 0,
            "workers": [],
            "message": "No workers found. Ensure workers are running"
        }
    
    workers = []
    for worker_name, worker_stats in stats.items():
        workers.append({
            "worker_id": worker_name,
            "status": "active",
            "pool": worker_stats.get("pool", {}).get("implementation", "unkwown"),
            "max_concurrency": worker_stats.get("pool", {}).get("max-concurrency", 0),
            "active_jobs": len(active.get(worker_name, [])) if active else 0,
            "total_tasks": worker_stats.get("total", {}),
        })
    return {
        "total_workers": len(workers),
        "workers": workers,
    }

@app.get("/workers/active-jobs")
def get_active_jobs():
    inspect = celery_app.control.inspect()
    active = inspect.active()

    if not active:
        return {
            "active_jobs": 0,
            "jobs": []
        }
    all_jobs = []
    for worker_name, jobs in active.items():
        for job in jobs:
            all_jobs.append({
                "worker": worker_name,
                "task_id": job.get("id"),
                "task_name": job.get("name"),
                "args": job.get("args"),
                "time_start": job.get("time_start"),
            })
            
    return {
        "active jobs": len(all_jobs),
        "jobs": all_jobs,
    }
@app.get("/workers/{worker_id}/health")
def get_worker_health(worker_id: str):
    monitor = get_health_monitor()
    stats = monitor.get_worker_stats(worker_id)

    if "error" in stats:
        raise HTTPException(status_code=404, detail=stats["error"])

    return stats

@app.get("/system/stats")
def get_system_stats(db: Session = Depends(get_db)):
    total_jobs = db.query(func.count(Job.id)).scalar()
    completed_jobs = db.query(func.count(Job.id)).filter(
        Job.status == JobStatus.COMPLETED
    ).scalar()
    failed_jobs = db.query(func.count(Job.id)).filter(
        Job.status == JobStatus.FAILED
    ).scalar()
    pending_jobs = db.query(func.count(Job.id)).filter(
        Job.status == JobStatus.PENDING
    ).scalar()
    running_jobs = db.query(func.count(Job.id)).filter(
        Job.status == JobStatus.RUNNING
    ).scalar()

    monitor = get_health_monitor()
    workers = monitor.get_all_workers()
    active_workers = sum(1 for w in workers if w["status"] == "active")

    return {
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "pending": pending_jobs,
            "running": running_jobs,
            "success_rate": (completed_jobs / total_jobs * 100 if total_jobs > 0 else 0),
        },
        "workers": {
            "total": len(workers),
            "active": active_workers,
            "stale": len(workers) - active_workers
        },
    }

@app.post("/predictor/train")
def train_predictor():
    predictor = get_predictor()
    success = predictor.train(min_samples=5)

    if success:
        return {
            "status": "success",
            "message": "predictor trained",
            "training samples": predictor.training_samples,
        }
    return {
        "status": "failed",
        "message": "Need at least 5 completed jobs for training data"
    }

@app.get("/predictor/evaluate")
def evaluate_predictor():
    predictor = get_predictor()
    return predictor.evaluate()

@app.get("/predictor/accuracy")
def get_prediction_accuracy():
    return calculate_prediction_accuracy()

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




