from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import structlog
import sys
from pathlib import Path
from datetime import datetime

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from workers.tasks import execute_job
from models import get_db, Job, JobStatus
from pydantic import BaseModel
from core.predictor import get_predictor
from core.accuracy_tracker import calculate_prediction_accuracy

logger = structlog.get_logger()

app = FastAPI(
    title="Machine-Learning Task Engine API",
    description="Distributed Task Engine with ML-based Resource Prediction",
    version="0.1.0",
)

class JobCreate(BaseModel):
    job_type: str
    config: dict
    user_id: str = "default_user"


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
    predicted_memory_db: float = None
    predicted_cpu_percent: float = None

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
        predicted_memory_db = predicted_memory,
        predicted_cpu_percent = predicted_cpu,
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

@app.get("/prediction/accuracy")
def get_prediction_accuracy():
    return calculate_prediction_accuracy()

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




