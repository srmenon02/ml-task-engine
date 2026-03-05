
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime
from models import JobPriority, JobStatus

class JobCreate(BaseModel):
    job_type: str
    config: dict
    user_id: str = "default_user"
    priority: int = Field(default=JobPriority.NORMAL.value, ge=0, le=20)
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
    model_config = ConfigDict(
        use_enum_values = True,
        from_attributes = True
    )

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
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
