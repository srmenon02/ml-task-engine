from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from models import get_db
from core.auth import verify_api_key
from core.bulk_operations import bulk_submit_jobs, bulk_cancel_jobs, BulkJobResult
from core.rate_limiter import TieredRateLimiter, UserTier

router = APIRouter(prefix = "/bulk", tags = ["Bulk Operations"])

class BulkJobSubmitRequest(BaseModel):
    jobs: List[Dict[str, Any]] = Field(
        description = "List of job configurations",
        min_items = 1,
        max_items = 100
    )

    class Config:
        json_schema_extra = {
            "example": {
                "jobs": [
                    {
                        "job_type": "train_sklearn_model",
                        "config": {"n_estimators": 100, "dataset_rows": 10000},
                        "priority": 10
                    },
                    {
                        "job_type": "train_sklearn_model",
                        "config": {"n_estimators": 200, "dataset_rows": 20000}, 
                        "priority": 5
                    }
                ]
            }
        }

class BulkCancelRequest(BaseModel):
    job_ids: List[int] = Field(
        description = "List of job IDs to cancel",
        min_items = 1,
        max_items = 100
    )

@router.post("/jobs", response_model = BulkJobResult, status_code = status.HTTP_201_CREATED)
def submit_bulk_jobs(
    request: BulkJobSubmitRequest,
    auth: dict = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    rate_limiter = TieredRateLimiter()

    user_tier = UserTier(auth.get("tier", "free"))

    allowed, info = rate_limiter.is_allowed(
        user_id = auth["user_id"],
        user_tier = user_tier,
        cost = len(request.jobs)
    )

    if not allowed:
        raise HTTPException(
            status_code = 429,
            detail = {
                "error": "Rate Limit exceeded",
                "limit": info["limit"],
                "tier": info["tier"],
                "retry_after": info["retry_after"]
            }
        )
    
    if len(request.jobs) > 100:
        raise HTTPException(
            status_code = 400,
            detail = "Maximum of 100 jobs per bulk request"
        )
    
    result = bulk_submit_jobs(
        job_configs = request.jobs,
        user_id = auth["user_id"],
        db = db
    )

    return result

@router.post("/jobs/cancel")
def cancel_bulk_jobs(
    request: BulkCancelRequest,
    auth: dict = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    if len(request.job_ids) > 100:
        raise HTTPException(
            status_code = 400,
            detail = "Maximum of 100 jobs per bulk cancellation"
        )
    
    result = bulk_cancel_jobs(
        job_ids = request.job_ids,
        user_id = auth["user_id"],
        db = db
    )

    return result