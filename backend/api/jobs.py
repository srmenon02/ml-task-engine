from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List

from models import get_db, Job, JobStatus
from api.pagination import PaginationParams, PaginatedResponse, paginate_query
from api.models import JobResponse
from core.auth import verify_api_key
from core.filtering import JobFilters, apply_job_filters, DateRangePreset, get_date_range_from_preset
from typing import Optional, List
from datetime import datetime

router = APIRouter(prefix = "/jobs", tags = ["Jobs"])

@router.get("", response_model = PaginatedResponse[JobResponse])
def list_jobs(
    page: int = Query(1, ge = 1, description = "Page #"),
    page_size: int = Query(50, ge = 1, le = 100, description = "Items/page"),

    status: Optional[List[JobStatus]] = Query(None, description = "Filter by Status"),
    job_type: Optional[str] = Query(None, description = "Filter by Job Type"),
    priority_min: Optional[int] = Query(None, ge = 0, le = 20, description = "Minimum Priority"),
    priority_max: Optional[int] = Query(None, ge = 0, le = 20, description = "Maximum Priority"),
    created_after: Optional[datetime] = Query(None, description = "Created After Timestamp"),
    created_before: Optional[datetime] = Query(None, description = "Created Before Timestamp"),
    date_range: Optional[str] = Query(None, description = "Preset: last_hour, last_24h, last_7d, last_30d"),
    search: Optional[str] = Query(None, description = "Search in job config"),

    auth: dict = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    params = PaginationParams(page = page, page_size = page_size)

    final_created_after = created_after
    final_created_before = created_before

    if date_range:
        try:
            final_created_after, final_created_before = get_date_range_from_preset(date_range)
        except:
            pass

    filters = JobFilters(
        status = status,
        job_type = job_type,
        priority_min = priority_min,
        priority_max = priority_max,
        created_after = final_created_after,
        created_before = final_created_before,
        search = search
    )

    query = db.query(Job)
    query = apply_job_filters(query, filters, auth["user_id"])
    query = query.order_by(Job.created_at.desc())

    items, total = paginate_query(query, params)

    return PaginatedResponse.create(
        items = [JobResponse.model_validate(job) for job in items],
        total = total,
        params = params
    )