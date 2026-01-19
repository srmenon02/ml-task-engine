from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from sqlalchemy.orm import Query
from sqlalchemy import cast, String
from models import Job, JobStatus

class JobFilters(BaseModel):
    status: Optional[List[JobStatus]] = Field(None, description = "Filter by Status")
    job_type: Optional[str] = Field(None, description = "Filter by Job Type")
    priority_min: Optional[int] = Field(None, ge=0, le=20, description = "Minimum Priority")
    priority_max: Optional[int] = Field(None, ge=0, le=20, description = "Maximum Priority")
    created_afer: Optional[datetime] = Field(None, description = "Created After Timestamp")
    created_before: Optional[datetime] = Field(None, description = "Created Before Timestamp")
    search: Optional[str] = Field(None, description = "Search in config (JSON)")

def apply_job_filters(query: Query, filters: JobFilters, user_id: str) -> Query:
    query = query.filter(Job.user_id == user_id)

    if filters.status:
        status_values = [s.value if isinstance(s, JobStatus) else s for s in filters.status]
        query = query.filter(Job.status.in_(status_values))

    if filters.job_type:
        query = query.filter(Job.job_type == filters.job_type)

    if filters.priority_min:
        query = query.filter(Job.priority >= filters.priority_min)

    if filters.priority_max:
        query = query.filter(Job.priority <= filters.priority_max)

    if filters.created_afer:
        query = query.filter(Job.created_at >= filters.created_afer)

    if filters.created_before:
        query = query.filter(Job.created_at <= filters.created_before)

    if filters.search:
        try:
            query = query.filter(
                cast(Job.config, String).ilike(f"%{filters.search}%")
            )
        except:
            pass

    return query

class DateRangePreset(str):
    LAST_HOUR = "last_hour"
    LAST_24H = "last_24h"
    LAST_7D = "last_7d"
    LAST_30D = "last_30d"

def get_date_range_from_preset(preset: DateRangePreset) -> tuple[datetime, datetime]:
    now = datetime.now()

    ranges = {
        DateRangePreset.LAST_HOUR: timedelta(hours = 1),
        DateRangePreset.LAST_24H: timedelta(days = 1),
        DateRangePreset.LAST_7D: timedelta(days = 7),
        DateRangePreset.LAST_30D: timedelta(days = 30),
    }

    delta = ranges.get(preset, timedelta(days = 1))
    return now - delta, now


