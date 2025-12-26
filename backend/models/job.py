from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
from .database import base

class JobStatus(str, Enum):
    PENDING = "pending"      
    RUNNING = "running"      
    COMPLETED = "completed"  
    FAILED = "failed"        
    TIMEOUT = "timeout"      
    RETRYING = "retrying"
    CANCELLED = "cancelled"  

class JobPriority(int, Enum):
    LOW = 0
    NORMAL = 5
    HIGH = 10
    URGENT = 20  

class Job(base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)

    config = Column(JSON, nullable=False)

    priority = Column(Integer, default=JobPriority.NORMAL.value, nullable=False, index=True)

    predicted_cpu_percent = Column(Float, nullable=True)
    predicted_memory_db = Column(Float, nullable=True)

    max_memory_mb = Column(Float, nullable=True)
    max_execution_time_sec = Column(Integer, default=3600, nullable=False)

    status = Column(SQLEnum(JobStatus), default = JobStatus.PENDING, nullable = False, index = True)
    error_msg = Column(String, nullable = True)

    cancelled_by = Column(String, nullable=True)
    cancceled_at = Column(DateTime, nullable=True)

    results = Column(JSON, nullable = True)

    created_at = Column(DateTime, default = func.now(), nullable = False)
    started_at = Column(DateTime, nullable = True)
    completed_at = Column(DateTime, nullable = True)

    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=5)

    def __repr__(self):
        return f"<Job id={self.id} type={self.job_type} status={self.status}>"
