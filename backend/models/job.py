from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
from .database import Base

class JobStatus(str, Enum):
    PENDING = "pending"      
    RUNNING = "running"      
    COMPLETED = "completed"  
    FAILED = "failed"        
    TIMEOUT = "timeout"      
    RETRYING = "retrying"    

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_type = Column(String, nullabke=False, index=True)
    user_id = Column(String, nullable=False, index=True)

    config = Column(JSON, nullable=False)

    predicted_cpu_percent = Column(Float, nullable=True)
    predicted_memory_db = Column(Float, nullable=True)

    status = Column(SQLEnum(JobStatus), default = JobStatus.PENDING, nullable = False, index = True)
    error_msg = Column(String, nullable = True)

    results = Column(JSON, nullable = True)

    created_at = Column(DateTime, default = func.now(), nullable = False)
    started_at = Column(DateTime, nullable = True)
    completed_at = Column(DateTime, nullable = True)

    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=5)

    def __repr__(self):
        return f"<Job id={self.id} type={self.job_type} status={self.status}>"
