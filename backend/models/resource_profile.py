from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.sql import func
from .database import base

class ResourceProfile(base):
    __tablename__ = "resource_profiles"

    id = Column(Integer, primary_key=True, index=True) 

    job_type = Column(String, nullable=False, index=True)
    config = Column(JSON, nullable=False)

    memory_mb = Column(Float, nullable=False)
    cpu_percent = Column(Float, nullable=False)
    execution_time = Column(Float, nullable=False)

    created_at = Column(DateTime, default=func.now(), nullable=False)

    def __repr__(self):
        return f"<ResourceProfile(job_type={self.job_type} memory={self.memory_mb})>"