from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.sql import func
from .database import Base

class Execution(Base):
    __tablename__ = "execution"

    id = Column(Integer, primary_key=True, index=True)

    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False, index=True)

    worker_id = Column(String, nullable = True)
    execution_num = Column(Integer, nullable = False)

    actual_cpu_percent_avg = Column(Float, nullable = True)
    actual_cpu_percent_max = Column(Float, nullable = True)
    actual_memory_mb_avg = Column(Float, nullable = True)
    actual_memory_mb_max = Column(Float, nullable = True)

    started_at = Column(DateTime, default = func.now(), nullable = False)
    completed_at = Column(DateTime, nullable = True)

    success = Column(Integer, default=0)
    error_msg = Column(String, nullable=True)

    def __repr__(self):
        return f"<Execution id={self.id} job_id={self.job_id} attempt={self.execution_num}>"