from typing import Dict, List
import numpy as np
import structlog
from models import local_session, Job, Execution, JobStatus

logger = structlog.get_logger()

def calculate_prediction_accuracy() -> Dict:
    db = local_session()

    try:
        print(f"Completed Jobs: {db.query(Job).filter(Job.predicted_memory_db.isnot(None)).all()}")
        print(f"Column Names: {Job.__table__.columns.keys()}")
        jobs = db.query(Job).filter(
            Job.status == JobStatus.COMPLETED,
            Job.predicted_memory_db.isnot(None)
        ).all()

        if len(jobs) == 0:
            return {"error": "no completed jobs w/ predictions"}
        
        memory_errors = []
        cpu_errors = []

        for job in jobs:
            execution = db.query(Execution).filter(
                Execution.job_id == job.id,
                Execution.success == 1,
            ).first()
            
            if not execution:
                continue

            if job.predicted_memory_db and execution.actual_memory_mb_max:
                memory_error = abs(job.predicted_memory_db - execution.actual_memory_mb_max) / execution.actual_memory_mb_max
                memory_errors.append(memory_error)

            if job.predicted_cpu_percent and execution.actual_cpu_percent_avg:
                cpu_error = abs(job.predicted_cpu_percent - execution.actual_cpu_percent_avg) / execution.actual_cpu_percent_avg
                cpu_errors.append(cpu_error)

        return {
            "total jobs": len(jobs),
            "memory_mape": float(np.mean(memory_errors) * 100) if memory_errors else None,
            "cpu_mape": float(np.mean(cpu_errors) * 100) if cpu_errors else None,
            "memory predictions within 20%": float(sum(1 for e in memory_errors if e < 0.2) / len(memory_errors) * 100) if memory_errors else None,
            "cpu predictions within 20%": float(sum(1 for e in cpu_errors if e < 0.2) / len(cpu_errors) * 100) if cpu_errors else None
        }
            
    except Exception as e:
        logger.error(f"Accuracy Tracker error {e}")
        return {"error": str(e)}
    
    finally:
        db.close()
