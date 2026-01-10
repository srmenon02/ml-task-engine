import structlog
from typing import Dict, List
from datetime import datetime, timedelta
from sqlalchemy import func, case
from models import local_session, Job, Execution, JobStatus

logger = structlog.get_logger()

class JobStatistics:
    def get_overall_stats() -> Dict:
        db = local_session()
        try:
            total_jobs = db.query(func.count(Job.id)).scalar()
            completed_jobs = db.query(func.count(Job.id)).filter(
                Job.status == JobStatus.COMPLETED
            ).scalar()
            failed_jobs = db.query(func.count(Job.id)).filter(
                Job.status == JobStatus.FAILED
            ).scalar()
            pending_jobs = db.query(func.count(Job.id)).filter(
                Job.status == JobStatus.PENDING
            ).scalar()
            running_jobs = db.query(func.count(Job.id)).filter(
                Job.status == JobStatus.RUNNING
            ).scalar()
            
            success_rate = (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0

            return {
                "total_jobs": total_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "pending": pending_jobs,
                "running": running_jobs,
                "success_rate_percent": round(success_rate, 2),
            }
        finally:
            db.close()

    def get_stats_by_job_type() -> List[Dict]:
        db = local_session()
        try:
            results = db.query(
                Job.job_type,
                func.count(Job.id).label('total'),
                func.sum(
                    case((Job.status == JobStatus.COMPLETED, 1), else_=0)
                ).label('completed'),
                func.sum(
                    case((Job.status == JobStatus.FAILED, 1), else_=0)
                ).label('failed'),
            ).group_by(Job.job_type).all()

            stats = []
            for row in results:
                completed = row.completed or 0
                failed = row.failed or 0
                total = row.total or 0
                success_rate = (completed / total * 100) if total > 0 else 0
                stats.append({
                    "job_type": row.job_type,
                    "total": total,
                    "completed": completed,
                    "failed": failed,
                    "success_rate_percent": round(success_rate, 2),
                })
            return stats
        finally:
            db.close()

    def get_execution_time_stats() -> Dict:
        db = local_session()

        try:
            completed_jobs = db.query(Job).filter(
                Job.status == JobStatus.COMPLETED,
                Job.started_at.isnot(None),
                Job.completed_at.isnot(None),
            ).all()

            if not completed_jobs:
                return {"message": "No completed jobs"}
            
            execution_times = [
                (job.completed_at - job.started_at).total_seconds() for job in completed_jobs
            ]

            return {
                "avg_execution_time_sec": round(sum(execution_times) / len(execution_times), 2),
                "min_execution_time_sec": round(min(execution_times), 2),
                "max_execution_time_sec": round(max(execution_times), 2)
            }
        
        except Exception as e:
            logger.error(f"Error getting execution time stats: {e}")
            return {"error": e}
        
        finally:
            db.close()

    def get_recent_jobs(limit: int = 10) -> List[Dict]:
        db= local_session()
        try:
            jobs = db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()

            return [
                {
                    "id": job.id,
                    "job_type": job.job_type,
                    "status": job.status,
                    "priority": job.priority,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                }
                for job in jobs
            ]
        
        finally:
            db.close()

    def get_timeseries_stats(hours: int = 24) -> List[Dict]:
        db = local_session()
        try:
            cutoff = datetime.now() - timedelta(hours = hours)

            jobs = db.query(Job).filter(Job.created_at >= cutoff).all()

            hourly_stats = {}
            for job in jobs:
                hour = job.created_at.replace(minute = 0, second = 0, microsecond = 0)
                hour_key = hour.isoformat()

                if hour_key not in hourly_stats:
                    hourly_stats[hour_key] = {
                        "timestamp": hour_key,
                        "total": 0,
                        "completed": 0,
                        "failed": 0,
                    }
                
                hourly_stats[hour_key]["total"] += 1
                if job.status == JobStatus.COMPLETED:
                    hourly_stats[hour_key]["completed"] += 1
                elif job.status == JobStatus.FAILED:
                    hourly_stats[hour_key]["failed"] += 1

            return sorted(hourly_stats.values(), key = lambda x: x["timestamp"])

        finally:
            db.close()


