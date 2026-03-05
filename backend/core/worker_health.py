from typing import Dict, List
from datetime import datetime, timedelta, timezone
import structlog

logger = structlog.get_logger()

class WorkerHealthMonitor:
    def __init__(self):
        self.workers: Dict[str, Dict] = {}

    def register_worker(self, worker_id: str):
        self.workers[worker_id] = {
            "worker_id": worker_id,
            "status": "active",
            "last_heartbeat": datetime.now(timezone.utc),
            "jobs_completed": 0,
            "jobs_failed": 0,
            "registered_at": datetime.now(timezone.utc),
        }

        logger.info("worker registered", worker_id=worker_id)

    def heartbeat(self, worker_id: str):
        if worker_id not in self.workers:
            self.register_worker(worker_id)

        self.workers[worker_id]["last_heartbeat"] = datetime.now(timezone.utc)
        self.workers[worker_id]["status"] = "active"

    def record_job_completion(self, worker_id: str, success: bool):
        if worker_id not in self.workers:
            self.register_worker(worker_id)
        if success:
            self.workers[worker_id]["jobs_completed"] += 1
        else:
            self.workers[worker_id]["jobs_failed"] += 1
    
    def get_worker_stats(self, worker_id: str) -> Dict:
        if worker_id not in self.workers:
            return {"error": "Worker not found"}
        
        worker = self.workers[worker_id]
        total_jobs = worker["jobs_completed"] + worker["jobs_failed"]

        return {
            "worker_id": worker_id,
            "status": worker["status"],
            "last_heartbeat": worker["last_heartbeat"],
            "jobs_completed": worker["jobs_completed"],
            "jobs_failed": worker["jobs_failed"],
            "success_rate": (worker["jobs_completed"] / total_jobs * 100 if total_jobs > 0 else 0),
            "uptime_seconds": (datetime.now(timezone.utc) - worker["registered_at"]).total_seconds(),
        }
    
    def get_all_workers(self) -> List[Dict]:
        return [self.get_worker_stats(wid) for wid in self.workers.keys()]

def check_stale_workers(self, timeout_secs: int = 300) -> List[str]:
    threshold = datetime.now(timezone.utc) - timedelta(seconds=timeout_secs)
    stale_workers = []

    for worker_id, worker in self.workers.items():
        if worker["last_heartbeat"] < threshold:
            worker["status"] = "stale"
            stale_workers.append(worker_id)
            logger.warning(
                "worker stale",
                worker_id = worker_id,
                last_heartbeat = worker["last_heartbeat"],
            )
    return stale_workers

_health_monitor = None

def get_health_monitor() -> WorkerHealthMonitor:
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = WorkerHealthMonitor()
    return _health_monitor
