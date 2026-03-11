from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from datetime import datetime, timezone
import structlog
import psutil
import time
import traceback
import threading
import os
import socket
from typing import Dict, Any

from pathlib import Path
import sys
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
from core.training_scheduler import get_training_scheduler
from workers.celery_app import celery_app
from models import local_session, Job, JobStatus, Execution, ResourceProfile

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import threading
from celery import signals
from core.worker_health import get_health_monitor
from core.metrics import track_job_metrics

logger = structlog.get_logger()
logger.info("tasks.py reloaded, new version")

class DBTask(Task):
    _db = None
    @property
    def db(self):
        if self._db is None:
            self._db = local_session()
        return self._db
    
    def after_return(self, *args, **kwards):
        if self._db is not None:
            self._db.close()
            self._db = None

@celery_app.task(base=DBTask, bind=True, name="workers.task.execute_job", max_retries=2)
def execute_job(self, job_id: int) -> Dict[str, Any]:
    logger.info("task.execute_job started", job_id=job_id, worker_id=self.request.id)

    db = self.db

    job = db.query(Job).filter(Job.id == job_id).first()
    job_snapshot = {
        "id": job.id,
        "job_type": job.job_type,
        "config": job.config,
        "max_memory_mb": job.max_memory_mb,
        "max_execution_time_sec": job.max_execution_time_sec
    }

    if not job:
        logger.error("task.execute_job not found", job_id=job_id)
        return {"error": f"Job {job_id} not found"}
    
    if job.status == JobStatus.CANCELED:
        logger.info("task.job cancelled before start", job_id = job_id)
        return {"status": "cancelled", "job_id": job_id}
    
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    db.commit()

    logger.info("task.running", job_id=job_id, job_type=job.job_type)

    execution = Execution(
        job_id = job_id,
        worker_id = self.request.id,
        execution_num = job.retry_count + 1,
        started_at = datetime.now(timezone.utc)
    )

    db.add(execution)
    db.commit()

    process = psutil.Process()
    cpu_samples = []
    memory_samples = []

    start_time = time.time()

    try:
        result = _execute_job_with_limits(job_snapshot, process, cpu_samples, memory_samples, self.request.id, db,)

        execution_time = time.time() - start_time

        cpu_avg = np.mean(cpu_samples) if cpu_samples else 0.0
        cpu_max = np.max(cpu_samples) if cpu_samples else 0.0
        memory_avg = np.mean(memory_samples) if memory_samples else 0.0
        memory_max = np.max(memory_samples) if memory_samples else 0.0

        execution.actual_cpu_percent_avg = cpu_avg
        execution.actual_cpu_percent_max = cpu_max
        execution.actual_memory_mb_avg = memory_avg / (1024 * 1024)
        execution.actual_memory_mb_max = memory_max / (1024 * 1024)
        execution.completed_at = datetime.now(timezone.utc)
        execution.success = 1

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        job.results = result

        db.commit()

        _store_resource_profile(db, job, execution, execution_time)

        scheduler = get_training_scheduler()
        scheduler.check_and_retrain()

        logger.info(
            "Task Completed",
            job_id=job.id,
            execution_time=execution_time,
            cpu_avg=cpu_avg,
            memory_mb_avg = memory_avg / (1024 * 1024),
        )

        track_job_metrics(
            job_type = job.job_type,
            status = "completed",
            duration = execution_time,
            memory_mb = execution.actual_memory_mb_max,
            cpu_percent = execution.actual_cpu_percent_max,
        )


        return {
            "status": "success",
            "job_id": job_id,
            "result": result,
            "execution_time": execution_time,
        }
    
    except SoftTimeLimitExceeded:
        error_msg = f"Job exceeded time limit of {job.max_execution_time_sec} seconds"
        logger.error("task timeout", job_id=job_id, max_time = job.max_execution_time_sec)

        execution.completed_at = datetime.now(timezone.utc)
        execution.success = 0
        execution.error_msg = error_msg

        job.status = JobStatus.TIMEOUT
        job.error_message = error_msg
        job.completed_at = datetime.now(timezone.utc)

        db.commit()

        return {
            "status": "timeout",
            "job_id": job_id,
            "error": error_msg
        }
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        logger.error(
            "Task Failed",
            job_id=job.id,
            error_msg=error_msg,
            traceback=error_trace,
        )

        execution.completed_at = datetime.now(timezone.utc)
        execution.success = 0
        execution.error_msg = error_msg

        job.status = JobStatus.FAILED
        job.error_msg = error_msg
        job.completed_at = datetime.now(timezone.utc)

        db.commit()

        if job.retry_count < job.max_retries:
            job.status = JobStatus.RETRYING
            job.retry_count += 1
            db.commit()

            logger.info(
                "Task Retrying",
                job_id=job.id,
                retry_count=job.retry_count,
            )
            
            raise self.retry(exc=e, countdown=2 ** job.retry_count)
        
        track_job_metrics(
            job_type = job.job_type,
            status = "failed",
        )
                
        return {
            "status": "failed",
            "job_id": job_id,
            "error_msg": error_msg,
        }

def _execute_job_with_limits(
        job_snapshot: Dict,
        process: psutil.Process,
        cpu_samples: list,
        memory_samples: list,
        task_id: str,
        db,
) -> Dict[str, Any]:
    stop_monitoring = threading.Event()
    memory_exceeded = threading.Event()
    baseline_rss = process.memory_info().rss

    def monitor_resources():
        monitor_db = local_session()
        try:
            while not stop_monitoring.is_set():
                try:
                    monitor_job = monitor_db.query(Job).filter(job_snapshot["id"] == job_snapshot["id"]).first()
                    if monitor_job and monitor_job.status == JobStatus.CANCELED:
                        logger.info("task cancelled during execution", job_id = job_snapshot["id"])
                        stop_monitoring.set()
                        raise MemoryError("Job canclled during Execution")

                    cpu_samples.append(process.cpu_percent(interval=0.1))
                    mem_info = process.memory_info()
                    memory_samples.append(mem_info.rss)

                    current_rss = process.memory_info().rss
                    delta_mb = (current_rss - baseline_rss) / (1024 * 1024)
                    if job_snapshot["max_memory_mb"] and delta_mb > job_snapshot["max_memory_mb"]:
                        logger.error(
                            "task.memory_limit execeeded",
                            job_id = job_snapshot["id"],
                            current_mb = delta_mb,
                            limit_mb = job_snapshot["max_memory_mb"],
                        )
                        memory_exceeded.set()
                        stop_monitoring.set()
                        raise MemoryError("Job exceeded memory limit")

                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"monitor error: {e}")
                    break
        finally:
            monitor_db.close()

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    try:
        job_type = job_snapshot["job_type"]
        job_config = job_snapshot["config"]
        job_id = job_snapshot["id"]
        result = _execute_job_by_type(job_type, job_config, job_id, process, cpu_samples, memory_samples)
        stop_monitoring.set()
        monitor_thread.join(timeout=2)

        if memory_exceeded.is_set():
            raise MemoryError(
                f"Job exceeded memory limit of {job_snapshot['max_memory_mb']} MB"
            )
        return result
    except Exception as e:
        stop_monitoring.set()
        raise
    
def _execute_job_by_type(
    job_type,
    job_config,
    job_id,
    process: psutil.Process,
    cpu_samples: list,
    memory_samples: list,
) -> Dict[str, Any]:
    if job_type == "train_sklearn_model":
        return _train_sklearn_model(job_id, job_config, process, cpu_samples, memory_samples)
    else:
        raise ValueError(f"Unknown job type: {job_type}")
    
def _train_sklearn_model(
        job_id,
        config,
        process: psutil.Process,
        cpu_samples: list,
        memory_samples: list,
) -> Dict[str, Any]:
    logger.info("sklearn.trainning started", job_id=job_id, config=config)
    model_type = config.get("model", "RandomForest")
    n_estimators = config.get("n_estimators", 100)
    n_samples = config.get("dataset_rows", 10000)
    n_features = config.get("n_features", 20)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.6),
        n_redundant=int(n_features * 0.2),
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mem_info = process.memory_info()
    cpu_samples.append(process.cpu_percent(interval=0.1))
    memory_samples.append(mem_info.rss)

    start = time.time()

    MODEL_MAP = {
        "RandomForest": lambda cfg: RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 100), random_state=42
        ),
        "GradientBoosting": lambda cfg: GradientBoostingClassifier(
            n_estimators=cfg.get("n_estimators", 100), random_state=42
        ),
        "LogisticRegression": lambda cfg: LogisticRegression(
            max_iter=cfg.get("max_iter", 1000), random_state=42
        ),
        "SVC": lambda cfg: SVC(
            C=cfg.get("C", 1.0), kernel=cfg.get("kernel", "rbf"), random_state=42
        ),
        "DecisionTree": lambda cfg: DecisionTreeClassifier(
            max_depth=cfg.get("max_depth", None), random_state=42
        ),
        "KNeighbors": lambda cfg: KNeighborsClassifier(
            n_neighbors=cfg.get("n_neighbors", 5)
        ),
    }

    if model_type not in MODEL_MAP:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = MODEL_MAP[model_type](config)


    model.fit(X_train, y_train)

    training_time = time.time() - start

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(
        "sklearn training done",
        job_id=job_id,
        accuracy=accuracy,
        time=training_time
    )

    return {
        "model_type": model_type,
        "n_estimators": n_estimators,
        "data size": n_samples,
        "accuracy": float(accuracy),
        "training_time (in seconds)": training_time
    }

def _store_resource_profile(
        db,
        job: Job,
        execution: Execution,
        execution_time: float,
):
    
    profile = ResourceProfile(
        job_type=job.job_type,
        config=job.config,
        memory_mb=execution.actual_memory_mb_max,
        cpu_percent=execution.actual_cpu_percent_max,
        execution_time=execution_time,
    )

    db.add(profile)
    db.commit()

    logger.info(
        "resource_profile.stored",
        job_id=job.id,
        memory_mdb=profile.memory_mb,
        cpu_percent=profile.cpu_percent,
    )