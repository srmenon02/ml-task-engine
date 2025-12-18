from celery import Task
from datetime import datetime
import structlog
import psutil
import time
import traceback
from typing import Dict, Any

from workers.celery_app import celery_app
from models import local_session, Job, JobStatus, Execution, ResourceProfile

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

import threading


logger = structlog.get_logger()

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

@celery_app.task(base=DBTask, bind=True, name="workers.task.execute_job")
def execute_job(self, job_id: int) -> Dict[str, Any]:
    logger.info("task.execute_job started", job_id=job_id, worker_id=self.request.id)

    db = self.db

    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        logger.error("task.execute_job not found", job_id=job_id)
        return {"error": f"Job {job_id} not found"}
    
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now()
    db.commit()

    logger.info("task.running", job_id=job_id, job_type=job.job_type)

    execution = Execution(
        job_id = job_id,
        worker_id = self.request.id,
        execution_num = job.retry_count + 1,
        started_at = datetime.now()
    )

    db.add(execution)
    db.commit()

    process = psutil.Process()
    cpu_samples = []
    memory_samples = []

    start_time = time.time()

    try:
        result = _execute_job_by_type(job, process, cpu_samples, memory_samples)

        execution_time = time.time() - start_time

        cpu_avg = np.mean(cpu_samples) if cpu_samples else 0.0
        cpu_max = np.max(cpu_samples) if cpu_samples else 0.0
        memory_avg = np.mean(memory_samples) if memory_samples else 0.0
        memory_max = np.max(memory_samples) if memory_samples else 0.0

        execution.actual_cpu_percent_avg = cpu_avg
        execution.actual_cpu_percent_max = cpu_max
        execution.actual_memory_mb_avg = memory_avg / (1024 * 1024)
        execution.actual_memory_mb_max = memory_max / (1024 * 1024)
        execution.completed_at = datetime.now()
        execution.success = 1

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        job.results = result

        db.commit()

        _store_resource_profile(db, job, execution, execution_time)

        logger.info(
            "Task Completed",
            job_id=job.id,
            execution_time=execution_time,
            cpu_avg=cpu_avg,
            memory_mb_avg = memory_avg / (1024 * 1024),
        )

        return {
            "status": "success",
            "job_id": job_id,
            "result": result,
            "execution_time": execution_time,
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

        execution.completed_at = datetime.now()
        execution.success = 0
        execution.error_msg = error_msg

        job.status = JobStatus.FAILED
        job.error_msg = error_msg
        job.completed_at = datetime.now()

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
        
        return {
            "status": "failed",
            "job_id": job_id,
            "error_msg": error_msg,
        }

def _execute_job_by_type(
    job: Job,
    process: psutil.Process,
    cpu_samples: list,
    memory_samples: list,
) -> Dict[str, Any]:
    if job.job_type == "train_sklearn_model":
        return _train_sklearn_model(job, process, cpu_samples, memory_samples)
    else:
        raise ValueError(f"Unknown job type: {job.job_type}")
    
def _train_sklearn_model(
        job: Job,
        process: psutil.Process,
        cpu_samples: list,
        memory_samples: list,
) -> Dict[str, Any]:
    logger.info("sklearn.trainning started", job_id=job.id, config=job.config)

    config = job.config
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

    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    stop_monitoring = threading.Event()

    def monitor_resources():
        while not stop_monitoring.is_set():
            try:
                cpu_samples.append(process.cpu_percent(interval=0.1))
                mem_info = process.memory_info()
                memory_samples.append(mem_info.rss)
                time.sleep(0.5)
            except:
                break

    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.start()

    model.fit(X_train, y_train)

    stop_monitoring.set()
    monitor_thread.join()

    training_time = time.time() - start

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(
        "sklearn training done",
        job_id=job.id,
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