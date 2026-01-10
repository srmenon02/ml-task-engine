from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
from prometheus_client import CollectorRegistry, multiprocess, generate_latest
import time
import structlog
from typing import Callable
from functools import wraps

logger = structlog.get_logger()

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP Requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP Request latency',
    ['method', 'endpoint'],
    buckets = (0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0)
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests currently being processed',
    ['method', 'endpoint']
)

jobs_submitted_total = Counter(
    'jobs_submitted_total',
    'Total jobs submitted',
    ['job_type', 'priority']
)

jobs_completed_total = Counter(
    'jobs_completed_total',
    'Total jobs completed',
    ['job_type', 'status']
)

job_duration_seconds = Histogram(
    'job_duration_seconds',
    'Job execution duration',
    ['job_type'],
    buckets = (1, 5, 10, 30, 60, 120, 300, 600, 1800)
)

jobs_in_queue = Gauge(
    'jobs_in_queue',
    'Jobs waiting in queue',
    ['priority']
)

jobs_running = Gauge(
    'jobs_running',
    'Jobs currently executing'
)

job_memory_usage_mb = Histogram(
    'job_memory_usage_mb',
    'Job Memeory Usage in mb',
    ['job_type'],
    buckets = (50, 100, 250, 500, 1000, 2500, 5000, 10000)
)

job_cpu_usage_percent = Histogram(
    'job_cpu_usage_percent',
    'Job CPU usage percentage',
    ['job_type'],
    buckets = (10, 25, 50, 75, 90, 100)
)

prediction_accuracy_percent = Gauge(
    'prediction_accuracy_percent',
    'ML predictor accuracy',
    ['resource_type']
)

workers_active = Gauge(
    'workers_active',
    'NUmber of active workers'
)

database_connections = Gauge(
    'database_connections',
    'Active database connections'
)

redis_connected = Gauge(
    'redis_connected',
    'Redis connectio status (1 = connected, 0 = disconnected)'
)

rate_limit_exceeded_total = Counter(
    'rate_limit_exceeded_total',
    'Total rate liit violations',
    ['user_id']
)

security_violations_total = Counter(
    'security_violations_total',
    'Total security violations',
    ['violation_type']
)

app_info = Info(
    'app',
    'Application information'
)

app_info.info({
    'version': '0.2.0',
    'name': 'ml-task-engine'
})

def track_request_metrics(
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
):
    http_requests_total.labels(
        method = method,
        endpoint = endpoint,
        status = f"{status_code // 100}xx"
    ).inc()

    http_request_duration_seconds.labels(
        method = method,
        endpoint = endpoint
    ).observe(duration)

def track_job_metrics(
        job_type: str,
        status: str,
        duration: float = None,
        memory_mb: float = None,
        cpu_percent: float = None
):
    jobs_completed_total.labels(
        job_type = job_type,
        status = status
    ).inc()

    if duration:
        job_duration_seconds.labels(job_type = job_type).observe(duration)

    if memory_mb:
        job_memory_usage_mb.labels(job_type = job_type).observe(memory_mb)

    if cpu_percent:
        job_cpu_usage_percent.labels(job_type = job_type).observe(cpu_percent)

def time_function(metric: Histogram):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metric.observe(duration)

        return wrapper
    return decorator


class MetricsCollector:
    def update_queue_depths(pending_by_priority: dict):
        for priority, count in pending_by_priority.items():
            jobs_in_queue.labels(priority = priority).set(count)

    def update_running_jobs(count: int):
        jobs_running.set(count)

    def update_worker_count(count: int):
        workers_active.set(count)

    def update_prediction_accuracy(memory_mape: float, cpu_mape: float):
        memory_accuracy = max(0, 100 - memory_mape)
        cpu_accuracy = max(0, 100 - cpu_mape)

        prediction_accuracy_percent.labels(resource_type = "memory").set(memory_accuracy)
        prediction_accuracy_percent.labels(resource_type = "cpu").set(cpu_accuracy)

    def record_security_violation(violation_type: str):
        security_violations_total.labels(violation_type = violation_type).inc()

    def record_rate_limit_exceeded(user_id: str):
        rate_limit_exceeded_total.labels(user_id = user_id).inc()