from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import structlog
import sys
import time
import json
from pathlib import Path
from datetime import datetime

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from workers.tasks import execute_job
from workers.celery_app import celery_app
from models import get_db, Job, JobStatus
from api.models import JobCreate, JobResponse
from core.predictor import get_predictor
from core.accuracy_tracker import calculate_prediction_accuracy
from core.scheduler import get_scheduler
from core.worker_health import get_health_monitor
from core.security import get_validator, get_rate_limiter_dep
from core.auth import verify_api_key, get_current_user
from core.audit import log_audit_event
import uuid
from core.logging_config import get_correlation_id, set_correlation_id, RequestLogger
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import JSONResponse, Response
import secrets
from fastapi import Request, Depends
from fastapi.responses import HTMLResponse
from core.metrics import track_request_metrics, track_job_metrics, jobs_submitted_total, MetricsCollector
from core.health import HealthCheck, HealthStatus
from core.statistics import JobStatistics
from core.error_tracking import get_error_tracker, ErrorSeverity
from core.config import get_settings

from api.v1 import v1_router
from api.jobs import router as legacy_jobs_router

logger = structlog.get_logger()
settings = get_settings()

app = FastAPI(
    title="Machine-Learning Task Engine API",
    description="Distributed Task Engine with ML-based Resource Prediction",
    version=settings.API_VERSION,
    docs_url = "/docs",
    redoc_url = "/redoc",
    openapi_url = "/openapi.json",
    debug=settings.DEBUG
)

app.include_router(v1_router, prefix = "/api")

app.include_router(
    legacy_jobs_router,
    tags = ["Legacy (Depracated)"],
    deprecated = True
)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(
        "api request",
        method = request.method,
        path = request.url.path,
        client_ip = request.client.host if request.client else "unknown",
    )

    response = await call_next(request)

    logger.info(
        "api response",
        status_code = response.status_code,
        path = request.url.path,
    )

    return response

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    set_correlation_id(correlation_id)

    RequestLogger.log_request(
        method = request.method,
        path = request.url.path,
        client_ip = request.client.host if request.client else "unknown",
        headers = dict(request.headers),
    )

    start_time = time.time()

    try:
        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000

        RequestLogger.log_response(
            method = request.method,
            path = request.url.path,
            status_code = response.status_code,
            duration_ms = duration_ms,
        )

        response.headers["X-Correlation-ID"] = correlation_id

        return response
    
    except Exception as e:
        RequestLogger.log_error(
            method = request.method,
            path = request.url.path,
            error = e,
        )

        raise

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    duration = time.time() - start_time
    track_request_metrics(
        method = request.method,
        endpoint = request.url.path,
        status_code = response.status_code,
        duration = duration
    )

    return response

class SecurityHeadersMiddleWare(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)

        response.headers["X-FRAME-OPTIONS"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response

app.add_middleware(SecurityHeadersMiddleWare)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)



class VersionDeprecationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if not request.url.path.startswith("/api/v"):
            response.headers["X-API-Warn"] = "This endpoint is deprecated. Use /api/v1/* instead"
            response.headers["Sunset"] = "2026-12-31"

        response.headers["X-API-Version"] = "2.0.0"

        return response
app.add_middleware(VersionDeprecationMiddleware)

class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if request.url.path == "/docs":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "style-src 'self' https://cdn.jsdelivr.net; "
                "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline' 'unsafe-eval'; "
                "img-src 'self' data: https://fastapi.tiangolo.com;"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "style-src 'self'; "
                "script-src 'self'; "
                "img-src 'self'; "
            )

        return response
app.add_middleware(CSPMiddleware)  

@app.get("/")
def root():
    return {
        "name": "ML Task Engine API",
        "version": "2.0.0",
        "api_versions": {
            "v1": "/api/v1",
            "legacy": "/" "deprecated"
        },
        "documentation": "/docs"
    }

@app.post("/jobs", response_model=JobResponse, status_code=201)
def create_job(
    job_data: JobCreate,
    request: Request,
    auth: dict = Depends(verify_api_key),
    db: Session = Depends(get_db),
    rate_limiter = Depends(get_rate_limiter_dep)
    ):
    job_data.user_id = auth["user_id"]
    request.state.user_id = auth["user_id"]
    request.state.rate_limter = rate_limiter
    logger.info("job.create requested", job_type=job_data.job_type, user_id=job_data.user_id)

    client_ip = request.client.host if request.client else "unknown"

    allowed, info = rate_limiter.is_allowed(user_id = job_data.user_id, ip_address = client_ip)
    if not allowed:
        raise HTTPException(
            status_code = 429,
            detail = {
                "error": "Rate Limit Exceeded",
                "limit": info["limit"],
                "window_seconds": info["window"],
                "retry_after": info["retry_after"],
            }
        )
    
    validator = get_validator()
    is_valid, error_msg = validator.validate_job(job_data.job_type, job_data.config)
    if not is_valid:
        logger.warning(
            "Validator.validate_job - invalid job rejected",
            user_id = job_data.user_id,
            job_type = job_data.job_type,
            reason = error_msg
        )

        raise HTTPException(status_code = 400, detail = error_msg)

    log_audit_event(
        event_type = "job_created",
        user_id = auth["user_id"],
        details = {
            "job_type": job_data.job_type,
            "priority": job_data.priority
        },
        severity = "info"
    )

    jobs_submitted_total.labels(
        job_type = job_data.job_type,
        priority = job_data.priority
    ).inc()

    predictor = get_predictor()
    predicted_memory, predicted_cpu = predictor.predict(
        job_data.config,
        job_data.job_type
    )

    max_memory = job_data.max_memory_mb
    if max_memory is None:
        max_memory = predicted_memory * 2.0

    logger.info(
        "job resources predicted",
        memory_mb = predicted_memory,
        cpu_percent = predicted_cpu,
    )

    job = Job(
        job_type=job_data.job_type,
        config=job_data.config,
        user_id=job_data.user_id,
        status=JobStatus.PENDING,
        predicted_cpu_percent=predicted_cpu,
        predicted_memory_db=predicted_memory,
        priority=job_data.priority,
        max_memory_mb=max_memory,
        max_execution_time_sec=job_data.max_execution_time_sec,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    logger.info("job.created", job_id=job.id, predicted_memory_db=job.predicted_memory_db, predicted_cpu_percent=job.predicted_cpu_percent)
    execute_job.delay(job.id)

    logger.info("job queued", job_id = job.id)

    usage = rate_limiter.get_usage(job_data.user_id)
    response_data = JobResponse.model_validate(job).model_dump(mode="json")
    response = JSONResponse(content=response_data, status_code=201)
    if "error" not in usage:
        response.headers["X-RateLimit-Limit"] = str(usage["requests_limit"])
        response.headers["X-RateLimit-Remaining"] = str(usage["requests_remaining"])
        response.headers["X-RateLimit-Reset"] = str(usage["window_seconds"])

    return response 

@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: int,
    auth: dict = Depends(verify_api_key),
    db: Session = Depends(get_db)
    ):
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.user_id != auth["user_id"]:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    return job


@app.get("/jobs", response_model=List[JobResponse])
def list_job(
    status: JobStatus = None,
    jobs_limit: int = 100,
    db: Session = Depends(get_db)
):
    
    query = db.query(Job)

    if status:
        query = query.filter(Job.status == status)

    jobs = query.order_by(Job.created_at.desc()).limit(jobs_limit).all()

    return jobs

@app.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: int,
    cancelled_by: str = "api_user",
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    job = db.get(Job, job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.user_id != current_user.user_id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    scheduler = get_scheduler()
    success = scheduler.cancel_job(job_id, cancelled_by=cancelled_by)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel job")
    
    return {"status": "cancelled", "job_id": job_id}

@app.get("/workers/health")
def get_workers_health():
    inspect = celery_app.control.inspect()

    stats = inspect.stats()
    active = inspect.active()
    registered = inspect.registered()

    if not stats:
        return {
            "total_workers": 0,
            "workers": [],
            "message": "No workers found. Ensure workers are running"
        }
    
    workers = []
    for worker_name, worker_stats in stats.items():
        workers.append({
            "worker_id": worker_name,
            "status": "active",
            "pool": worker_stats.get("pool", {}).get("implementation", "unkwown"),
            "max_concurrency": worker_stats.get("pool", {}).get("max-concurrency", 0),
            "active_jobs": len(active.get(worker_name, [])) if active else 0,
            "total_tasks": worker_stats.get("total", {}),
        })
    return {
        "total_workers": len(workers),
        "workers": workers,
    }

@app.get("/workers/active-jobs")
def get_active_jobs():
    inspect = celery_app.control.inspect()
    active = inspect.active()

    if not active:
        return {
            "active_jobs": 0,
            "jobs": []
        }
    all_jobs = []
    for worker_name, jobs in active.items():
        for job in jobs:
            all_jobs.append({
                "worker": worker_name,
                "task_id": job.get("id"),
                "task_name": job.get("name"),
                "args": job.get("args"),
                "time_start": job.get("time_start"),
            })
            
    return {
        "active jobs": len(all_jobs),
        "jobs": all_jobs,
    }
@app.get("/workers/{worker_id}/health")
def get_worker_health(worker_id: str):
    monitor = get_health_monitor()
    stats = monitor.get_worker_stats(worker_id)

    if "error" in stats:
        raise HTTPException(status_code=404, detail=stats["error"])

    return stats

@app.get("/system/stats")
def get_system_stats(db: Session = Depends(get_db)):
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

    monitor = get_health_monitor()
    workers = monitor.get_all_workers()
    active_workers = sum(1 for w in workers if w["status"] == "active")

    return {
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "pending": pending_jobs,
            "running": running_jobs,
            "success_rate": (completed_jobs / total_jobs * 100 if total_jobs > 0 else 0),
        },
        "workers": {
            "total": len(workers),
            "active": active_workers,
            "stale": len(workers) - active_workers
        },
    }

@app.post("/predictor/train")
def train_predictor():
    predictor = get_predictor()
    success = predictor.train(min_samples=5)

    if success:
        return {
            "status": "success",
            "message": "predictor trained",
            "training samples": predictor.training_samples,
        }
    return {
        "status": "failed",
        "message": "Need at least 5 completed jobs for training data"
    }

@app.get("/predictor/evaluate")
def evaluate_predictor():
    predictor = get_predictor()
    return predictor.evaluate()

@app.get("/predictor/accuracy")
def get_prediction_accuracy():
    return calculate_prediction_accuracy()


def get_rate_limiter_for_middleware(rate_limiter=Depends(get_rate_limiter_dep)):
    return rate_limiter

@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    response = await call_next(request)

    if hasattr(request.state, "user_id") and hasattr(request.state, "rate_limiter"):
        rate_limiter = get_rate_limiter_for_middleware()
        usage = rate_limiter.get_usage(request.state.user_id)

        if "error" not in usage:
            response.headers["X-RateLimit-Limit"] = str(usage["requests_limit"])
            response.headers["X-RateLimit-Remaining"] = str(usage["requests_remaining"])
            response.headers["X-RateLimit-Reset"] = str(usage["window_seconds"])

    return response

@app.get("/admin/rate-limit/{user_id}")
def get_rate_limit_usage(user_id: str):
    rate_limiter = get_rate_limiter_dep()
    usage = rate_limiter.get_usage(user_id)
    return usage

@app.post("/admin/rate-limit/{user_id}/reset")
def reset_rate_limit(user_id: str):
    rate_limiter = get_rate_limiter_dep()
    success = rate_limiter.reset(user_id)

    if success:
        return {
            "status": "success",
            "user_id": user_id,
            "message": "Rate Limit reset"
        }
    raise HTTPException(status_code = 500, detail = "Failed to reset rate limit")

@app.get("/metrics")
def metrics():
    return Response(
        content = generate_latest(),
        media_type = CONTENT_TYPE_LATEST
    )

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestampe": datetime.now().isoformat() + "Z"
    }

@app.get("/health/live")
def live_probe():
    if HealthCheck.is_alive():
        return {"status": "alive"}
    else:
        raise HTTPException(status_code=503, detail="Service not alive")
    
@app.get("/health/ready")
def readiness_probe():
    if HealthCheck.is_ready():
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")
    
@app.get("/health/detailed")
def detailed_health():
    health = HealthCheck.get_comprehensive_health()

    if health["status"] == HealthStatus.UNHEALTHY:
        status_code = 503
    else:
        status_code = 200

    return Response(
        content = json.dumps(health, indent = 2),
        status_code = status_code,
        media_type = "application/json"
    )

@app.get("/stats/overall")
def get_overall_stats():
    return JobStatistics.get_overall_stats()

@app.get("/stats/by-type")
def get_stats_by_type():
    return JobStatistics.get_stats_by_job_type()

@app.get("/stats/execution-times")
def get_execution_time_stats():
    return JobStatistics.get_execution_time_stats()

@app.get("/stats/recent")
def get_recent_jobs(limit: int = 10):
    return JobStatistics.get_recent_jobs(limit = limit)

@app.get("stats/timeseries")
def get_timeseries_stats(hours: int = 24):
    return JobStatistics.get_timeseries_stats(hours = hours)

@app.get("/errors/summary/")
def get_error_summary(hours: int = 1):
    tracker = get_error_tracker()
    return tracker.get_error_summary(hours = hours)

@app.get("/errors/rate/")
def get_error_rate(minutes: int = 5):
    tracker = get_error_tracker()
    rate = tracker.get_error_rate(minutes = minutes)
    return {
        "errors_per_minute": round(rate, 2),
        "time_window_minutes": minutes,
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tracker = get_error_tracker()

    tracker.record_error(
        error_type = "Exception",
        error_message = str(exc),
        severity = ErrorSeverity.HIGH,
        context = {
            "path": request.url.path,
            "method": request.method
        }
    )

    logger.error(
        "unhandled exception",
        error_type = type(exc).__name__,
        error = str(exc),
        path = request.url.path,
    )

    return JSONResponse(
        status_code = 500,
        content = {"detail", "Internal Server Error"}
    )
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    tracker = get_error_tracker()
    
    if exc.status_code >= 400:
        tracker.record_error(
            error_type="HTTPException",
            error_message=exc.detail,
            severity=ErrorSeverity.HIGH if exc.status_code >= 500 else ErrorSeverity.MEDIUM,
            context={
                "path": request.url.path,
                "method": request.method,
                "status_code": exc.status_code
            }
        )
        
        logger.error(
            "http_exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            severity="high" if exc.status_code >= 500 else "warning"
        )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

def custom_openapi():
    """Generate enhanced OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="ML Task Engine API",
        version="2.0.0",
        description="""
# ML Task Engine

A distributed task execution engine with ML-based resource prediction.

## Features

- **Smart Resource Prediction**: ML models predict CPU and memory usage before execution
- **Priority Queue**: Jobs execute based on priority (0-20 scale)
- **Rate Limiting**: Per-user and per-IP rate limits prevent abuse
- **Real-time Monitoring**: Track job execution with detailed metrics
- **Bulk Operations**: Submit or cancel multiple jobs efficiently

## Authentication

All endpoints require an API key passed in the `Authorization` header:

```
Authorization: Bearer YOUR_API_KEY
```

Get your API key from your account settings or contact an administrator.

## Rate Limits

- **Standard users**: 100 requests/minute
- **IP-based limit**: 300 requests/minute
- **Global limit**: 1000 requests/minute

Exceeded limits return HTTP 429 with `Retry-After` header.

## Pagination

List endpoints support pagination via query parameters:

- `page`: Page number (default: 1)
- `page_size`: Items per page (default: 50, max: 100)

## Response Format

All paginated responses follow this structure:

```json
{
  "items": [...],
  "total": 150,
  "page": 1,
  "page_size": 50,
  "total_pages": 3,
  "has_next": true,
  "has_prev": false
}
```
""",
        routes=app.routes,
        tags=[
            {
                "name": "Jobs",
                "description": "Create, retrieve, and manage ML training jobs"
            },
            {
                "name": "Bulk Operations",
                "description": "Submit or cancel multiple jobs in a single request"
            },
            {
                "name": "System",
                "description": "System health and statistics"
            },
            {
                "name": "Admin",
                "description": "Administrative operations (requires admin role)"
            }
        ]
)
    
    openapi_schema["components"]["SecuritySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
            "description": "Enter your API key"
        }
    }

    openapi_schema["security"] = [{"BearerAuth": []}]

    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development"},
        {"url": "https://api.mltaskengine.com", "description": "Production"}
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_favicon_url="/static/favicon.ico",
    )

    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
        "style-src 'self' https://cdn.jsdelivr.net;"
    )

    return response

app.openapi = custom_openapi

if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




