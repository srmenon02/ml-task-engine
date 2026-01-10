import psutil
import structlog
from typing import Dict, List
from datetime import datetime
from sqlalchemy import text
from models import local_session
import redis as redis_lib
from workers.celery_app import celery_app

logger = structlog.get_logger()

class HealthStatus:
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheck:
    def check_database() -> Dict:
        try:
            db = local_session()
            start = datetime.now()

            result = db.execute(text("SELECT 1")).fetchone()

            latency_ms = (datetime.now() - start).total_seconds() * 1000

            db.close()

            return {
                "status": HealthStatus.HEALTHY,
                "latency_ms": round(latency_ms, 2),
                "message": "db connection succesful"
            }
        
        except Exception as e:
            logger.error(f"health db check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": e,
                "message": "db connection failed"
            }

    def check_redis(redis_url: str = "redis://localhost:6379/0") -> Dict:
        try:
            client = redis_lib.from_url(redis_url, socket_timeout = 5)
            start = datetime.now()

            client.ping()

            latency_ms = (datetime.now() - start).total_seconds() * 1000

            info = client.info()

            client.close()

            return {
                "status": HealthStatus.HEALTHY,
                "latency_ms": round(latency_ms, 2),
                "version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                "info": info,
                "message": "redis connection successful"
            }
        
        except Exception as e:
            logger.error(f"health redis check failed: {e}")
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": e,
                "message": "Redis connection failed"
            }
        
    def check_workers() -> Dict:
        try:
            inspect = celery_app.control.inspect()
            stats = inspect.stats()

            if not stats:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "worker_count": 0,
                    "message": "No workers available"
                }
            
            worker_count = len(stats)

            return {
                "status": HealthStatus.HEALTHY if worker_count > 0 else HealthStatus.UNHEALTHY,
                "worker_count": worker_count,
                "workers": list(stats.keys()),
                "message": f"{worker_count} worker(s) active"
            }
        
        except Exception as e:
            logger.error(f"health workers check failed: {e}")
            return {
                "status": HealthStatus.DEGRADED,
                "error": e,
                "message": "Unable to check worker status"
            }
        
    def check_disk_space() -> Dict:
        try:
            disk = psutil.disk_usage('/')

            percent_used = disk.percent

            status = HealthStatus.HEALTHY

            if percent_used > 90:
                status = HealthStatus.UNHEALTHY
            elif percent_used > 80:
                status = HealthStatus.DEGRADED

            return {
                "status": status,
                "total_gb": round(disk.total / (1024 **3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "percent_used": percent_used,
                "message": f"{percent_used} disk used%"
            }
        
        except Exception as e:
            logger.error(f"Health disk check failed {e}")
            return {
                "status": "HealthStatus.DEGRADED",
                "error": e
            }

    def check_memory() -> Dict:
        try:
            memory = psutil.virtual_memory()

            percent_used = memory.percent

            status = HealthStatus.HEALTHY
            if percent_used > 90:
                status = HealthStatus.UNHEALTHY
            elif percent_used > 80:
                status = HealthStatus.DEGRADED

            return {
                "status": status,
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "used_gb": round(memory.used / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "percent_used": percent_used,
                "message": f"{percent_used} memory used%"
            }
        
        except Exception as e:
            logger.error(f"Health memory check failed: {e}")
            return {
                "status": HealthStatus.DEGRADED,
                "error": e
            }
    
    def get_comprehensive_health() -> Dict:
        checks = {
            "database": HealthCheck.check_database(),
            "redis": HealthCheck.check_redis(),
            "workers": HealthCheck.check_workers(),
            "disk": HealthCheck.check_disk_space(),
            "memory": HealthCheck.check_memory()
        }

        statuses = [check['status'] for check in checks.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat() + "Z",
            "checks": checks
        }

    def is_ready() -> bool:
        db_health = HealthCheck.check_database()
        redis_health = HealthCheck.check_redis()

        return (
            db_health["status"] == HealthStatus.HEALTHY and
            redis_health["status"] == HealthStatus.HEALTHY
        )

    def is_alive() -> bool:
        return True