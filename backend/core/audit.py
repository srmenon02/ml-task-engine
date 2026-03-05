import structlog
from datetime import datetime, timezone
from models import local_session
from sqlalchemy import Column, Integer, String, DateTime, JSON
from models.database import base

logger = structlog.get_logger()

class AuditLog(base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    event_type = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    ip_address = Column(String, nullable=True)
    details = Column(JSON, nullable=True)
    severity = Column(String, nullable=False)

def log_audit_event(
        event_type: str,
        user_id: str,
        details: dict = None,
        severity: str = "info",
        ip_address: str = None,
):
    try:
        db = local_session()
        try:
            db.add(
                AuditLog(
                    event_type=event_type,
                    user_id=user_id,
                    details=details or {},
                    severity=severity,
                    ip_address=ip_address,
                )
            )
            db.commit()

            logger.info(
                "AuditLog.event logged",
                event_type=event_type,
                user_id=user_id,
                severity=severity,
            )
        except Exception as e:
            logger.error(f"AuditLog.event log failed {e}")
            db.rollback()
        finally:
            db.close()
    except Exception as e:
        logger.error(f"AuditLog connection failed {e}")

