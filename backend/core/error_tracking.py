import structlog
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

logger = structlog.get_logger()

class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorTracker:
    def __init__(self):
        self.errors = []
        self.error_counts = defaultdict(int)
        self.alert_thresholds = {
            ErrorSeverity.LOW: 100,
            ErrorSeverity.MEDIUM: 50,
            ErrorSeverity.HIGH: 10,
            ErrorSeverity.CRITICAL: 1
        }

    def record_error(
            self,
            error_type: str,
            error_message: str,
            severity: ErrorSeverity = ErrorSeverity.MEDIUM,
            context: Dict = None,
            user_id: str = None,
    ):
        error_record = {
            "timestamp": datetime.now(),
            "type": error_type,
            "message": error_message,
            "severity": severity,
            "context": context or {},
            "user_id": user_id,
        }

        self.errors.append(error_record)
        self.error_counts[error_type] += 1

        logger.error(
            "error tracked",
            error_type = error_type,
            severity = severity.value,
            user_id = user_id,
        )

        if self._should_alert(error_type, severity):
            self._send_alert(error_record)

    def _should_alert(self, error_type: str, severity: ErrorSeverity) -> bool:
        threshold = self.alert_thresholds.get(severity, 100)
        count = self.error_counts[error_type]

        return count >= threshold
    
    def _send_alert(self, error_record: Dict):
        logger.critical(
            "alert error threshold exceeded",
            error_type = error_record["type"],
            severity = error_record["severity"].value,
            message = error_record["message"],
        )

    def get_error_summary(self, hours: int = 1) -> Dict:
        cutoff = datetime.now() - timedelta(hours = hours)

        recent_errors = [
            error for error in self.errors if error["timestamp"] >= cutoff
        ]

        by_type = defaultdict(int)
        by_severity = defaultdict(int)

        for error in recent_errors:
            by_type[error["type"]] += 1
            by_severity[error["severity"]] += 1

        return {
            "time_period_hours": hours,
            "total_errors": len(recent_errors),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "recent_errors": recent_errors[-10:],
        }
    
    def get_error_rate(self, minutes: int = 5) -> float:
        cutoff = datetime.now() - timedelta(minutes = minutes)

        recent_errors = [
            error for error in self.errors if error["timestamp"] >= cutoff
        ]

        return len(recent_errors) / minutes if minutes > 0 else 0
    
_error_tracker = None

def get_error_tracker() -> ErrorTracker:
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker()
    return _error_tracker