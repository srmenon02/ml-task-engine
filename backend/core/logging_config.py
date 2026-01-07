import structlog
import logging
import uuid
from datetime import datetime
from typing import Any, Dict
from contextvars import ContextVar

correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default = None)

def get_correlation_id() -> str:
    correlation_id = correlation_id_var.get()
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id

def set_correlation_id(correlation_id: str):
    correlation_id_var.set(correlation_id)

def add_correlation_id(logger, method_naeme, event_dict):
    event_dict['correlation_id'] = get_correlation_id()
    return event_dict

def add_timestamp(logger, method_name, event_dict):
    event_dict['correlation_id'] = get_correlation_id()
    return event_dict

def add_log_level(logger, method_name, event_dict):
    event_dict['level'] = method_name
    return event_dict

def configure_logging(log_level: str = "INFO", json_logs: bool = False):
    logging.basicConfig(
        format = "%(message)s",
        level = getattr(logging, log_level.upper()),
    )

    processors = [
        structlog.stdlib.add_log_level,
        add_correlation_id,
        add_timestamp,
        add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors = processors,
        wrapper_class = structlog.stdlib.BoundLogger,
        context_class = dict,
        logger_factory = structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use = True,
    )

class RequestLogger:
    def log_request(
        method: str,
        path: str,
        client_ip: str,
        user_id: str = None,
        headers: Dict = None,
    ):
        logger = structlog.get_logger()
        logger.info(
            "request received",
            method = method,
            path = path,
            client_ip = client_ip,
            user_id = user_id,
            headers = headers.get("user-agent") if headers else None,
        )

    def log_response(
            method: str,
            path: str,
            status_code: int,
            duration_ms: float,
            user_id: str = None,
    ):
        logger = structlog.get_logger()

        if status_code >= 500:
            log_level = "error"
        elif status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        log_func = getattr(logger, log_level)
        log_func(
            "request completed",
            method = method,
            path = path,
            status_code = status_code,
            duration_ms = round(duration_ms, 2),
            user_id = user_id
        )

    def log_error(
        method: str,
        path: str,
        error: Exception,
        user_id: str = None,
    ):
        logger = structlog.get_logger()
        logger.error(
            "request error",
            method = method,
            path = path,
            error = error,
            user_id = user_id,
        )

configure_logging(log_level = "INFO", json_logs=False)