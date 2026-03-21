from celery import Celery
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "task_engine",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_time_limit=3600,
    task_soft_time_limit=3540,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    broker_connection_retry_on_startup=True,

    worker_disable_gossip = True,
    worker_disable_mingle=True,
    worker_heartbeat = None,
    worker_cancel_long_running_tasks_on_connection_loss = True,

    task_default_priority = 5,
    broker_transport_options = {
        'priority_steps': list(range(21)),
        'queue_order_strategy': 'priority',
    },
    result_expires = 300,
    task_ignore_result = True,
)

celery_app.autodiscover_tasks(['workers'])