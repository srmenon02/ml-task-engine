import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import redis
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from models.database import base, get_db
from api.main import app
from workers.celery_app import celery_app
from freezegun import freeze_time
import core.predictor
import core.scheduler
import core.worker_health
from models import Job, Execution, ResourceProfile, JobStatus, JobPriority
from core.audit import AuditLog
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, PropertyMock
from workers.tasks import DBTask

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

@pytest.fixture(scope = "function")
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    base.metadata.create_all(bind = engine)

    TestingSessionLocal = sessionmaker(
        autocommit = False,
        autoflush = False,
        bind = engine
    )

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        base.metadata.drop_all(bind = engine)

@pytest.fixture(scope = "function")
def override_get_db(test_db):
    def _override():
        return test_db
    return _override

@pytest.fixture(scope = "function")
def client(override_get_db):
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_api_key_user123"}

@pytest.fixture
def admin_headers():
    return {"Authorization": "Bearer test_api_key_admin"}

@pytest.fixture(scope="function")
def mock_redis():
    mock = MagicMock(spec=redis.Redis)
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    
    pipeline_mock = MagicMock()
    pipeline_mock.zremrangebyscore.return_value = pipeline_mock
    pipeline_mock.zcard.return_value = pipeline_mock
    pipeline_mock.zadd.return_value = pipeline_mock
    pipeline_mock.expire.return_value = pipeline_mock
    pipeline_mock.execute.return_value = [0, 0, True, True]
    pipeline_mock.zrem.return_value = True
    
    pipeline_mock.__enter__.return_value = pipeline_mock
    pipeline_mock.__exit__.return_value = None
    
    mock.pipeline.return_value = pipeline_mock
    
    mock._pipeline_mock = pipeline_mock
    
    return mock

@pytest.fixture(scope="function")
def patch_db_task(test_db):
    with patch.object(DBTask, 'db', new_callable=PropertyMock) as mock_db:
        mock_db.return_value = test_db
        yield


@pytest.fixture(scope = "function")
def celery_worker():
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
        worker_log_format='[%(levelname)s] %(message)s',
        worker_loglevel='DEBUG'
    )
    yield celery_app
    celery_app.conf.update(
        task_always_eager=False,
        task_eager_propagates=False
    )

@pytest.fixture
def freeze_time():
    return freeze_time

@pytest.fixture(autouse = True)
def reset_singletons():
    core.predictor._predictor = None
    core.scheduler._scheduler = None
    core.worker_health._health_monitor = None

    yield


