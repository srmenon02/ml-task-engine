import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import redis
from pathlib import Path
from unittest.mock import Mock

from models.database import base
from api.main import app
from workers.celery_app import celery_app
from freezegun import freeze_time
import core.predictor
import core.scheduler
import core.worker_health


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

@pytest.fixture(scope = "function")
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    base.metabase.create_all(bind = engine)

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
        base.metabase.drop_all(bind = engine)

@pytest.fixture(scope = "function")
def override_get_db(test_db):
    def _override():
        try:
            yield test_db
        finally:
            pass
    return _override

@pytest.fixture(scope = "function")
def client(override_get_db):
    app.dependency_overrides[base.get_db] = override_get_db
    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test_api_key_user123"}

@pytest.fixture
def admin_headers():
    return {"Authorization": "Bearer test_api_key_admin"}

@pytest.fixture(scope = "function")
def mock_redis():
    mock = Mock(spec = redis.Redis)
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.pipeline.return_value.__enter__.return_value = mock
    mock.pipeline.return_value.__exit__.return_value = None
    mock.execute.return_value = [0,0, True, True]
    return mock

@pytest.fixture(scope = "function")
def celery_worker():
    celery_app.conf.update(
        task_always_eager = True,
        task_eager_propagates = True,
    )

    yield celery_app
    celery_app.conf.update(
        task_always_eager = False,
        task_eager_propagates = False,
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

