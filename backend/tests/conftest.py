import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import redis
import sys
from pathlib import Path
from unittest.mock import MagicMock

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import os 
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "True"
os.environ["CELERY_TASK_EAGER_PROPAGATES"] = "True"
os.environ["CELERY_BROKER_URL"] = "memory://"
os.environ["CELERY_RESULT_BACKEND"] = "cache+memory://"
os.environ["DB_URL_CI"] = "sqlite:///:memory:"

from models.database import base, get_db
from api.main import app
from api.main import app as fastapi_app
from workers.celery_app import celery_app
from freezegun import freeze_time
import core.predictor
import core.scheduler
import core.worker_health
import core.security
from core.security import RedisRateLimiter
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, PropertyMock
from workers.tasks import DBTask
from tests.factories.job_factory import JobFactory
import models.database as db_module
from sqlalchemy import text
import core.rate_limiter as rate_limiter
from core.rate_limiter import RedisRateLimiter


@pytest.fixture(scope="session", autouse=True)
def force_celery_test_config():
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )
    yield

@pytest.fixture(scope="session", autouse=True)
def assert_test_db():
    from models.database import engine
    url = str(engine.url)
    assert url.startswith("sqlite"), f"Engine is not SQLite: {url}"

@pytest.fixture(scope = "function")
def test_db(test_engine):
    connection = test_engine.connect()
    transaction = connection.begin()

    TestingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=connection,
    )

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        transaction.rollback()
        connection.close()

@pytest.fixture(scope="function")
def mock_local_session(test_db):
    def get_test_db():
        return test_db

    with patch('models.database.local_session', side_effect=get_test_db):
        yield
@pytest.fixture(scope="function")
def job_factory(test_db):
    JobFactory._meta.sqlalchemy_session = test_db
    yield JobFactory
    JobFactory._meta.sqlalchemy_session = None

@pytest.fixture(scope = "function")
def override_get_db(test_db):
    def _override():
        try:
            yield test_db
        finally:
            pass
    return _override

@pytest.fixture(autouse=True, scope="function")
def clean_db(test_engine):
    connection = test_engine.connect()
    trans = connection.begin()

    connection.execute(text("PRAGMA foreign_keys=OFF;"))

    for table in reversed(base.metadata.sorted_tables):
        connection.execute(table.delete())

    trans.commit()
    connection.close()

    yield

@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    db_module.engine = engine
    db_module.local_session.configure(bind=engine)

    base.metadata.create_all(bind=engine)

    yield engine

    base.metadata.drop_all(bind=engine)

@pytest.fixture(scope = "function")
def client(override_get_db, mock_local_session):
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


@pytest.fixture(scope="session", autouse=True)
def configure_celery_for_tests():
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
        broker_url="memory://",
        result_backend="cache+memory://",
    )
    yield
    celery_app.conf.update(
        task_always_eager=False,
        task_eager_propagates=False,
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

@pytest.fixture(scope="function")
def override_rate_limiter(client, mock_redis):
    test_limiter = RedisRateLimiter(redis_client=mock_redis)
    client.app.dependency_overrides[core.security.get_rate_limiter_dep] = lambda: test_limiter
    yield test_limiter
    client.app.dependency_overrides.clear()

@pytest.fixture
def test_app():
    return fastapi_app

