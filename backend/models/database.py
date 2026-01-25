from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    print(".env not found, using OS environment variables")

ENVIRONMENT = os.environ.get("ENVIRONMENT")

if ENVIRONMENT == "production":
    db_url = os.environ.get("DB_URL")
elif ENVIRONMENT == "ci":
    db_url = os.environ.get("DB_URL_CI")
else:
    db_url = os.environ.get("DB_URL_DEV")

if not db_url:
    raise ValueError(f"DATABASE URL env variable not set for {ENVIRONMENT}")

if not db_url:
    raise ValueError("DATABASE URL env variable not set")

if ENVIRONMENT == "development":
    engine = create_engine(
        db_url,
        pool_pre_ping = True,
        pool_size=10,
        max_overflow=20
    )
elif ENVIRONMENT == "production":
    engine = create_engine(
        db_url,
        pool_pre_ping = True,
        pool_size=20,
        max_overflow=40,
        pool_recycle=3600,
        echo = False
    )
else:
    engine = create_engine(
        db_url,
        pool_pre_ping = True,
        pool_size=10,
    )

print(f"ENGINE URL: {engine.url}")
local_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

base = declarative_base()

def get_db():
    db = local_session()
    try:
        yield db
    finally:
        db.close()