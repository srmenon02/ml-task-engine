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

db_url = os.environ.get("DB_URL")

if not db_url:
    raise ValueError("DATABASE URL env variable not set")

engine = create_engine(
    db_url,
    pool_pre_ping = True,
    pool_size=10,
    max_overflow=20
)

local_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

base = declarative_base()

def get_db():
    db = local_session()
    try:
        yield db
    finally:
        db.close()