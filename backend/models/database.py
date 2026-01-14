from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

if not ENV_PATH.exists():
    raise RuntimeError(f".env file not found at {ENV_PATH}")

load_dotenv(ENV_PATH)

db_url = os.getenv("DB_URL")
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