from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os


db_url = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/taskengine")

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