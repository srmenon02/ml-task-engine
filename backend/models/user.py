from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.sql import func
from .database import base
import uuid

class User(base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    clerk_id = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=True, index=True)
    tier = Column(String, default="free", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    last_seen_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<User id={self.id} email={self.email} tier={self.tier}>"