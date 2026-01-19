from fastapi import APIRouter
from ..jobs import router as jobs_router
from .bulk import router as bulk_router

v1_router = APIRouter(prefix = "/v1")

v1_router.include_router(jobs_router)
v1_router.include_router(bulk_router)

__all__ = ["v1_router"]