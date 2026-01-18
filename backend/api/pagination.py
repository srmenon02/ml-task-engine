from typing import TypeVar, Generic, List, Optional
from pydantic import BaseModel, Field
from math import ceil

T = TypeVar('T')

class PaginationParams(BaseModel):
    page: int = Field(default = 1, ge = 1, description = "Page # (1-indexed)")
    page_size: int = Field(default = 50, ge = 1, le = 100, description = "Items/page")

    def offset(self) -> int:
        return (self.page - 1) * self.page_size
    
    def limit(self) -> int:
        return self.page_size
    
class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool

    @classmethod
    def create(
        cls,
        items: List[T],
        total: int,
        params: PaginationParams
    ):
        total_pages = ceil(total / params.page_size) if total > 0 else 0

        return cls(
            items = items,
            total = total,
            page = params.page,
            page_size = params.page_size,
            total_pages = total_pages,
            has_next = params.page < total_pages,
            has_prev = params.page > 1
        )
    
def paginate_query(query, params: PaginationParams):
    total = query.count()
    items = query.offset(params.offset()).limit(params.limit()).all()
    return items, total