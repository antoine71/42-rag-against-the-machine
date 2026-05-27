from typing import Protocol

from rag.models.question import UnansweredQuestion
from rag.models.search_result import StudentSearchResults


class RetrievingProcessor(Protocol):
    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults: ...
