from abc import ABC, abstractmethod

from rag.models.question import UnansweredQuestion
from rag.models.search_result import StudentSearchResults


class RetrievingProcessor(ABC):
    @abstractmethod
    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults: ...
