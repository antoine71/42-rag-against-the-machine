from typing import Protocol

from rag.models.question import UnansweredQuestion
from rag.models.search_result import StudentSearchResults


class RetrievingProcessor(Protocol):
    """Protocol defining the interface for all retrieving processors."""

    WEIGHT: float

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        """Retrieves top-k most relevant sources for the given queries.

        Args:
            queries: A list of UnansweredQuestion objects containing the
                search queries.
            k: The number of top results to retrieve.

        Returns:
            A StudentSearchResults object containing retrieved sources for each
                query.
        """
        ...
