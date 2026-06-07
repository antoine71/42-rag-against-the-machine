from typing import Any, Mapping, Sequence

from pydantic import AliasChoices, BaseModel, Field

from rag.models.minimal_source import MinimalSource
from rag.models.question import UnansweredQuestion


class MinimalSearchResults(BaseModel):
    """Represents a minimal retrieval result for a single query."""

    question_id: str
    question: str = Field(
        validation_alias=AliasChoices("question", "question_str"),
        serialization_alias="question_str",
    )
    retrieved_sources: list[MinimalSource]

    @classmethod
    def from_query_and_sources(
        cls, query: UnansweredQuestion, sources: list[Mapping[str, Any]]
    ) -> "MinimalSearchResults":
        """Create a MinimalSearchResults instance from a query and sources."""
        return cls(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=[MinimalSource(**source) for source in sources],
        )


class MinimalAnswer(MinimalSearchResults):
    """Extends MinimalSearchResults with a generated answer text."""

    answer: str


class StudentSearchResults(BaseModel):
    """Represents student search results for multiple queries."""

    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Represents returned answers paired with student search results."""

    search_results: Sequence[MinimalAnswer]
