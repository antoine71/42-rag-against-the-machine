from typing import Any

from numpy.typing import NDArray
from pydantic import AliasChoices, BaseModel, Field

from rag.models.minimal_source import MinimalSource
from rag.models.question import UnansweredQuestion


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str = Field(
        validation_alias=AliasChoices("question", "question_str"),
        serialization_alias="question_str",
    )
    retrieved_sources: list[MinimalSource]

    @classmethod
    def from_query_and_sources(
        cls, query: UnansweredQuestion, sources: NDArray[Any]
    ) -> "MinimalSearchResults":
        return cls(
            question_id=query.question_id,
            question=query.question,
            retrieved_sources=[MinimalSource(**source) for source in sources],
        )


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: list[MinimalAnswer]
