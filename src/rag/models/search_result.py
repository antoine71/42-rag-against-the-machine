from pydantic import AliasChoices, BaseModel, Field

from rag.models.minimal_source import MinimalSource


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str = Field(
        validation_alias=AliasChoices("question", "question_str"),
        serialization_alias="question_str",
    )
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: list[MinimalAnswer]
