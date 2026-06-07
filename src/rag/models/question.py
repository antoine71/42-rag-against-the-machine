import uuid

from pydantic import BaseModel, Field

from rag.models.minimal_source import MinimalSource


class UnansweredQuestion(BaseModel):
    """Represents a student query without an answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents a query with ground truth sources and an answer."""
    sources: list[MinimalSource]
    answer: str
