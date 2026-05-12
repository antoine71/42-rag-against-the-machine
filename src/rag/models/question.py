import uuid
from typing import Annotated

from pydantic import BaseModel, Field

from rag.models.minimal_source import MinimalSource


class UnansweredQuestion(BaseModel):
    question_id: Annotated[
        str, Field(default_factory=lambda: str(uuid.uuid4()))
    ]
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str
