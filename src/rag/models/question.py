import uuid

from pydantic import BaseModel

from rag.models.minimal_source import MinimalSource


class UnansweredQuestion(BaseModel):
    question_id: str = str(uuid.uuid4())
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str
