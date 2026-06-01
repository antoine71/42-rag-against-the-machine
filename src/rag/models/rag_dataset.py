from typing import Generic, TypeVar

from pydantic import BaseModel

from rag.models.question import AnsweredQuestion, UnansweredQuestion

QuestionT = TypeVar("QuestionT", AnsweredQuestion, UnansweredQuestion)


class RagDataset(BaseModel, Generic[QuestionT]):
    """Generic dataset container for answered or unanswered RAG questions."""

    rag_questions: list[QuestionT]
