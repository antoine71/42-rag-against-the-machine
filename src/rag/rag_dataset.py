from typing import Generic, TypeVar

from pydantic import BaseModel

from rag.models.question import AnsweredQuestion, UnansweredQuestion

QuestionT = TypeVar("QuestionT", AnsweredQuestion, UnansweredQuestion)


class RagDataset(BaseModel, Generic[QuestionT]):
    rag_questions: list[QuestionT]
