from pydantic import BaseModel

from rag.models.question import AnsweredQuestion, UnansweredQuestion


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
