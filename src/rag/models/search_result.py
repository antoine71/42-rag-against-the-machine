from typing import Any, Sequence

from pydantic import BaseModel

from rag.models.minimal_source import MinimalSource


class MinimalSearchResults(BaseModel):
    question_id: str
    question_str: str
    retrieved_sources: list[MinimalSource]

    def model_dump_for_moulinette(self) -> dict[str, Any]:
        dump = self.model_dump(exclude={"question"})
        dump["question_str"] = self.question
        return dump


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: Sequence[MinimalSearchResults]
    k: int

    def model_dump_for_moulinette(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "search_results": [
                result.model_dump_for_moulinette()
                for result in self.search_results
            ],
        }


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: Sequence[MinimalAnswer]
