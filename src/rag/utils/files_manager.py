import json
from pathlib import Path
from typing import Literal, overload

from pydantic import ValidationError

from rag.exceptions import RAGException
from rag.models.minimal_source import MinimalSource
from rag.models.question import AnsweredQuestion, UnansweredQuestion
from rag.models.rag_dataset import RagDataset
from rag.models.search_result import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class FilesManagerError(RAGException):
    pass


class FilesManager:
    def save_results(
        self,
        results: StudentSearchResults | StudentSearchResultsAndAnswer,
        file_path: str,
    ) -> None:
        file_path_obj = Path(file_path)
        try:
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            file_path_obj.write_text(
                results.model_dump_json(
                    by_alias=True,
                    indent=4,
                )
            )
        except OSError as e:
            raise FilesManagerError(
                f"Failed to save results '{file_path}': {str(e)}"
            ) from e

    @overload
    def load_dataset(
        self,
        dataset_path: str,
        dataset_type: Literal["answered_questions"],
    ) -> RagDataset[AnsweredQuestion]: ...

    @overload
    def load_dataset(
        self,
        dataset_path: str,
        dataset_type: Literal["unanswered_questions"],
    ) -> RagDataset[UnansweredQuestion]: ...

    def load_dataset(
        self,
        dataset_path: str,
        dataset_type: Literal["answered_questions", "unanswered_questions"],
    ) -> RagDataset[AnsweredQuestion] | RagDataset[UnansweredQuestion]:
        try:
            file_content_obj = json.loads(Path(dataset_path).read_text())
            if dataset_type == "answered_questions":
                return RagDataset[AnsweredQuestion](**file_content_obj)
            else:
                return RagDataset[UnansweredQuestion](**file_content_obj)
        except (
            OSError,
            ValidationError,
            json.JSONDecodeError,
        ) as e:
            raise FilesManagerError(
                f"Failed to load dataset '{dataset_path}': {e}"
            ) from e

    def load_chunk(self, source: MinimalSource) -> str:
        try:
            content = Path(source.file_path).read_text()
        except OSError as e:
            raise FilesManagerError(
                f"Failed to open file '{source.file_path}': {e}"
            ) from e
        try:
            start = source.first_character_index
            end = source.last_character_index + 1
            return content[start:end]
        except IndexError as e:
            raise FilesManagerError(
                f"Failed extract chunk from file '{source.file_path}': {e}"
            ) from e

    def load_search_results(self, file: str) -> StudentSearchResults:
        try:
            content = Path(file).read_text()
        except OSError as e:
            raise FilesManagerError(
                f"Failed to open file '{file}': {e}"
            ) from e
        try:
            return StudentSearchResults(**json.loads(content))
        except (json.JSONDecodeError, ValidationError) as e:
            raise FilesManagerError(
                f"Failed to load search results from file '{file}': {e}"
            ) from e
