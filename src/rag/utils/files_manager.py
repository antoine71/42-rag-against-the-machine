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
    """Utility class to manage files, load datasets, retrieve chunks, and
    save results.
    """

    def save_results(
        self,
        results: StudentSearchResults | StudentSearchResultsAndAnswer,
        file_path: str,
    ) -> None:
        """Saves search or answer results to a JSON file.

        Args:
            results: The StudentSearchResults or StudentSearchResultsAndAnswer
                object to save.
            file_path: The file path where the results should be saved.

        Raises:
            FilesManagerError: If writing to the file fails.
        """
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
    ) -> RagDataset[AnsweredQuestion]:
        """Overload for loading answered question datasets."""

    @overload
    def load_dataset(
        self,
        dataset_path: str,
        dataset_type: Literal["unanswered_questions"],
    ) -> RagDataset[UnansweredQuestion]:
        """Overload for loading unanswered question datasets."""

    def load_dataset(
        self,
        dataset_path: str,
        dataset_type: Literal["answered_questions", "unanswered_questions"],
    ) -> RagDataset[AnsweredQuestion] | RagDataset[UnansweredQuestion]:
        """Load and parse a RAG question dataset from a JSON file.

        Args:
            dataset_path: The path of the JSON dataset file.
            dataset_type: The type of dataset ('answered_questions' or
                'unanswered_questions').

        Returns:
            A RagDataset populated with AnsweredQuestion or UnansweredQuestion
                objects.

        Raises:
            FilesManagerError: If loading, parsing, or Pydantic validation
                fails.
        """
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
        """Read and extract a specific text chunk from a file.

        Args:
            source: A MinimalSource object containing the file path and
                character bounds.

        Returns:
            The extracted chunk text content.

        Raises:
            FilesManagerError: If reading the file or extracting the
                character range fails.
        """
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
        """Load and parse student search results from a JSON file.

        Args:
            file: Path of the JSON file containing the search results.

        Returns:
            A StudentSearchResults object.

        Raises:
            FilesManagerError: If loading, parsing, or Pydantic validation
                fails.
        """
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
