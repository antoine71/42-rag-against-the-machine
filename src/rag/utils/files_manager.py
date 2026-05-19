import json
from pathlib import Path

from pydantic import ValidationError

from rag.exceptions import RAGException
from rag.models.minimal_source import MinimalSource
from rag.models.question import UnansweredQuestion
from rag.models.search_result import StudentSearchResults


class FilesManagerError(RAGException):
    pass


class FilesManager:
    def save_results(
        self, results: StudentSearchResults, file_path: str
    ) -> None:
        file_path_obj = Path(file_path)
        try:
            file_path_obj.parent.mkdir(parents=True, exist_ok=True)
            file_path_obj.write_text(
                json.dumps(
                    results.model_dump_for_moulinette(),
                    indent=4,
                )
            )
        except OSError as e:
            raise FilesManagerError(
                f"Failed to save results '{file_path}': {str(e)}"
            )

    def load_dataset(self, dataset_path: str) -> list[UnansweredQuestion]:
        try:
            file_content_obj = json.loads(Path(dataset_path).read_text())
            questions = file_content_obj["rag_questions"]
            return [UnansweredQuestion(**data) for data in questions]
        except (OSError, ValidationError, json.JSONDecodeError) as e:
            raise FilesManagerError(
                f"Failed to load dataset '{dataset_path}': {e}"
            )

    def load_chunk(self, source: MinimalSource) -> str:
        try:
            content = Path(source.file_path).read_text()
        except OSError as e:
            raise FilesManagerError(
                f"Failed to open file '{source.file_path}': {e}"
            )
        try:
            return content[
                source.first_character_index : source.last_character_index + 1
            ]
        except IndexError as e:
            raise FilesManagerError(
                f"Failed extract chunk from file '{source.file_path}': {e}"
            )

    def load_search_results(self, file: str) -> StudentSearchResults:
        try:
            content = Path(file).read_text()
        except OSError as e:
            raise FilesManagerError(f"Failed to open file '{file}': {e}")
        try:
            return StudentSearchResults(**json.loads(content))
        except (json.JSONDecodeError, ValidationError) as e:
            raise FilesManagerError(
                f"Failed to load search results from file '{file}': {e}"
            )
