import logging
from pathlib import Path

from rag.exceptions import RAGException

logger = logging.getLogger(__name__)


class FilesRepositoryScannerError(RAGException):
    pass


class FilesRepositoryScanner:
    def __init__(self, repository: str) -> None:
        self._init_repository(repository)

    def _init_repository(self, repository: str) -> None:
        self._repository_path = Path(repository)
        if not self._repository_path.is_dir():
            raise FilesRepositoryScannerError(
                f"{self._repository_path} is not a directory"
            )

    def _error_handler(self, e: OSError) -> None:
        logger.warning(f"Error during indexing: {str(e)}")

    def list_files(self) -> list[Path]:
        data_files: list[Path] = []
        for root, _, files in self._repository_path.walk(
            on_error=self._error_handler
        ):
            for file in files:
                file_path = root / file
                if self._is_file_valid(file_path):
                    data_files.append(root / file)
        return data_files

    def _is_file_valid(self, file_path: Path) -> bool:
        try:
            return (
                file_path.suffix in {".py", ".md"}
                and file_path.read_text().strip() != ""
            )
        except OSError as e:
            self._error_handler(e)
            return False
