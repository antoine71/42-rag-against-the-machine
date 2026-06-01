import logging
import os
from pathlib import Path

from rag.exceptions import RAGException

logger = logging.getLogger(__name__)


class FilesRepositoryScannerError(RAGException):
    """Exception raised when an error occurs during repository file scanning.
    """


class FilesRepositoryScanner:
    """Scanner that walks the repository directory to locate valid Python
    and Markdown files.
    """

    def __init__(self, repository: str) -> None:
        """Initializes the FilesRepositoryScanner.

        Args:
            repository: The directory path of the repository to scan.
        """
        self._init_repository(repository)

    def _init_repository(self, repository: str) -> None:
        """Validates and sets the repository path.

        Args:
            repository: The directory path to validate.

        Raises:
            FilesRepositoryScannerError: If the provided path is not a
                directory.
        """
        self._repository_path = Path(repository)
        if not self._repository_path.is_dir():
            raise FilesRepositoryScannerError(
                f"{self._repository_path} is not a directory"
            )

    def _error_handler(self, e: OSError) -> None:
        """Handles directory walking or OS errors gracefully by logging warning
        messages.

        Args:
            e: The OSError exception raised.
        """
        logger.warning(f"Error during indexing: {str(e)}")

    def list_files(self) -> list[Path]:
        """Scans the repository and returns a list of valid Python and Markdown
        files.

        Returns:
            A list of Path objects pointing to valid repository files.
        """
        data_files: list[Path] = []
        for root, _, files in os.walk(
            str(self._repository_path), onerror=self._error_handler
        ):
            for file in files:
                file_path = Path(root) / file
                if self._is_file_valid(file_path):
                    data_files.append(Path(root) / file)
        logger.debug(
            f"Found {len(data_files)} files:\n {[str(f) for f in data_files]}"
        )
        return data_files

    def _is_file_valid(self, file_path: Path) -> bool:
        """Checks if a file is a non-empty Python or Markdown file.

        Uses stat().st_size for checking emptiness to avoid costly full-file
        reading.

        Args:
            file_path: The path of the file to validate.

        Returns:
            True if the file is a valid, non-empty Python or Markdown
                file, False otherwise.
        """
        try:
            return (
                file_path.suffix in {".py", ".md"}
                and file_path.stat().st_size > 0
            )
        except OSError as e:
            self._error_handler(e)
            return False
