import logging
import os
from collections import defaultdict
from pathlib import Path

from rag.exceptions import RAGException
from rag.models.file_category import FileCategory
from rag.models.file_type import FileType
from rag.tui import TUI

logger = logging.getLogger(__name__)


class FilesRepositoryScannerError(RAGException):
    """Exception raised when an error occurs during repository scanning."""


class FilesRepositoryScanner:
    """Scanner that walks the repository directory to locate valid Python
    and Markdown files.
    """

    def __init__(self, repository: Path, tui: TUI) -> None:
        """Initializes the FilesRepositoryScanner.

        Args:
            repository: The directory path of the repository to scan.
        """
        self._repository_path = repository
        self._tui = tui

    def _error_handler(self, e: OSError) -> None:
        """Handles directory walking or OS errors gracefully by logging warning
        messages.

        Args:
            e: The OSError exception raised.
        """
        logger.warning(f"Error during indexing: {str(e)}")

    def list_files(self, files_category: FileCategory) -> list[Path]:
        """Scans the repository and returns a list of valid Python and Markdown
        files.

        Returns:
            A list of Path objects pointing to valid repository files.
        """
        self._tui.print(
            f"Listing {files_category} files from '{self._repository_path}'"
        )
        data_files: list[Path] = []
        for root, _, files in os.walk(
            str(self._repository_path), onerror=self._error_handler
        ):
            for file in files:
                file_path = Path(root) / file
                if self._is_file_valid(file_path, files_category):
                    data_files.append(Path(root) / file)
        logger.debug(
            f"Found {len(data_files)} files:\n {[str(f) for f in data_files]}"
        )
        count = self._count_files(data_files)
        self._tui.print(f"Found {len(data_files)} files: {count}.")
        return data_files

    def _count_files(self, data_files: list[Path]) -> str:
        count: dict[FileType, int] = defaultdict(int)
        for file in data_files:
            count[FileType.from_suffix(file.suffix)] += 1
        return ", ".join(
            f"{file_type}: {file_count}"
            for file_type, file_count in count.items()
        )

    def _is_file_valid(
        self, file_path: Path, files_category: FileCategory
    ) -> bool:
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
                file_path.suffix
                in (
                    file_type.suffix for file_type in files_category.file_types
                )
                and file_path.stat().st_size > 0
            )
        except OSError as e:
            self._error_handler(e)
            return False
