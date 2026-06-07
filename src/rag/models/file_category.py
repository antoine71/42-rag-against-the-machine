from enum import Enum
from pathlib import Path

from rag.models.file_type import FileType


class FileCategory(str, Enum):
    """Supported high-level categories of repository files."""

    CODE = "code"
    DOCUMENTATION = "documentation"
    ALL = "all"

    @classmethod
    def from_file(cls, file: Path) -> "FileCategory":
        """Returns the file category for a file path.

        Args:
            file: File path to categorize.

        Returns:
            The matching file category.

        Raises:
            ValueError: If the file suffix is not supported.
        """
        match file.suffix:
            case ".py":
                return cls.CODE
            case ".md" | ".txt":
                return cls.DOCUMENTATION
        raise ValueError(f"Invalid file suffix '{file.suffix}'")

    @property
    def file_types(self) -> tuple[FileType, ...]:
        """Returns concrete file types included in the category.

        Returns:
            File types represented by this category.
        """
        return {
            FileCategory.CODE: (FileType.PYTHON,),
            FileCategory.DOCUMENTATION: (FileType.MARKDOWN, FileType.TEXT),
            FileCategory.ALL: tuple(FileType),
        }[self]
