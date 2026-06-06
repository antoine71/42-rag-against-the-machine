from enum import Enum
from pathlib import Path

from rag.models.file_type import FileType


class FileCategory(str, Enum):
    """Enum representing supported repository file types."""

    CODE = "code"
    DOCUMENTATION = "documentation"
    ALL = "all"

    @classmethod
    def from_file(cls, file: Path) -> "FileCategory":
        """Returns the FileType enum value for a given file path."""
        match file.suffix:
            case ".py":
                return cls.CODE
            case ".md" | ".txt":
                return cls.DOCUMENTATION
        raise ValueError(f"Invalid file suffix '{file.suffix}'")

    @property
    def file_types(self) -> tuple[FileType, ...]:
        return {
            FileCategory.CODE: (FileType.PYTHON,),
            FileCategory.DOCUMENTATION: (FileType.MARKDOWN, FileType.TEXT),
            FileCategory.ALL: tuple(FileType),
        }[self]
