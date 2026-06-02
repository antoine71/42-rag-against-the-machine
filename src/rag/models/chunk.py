from enum import Enum
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, computed_field

from rag.models.minimal_source import MinimalSource


class FileType(str, Enum):
    """Enum representing supported repository file types."""

    CODE = "code"
    DOCUMENTATION = "documentation"

    @classmethod
    def from_file(cls, file: Path) -> "FileType":
        """Returns the FileType enum value for a given file path."""
        match file.suffix:
            case ".py":
                return cls.CODE
            case ".md" | ".txt":
                return cls.DOCUMENTATION
        raise ValueError(f"Invalid file suffix '{file.suffix}'")


class Chunk(BaseModel):
    """Represents a chunk of file text with source metadata and offsets."""

    text: str
    file_path: str
    type: FileType
    first_character_index: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def last_character_index(self) -> int:
        """Returns the ending character index for the chunk."""
        return self.first_character_index + len(self.text)

    @classmethod
    def from_document(cls, document: Document) -> "Chunk":
        """Creates a Chunk object from a LangChain Document."""
        return cls(
            text=document.page_content,
            file_path=document.metadata["file_path"],
            type=document.metadata["type"],
            first_character_index=document.metadata["start_index"],
        )

    def to_minimal_source(self) -> MinimalSource:
        """Converts the chunk into a MinimalSource metadata object."""
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )

    @property
    def metadata(self) -> dict[str, str | int]:
        """Returns chunk metadata excluding the full text content."""
        return self.model_dump(exclude={"text"})
