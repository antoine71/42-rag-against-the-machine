from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, computed_field
from strenum import StrEnum

from rag.models.minimal_source import MinimalSource


class FileType(StrEnum):
    PYTHON = "python"
    MARKDOWN = "markdown"

    @classmethod
    def from_file(cls, file: Path) -> "FileType":
        match file.suffix:
            case ".py":
                return cls.PYTHON
            case ".md":
                return cls.MARKDOWN
        raise ValueError(f"Invalid file suffix '{file.suffix}'")


class Chunk(BaseModel):
    text: str
    file_path: str
    type: FileType
    first_character_index: int

    @property
    @computed_field
    def last_character_index(self) -> int:
        return self.first_character_index + len(self.text)

    @classmethod
    def from_document(cls, document: Document) -> "Chunk":
        return cls(
            text=document.page_content,
            file_path=document.metadata["file_path"],
            type=document.metadata["type"],
            first_character_index=document.metadata["start_index"],
        )

    def to_minimal_source(self) -> MinimalSource:
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )

    @property
    def metadata(self) -> dict[str, str | int]:
        return self.model_dump(exclude={"text"})
