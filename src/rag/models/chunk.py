from langchain_core.documents import Document
from pydantic import BaseModel

from rag.models.file_type import FileType
from rag.models.minimal_source import MinimalSource


class Chunk(BaseModel):
    """Represents a chunk of file text with source metadata and offsets.

    Character offsets are stored as inclusive start and end indexes.
    """

    text: str
    file_path: str
    file_name: str
    type: FileType
    first_character_index: int
    last_character_index: int
    repository: str

    @classmethod
    def from_document(cls, document: Document) -> "Chunk":
        """Creates a chunk from a LangChain document.

        Args:
            document: LangChain document containing text and metadata.

        Returns:
            Chunk with source metadata and inclusive character offsets.
        """
        last_character_index = (
            document.metadata["start_index"] + len(document.page_content) - 1
        )
        return cls(
            text=document.page_content,
            file_path=document.metadata["file_path"],
            file_name=document.metadata["file_name"],
            type=document.metadata["type"],
            first_character_index=document.metadata["start_index"],
            last_character_index=last_character_index,
            repository=document.metadata["repository"],
        )

    def to_minimal_source(self) -> MinimalSource:
        """Converts the chunk into minimal source metadata.

        Returns:
            Minimal source pointing to the same file and character range.
        """
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )

    @property
    def metadata(self) -> dict[str, str | int]:
        """Returns chunk metadata excluding the full text content.

        Returns:
            Serializable metadata for index storage.
        """
        return self.model_dump(exclude={"text"})
