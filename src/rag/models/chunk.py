from langchain_core.documents import Document
from pydantic import BaseModel, computed_field

from rag.models.minimal_source import MinimalSource


class Chunk(BaseModel):
    text: str
    source: str
    type: str
    start_index: int

    @computed_field
    @property
    def end_index(self) -> int:
        return self.start_index + len(self.text)

    @classmethod
    def from_document(cls, document: Document) -> "Chunk":
        return cls(text=document.page_content, **document.metadata)

    def to_minimal_source(self) -> MinimalSource:
        return MinimalSource(
            file_path=self.source,
            first_character_index=self.start_index,
            last_character_index=self.end_index,
        )

    @property
    def metadata(self) -> dict[str, str | int]:
        return self.model_dump(exclude={"text"})
