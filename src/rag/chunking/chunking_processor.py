from typing import Protocol

from langchain_core.documents import Document


class ChunkingProcessor(Protocol):
    def split_documents(self, documents: list[Document]) -> list[Document]: ...
