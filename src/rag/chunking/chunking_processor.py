from typing import Protocol

from langchain_core.documents import Document


class ChunkingProcessor(Protocol):
    """Protocol implemented by document splitters."""

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits documents into smaller documents.

        Args:
            documents: Documents to split.

        Returns:
            Split documents.
        """
        ...
