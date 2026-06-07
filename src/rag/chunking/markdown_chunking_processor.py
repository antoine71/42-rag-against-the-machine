import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from rag.chunking.chunking_processor import ChunkingProcessor
from rag.tui import TUI

logger = logging.getLogger(__name__)


class MarkdownChunkingProcessor(ChunkingProcessor):
    """Chunking processor for Markdown documents."""

    def __init__(self, max_chunk_size: int, overlap: int, tui: TUI) -> None:
        """Initializes the Markdown recursive text splitter.

        Args:
            max_chunk_size: Maximum chunk size in characters.
            overlap: Chunk overlap in characters.
            tui: Terminal UI used by chunking processors.
        """
        self._markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
            length_function=len,
        )
        self._tui = tui

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits Markdown documents.

        Args:
            documents: Markdown documents to split.

        Returns:
            Split Markdown documents with start indexes.
        """

        return self._markdown_splitter.split_documents(documents)
