import itertools
import logging
from collections.abc import Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from rag.chunking.chunking_processor import ChunkingProcessor
from rag.chunking.markdown_chunking_processor import MarkdownChunkingProcessor
from rag.config.chunking_config import ChunkingConfig
from rag.models.chunk import Chunk
from rag.models.file_type import FileType
from rag.tui import TUI

logger = logging.getLogger(__name__)


class ChunkingManager:
    """Splits repository files into typed text chunks."""

    def __init__(
        self,
        config: ChunkingConfig,
        files: list[Path],
        tui: TUI,
        repository: str,
    ) -> None:
        """Initializes splitters for supported file types.

        Args:
            config: Chunking configuration.
            files: Files to split into chunks.
            tui: Terminal UI used for progress output.
            repository: Repository root path used in chunk metadata.
        """
        self._files = files
        self._tui = tui
        self._config = config
        self._repository = repository
        self._markdown_splitter: ChunkingProcessor = MarkdownChunkingProcessor(
            config.chunk_size, config.overlap, self._tui
        )
        self._python_splitter: ChunkingProcessor = (
            RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.overlap,
                add_start_index=True,
                length_function=len,
            )
        )
        self._plain_text_splitter: ChunkingProcessor = (
            RecursiveCharacterTextSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.overlap,
                add_start_index=True,
                length_function=len,
            )
        )

    def _documents_generator(self) -> Generator[Document]:
        """Yields LangChain documents with source metadata.

        Yields:
            Documents built from repository files.
        """
        logger.debug(f"Splitting files: '{self._files}'")
        for file in self._files:
            yield Document(
                page_content=file.read_text(),
                metadata={
                    "file_path": str(file),
                    "file_name": file.name,
                    "type": FileType.from_suffix(file.suffix),
                    "repository": self._repository,
                },
            )

    def _split_documents(
        self, doc_type: str, splitter: ChunkingProcessor
    ) -> list[Document]:
        """Splits documents matching a specific suffix.

        Args:
            doc_type: File suffix to select, including the leading dot.
            splitter: Splitter used for matching documents.

        Returns:
            Split documents produced by the splitter.
        """
        documents = [
            doc
            for doc in self._documents_generator()
            if Path(doc.metadata["file_path"]).suffix == doc_type
        ]
        split = splitter.split_documents(documents)
        return split

    def split(self) -> list[Chunk]:
        """Splits all configured files into chunks.

        Returns:
            Repository chunks for supported file types.
        """
        chunks = list(
            itertools.chain(
                self._split_documents(".py", self._python_splitter),
                self._split_documents(".md", self._markdown_splitter),
                self._split_documents(".txt", self._plain_text_splitter),
            )
        )
        return [Chunk.from_document(chunk) for chunk in chunks]
