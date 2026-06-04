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
from rag.models.chunk import Chunk, FileType
from rag.tui import TUI

logger = logging.getLogger(__name__)


class ChunkingManager:
    def __init__(
        self,
        config: ChunkingConfig,
        files: list[Path],
        tui: TUI,
        repository: str,
    ) -> None:
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
        logger.debug(f"Splitting files: '{self._files}'")
        for file in self._files:
            yield Document(
                page_content=file.read_text(),
                metadata={
                    "file_path": str(file),
                    "file_name": file.name,
                    "type": FileType.from_file(file),
                    "repository": self._repository,
                },
            )

    def _split_documents(
        self, doc_type: str, splitter: ChunkingProcessor
    ) -> list[Document]:
        documents = [
            doc
            for doc in self._documents_generator()
            if Path(doc.metadata["file_path"]).suffix == doc_type
        ]
        split = splitter.split_documents(documents)
        return split

    def split(self) -> list[Chunk]:
        chunks = list(
            itertools.chain(
                self._split_documents(".py", self._python_splitter),
                self._split_documents(".md", self._markdown_splitter),
                self._split_documents(".txt", self._plain_text_splitter),
            )
        )
        return [Chunk.from_document(chunk) for chunk in chunks]
