import itertools
import logging
from collections.abc import Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    TextSplitter,
)

from rag.models.chunk import Chunk, FileType

logger = logging.getLogger(__name__)


class LangChainChunkingProcessor:
    def __init__(
        self,
        chunk_size: int,
        files: list[Path],
    ) -> None:
        self._files = files
        self._markdown_text_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=max(10, chunk_size // 20),
            add_start_index=True,
            length_function=len,
        )
        self._python_text_splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=max(10, chunk_size // 20),
            add_start_index=True,
            length_function=len,
        )

    def _documents_generator(self) -> Generator[Document]:
        logger.debug(f"Splitting files: '{self._files}'")
        for file in self._files:
            yield Document(
                page_content=file.read_text(),
                metadata={
                    "file_path": str(file),
                    "type": FileType.from_file(file),
                },
            )

    def _split_documents(
        self, doc_type: str, splitter: TextSplitter
    ) -> list[Document]:
        documents = (
            doc
            for doc in self._documents_generator()
            if doc.metadata["type"] == doc_type
        )
        split = splitter.split_documents(documents)
        return split

    def split(self) -> list[Chunk]:
        chunks = list(
            itertools.chain(
                self._split_documents("python", self._python_text_splitter),
                self._split_documents(
                    "markdown", self._markdown_text_splitter
                ),
            )
        )
        return [Chunk.from_document(chunk) for chunk in chunks]
