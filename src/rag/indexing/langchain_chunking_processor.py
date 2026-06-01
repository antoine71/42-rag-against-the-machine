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
    """Processor that splits code and markdown documents into logical chunks using LangChain splitters."""

    def __init__(
        self,
        chunk_size: int,
        files: list[Path],
    ) -> None:
        """Initializes the LangChainChunkingProcessor.

        Args:
            chunk_size: The maximum character length of each chunk.
            files: A list of file Path objects to be split.
        """
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
        """Generates LangChain Document objects from files.

        Yields:
            A Document object with loaded page content and path metadata.
        """
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
        """Splits files of a specific type using the provided text splitter.

        Args:
            doc_type: The string representing file type (e.g., 'python' or 'markdown').
            splitter: The LangChain TextSplitter instance to use.

        Returns:
            A list of split Document chunks.
        """
        documents = (
            doc
            for doc in self._documents_generator()
            if doc.metadata["type"] == doc_type
        )
        split = splitter.split_documents(documents)
        return split

    def split(self) -> list[Chunk]:
        """Splits all valid files into Python and Markdown chunks and wraps them in Chunk models.

        Returns:
            A list of parsed Chunk objects with verified starting character indices.
        """
        chunks = list(
            itertools.chain(
                self._split_documents("python", self._python_text_splitter),
                self._split_documents(
                    "markdown", self._markdown_text_splitter
                ),
            )
        )
        return [Chunk.from_document(chunk) for chunk in chunks]
