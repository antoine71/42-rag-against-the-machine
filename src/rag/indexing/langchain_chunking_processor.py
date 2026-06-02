import itertools
import logging
from collections.abc import Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from rag.models.chunk import Chunk, FileType

logger = logging.getLogger(__name__)


class MarkdownChunkingProcessor:
    def __init__(self) -> None:
        self._headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers_to_split_on, strip_headers=False
        )

    def _enrich_header_splits(self, md_header_splits: list[Document]) -> None:
        start_index = 0
        for doc in md_header_splits:
            first_char_index = start_index
            doc.metadata["first_character_index"] = start_index
            start_index += len(doc.page_content)

    def _header_split(self, text: str) -> list[Document]:
        md_header_splits = self._markdown_splitter.split_text(text)

        start_index = 0
        for doc in md_header_splits:
            doc.metadata["first_character_index"] = start_index
            start_index += len(doc.page_content)

    def split(
        self, text: str, max_chunk_size: int, overlap: float
    ) -> list[Document]:

        start_index = 0
        for doc in md_header_splits:
            doc.metadata["first_character_index"] = start_index
            start_index += len(doc.page_content)

        # Char-level splits
        chunk_size = max_chunk_size
        chunk_overlap = int(max_chunk_size * overlap)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(md_header_splits)
        for doc in splits:
            doc.metadata["first_character_index"] += doc.metadata[
                "start_index"
            ]
            del doc.metadata["start_index"]
        return splits


class LangChainChunkingProcessor:
    """Processor that splits code and markdown documents into logical chunks
    using LangChain splitters."""

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
        self._markdown_text_splitter = MarkdownHeaderTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=max(10, chunk_size // 5),
            add_start_index=True,
            length_function=len,
        )
        self._python_text_splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=max(10, chunk_size // 5),
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
            doc_type: The string representing file type (e.g., 'python' or
                'markdown').
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
        """Splits all valid files into Python and Markdown chunks and wraps
        them in Chunk models.

        Returns:
            A list of parsed Chunk objects with verified starting character
                indices.
        """
        chunks = list(
            itertools.chain(
                self._split_documents(
                    FileType.CODE, self._python_text_splitter
                ),
                self._split_documents(
                    FileType.DOCUMENTATION, self._markdown_text_splitter
                ),
            )
        )
        return [Chunk.from_document(chunk) for chunk in chunks]
