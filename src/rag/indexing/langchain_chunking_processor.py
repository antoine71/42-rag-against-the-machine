import itertools
import logging
from collections.abc import Callable, Generator
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    TextSplitter,
)
from transformers import PreTrainedTokenizerBase

from rag.models.chunk import Chunk

logger = logging.getLogger(__name__)


class LangChainChunkingProcessor:
    def __init__(
        self,
        chunk_size: int,
        files: list[Path],
        tokenize: Callable[[str], list[int]],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self._files = files
        self._tokenize = tokenize
        self._markdown_text_splitter = (
            MarkdownTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=max(10, chunk_size // 20),
                add_start_index=True,
            )
        )
        self._python_text_splitter = (
            PythonCodeTextSplitter.from_huggingface_tokenizer(
                tokenizer,
                chunk_size=chunk_size,
                chunk_overlap=max(10, chunk_size // 20),
                add_start_index=True,
            )
        )

    def _get_document_type(self, file: Path) -> str:
        match file.suffix:
            case ".py":
                return "python"
            case ".md":
                return "markdown"
            case _:
                raise NotImplementedError

    def _documents_generator(self) -> Generator[Document]:
        for file in self._files:
            yield Document(
                page_content=file.read_text(),
                metadata={
                    "source": str(file),
                    "type": self._get_document_type(file),
                },
            )

    def _len_chunk(self, chunk: str) -> int:
        return len(self._tokenize(chunk))

    def _split_documents(
        self, doc_type: str, splitter: TextSplitter
    ) -> list[Document]:
        documents = (
            doc
            for doc in self._documents_generator()
            if doc.metadata["type"] == doc_type
        )
        return splitter.split_documents(documents)

    def split(self) -> list[Chunk]:
        chunks = list(
            itertools.chain(
                self._split_documents("python", self._python_text_splitter),
                self._split_documents(
                    "markdown", self._markdown_text_splitter
                ),
            )
        )
        logger.info(
            f"Split {len(self._files)} files into {len(chunks)} chunks."
        )
        print(chunks[0].metadata)
        return [Chunk.from_document(chunk) for chunk in chunks]
