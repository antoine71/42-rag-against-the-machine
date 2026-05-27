import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

from rag.indexing.bm25_configuration import BM25Configuration
from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk, FileType
from rag.tui import TUI

logger = logging.getLogger(__name__)


class BM25MultiIndexingProcessor(IndexingProcessor):
    stemmer_map: dict[FileType, bool] = {
        FileType.MARKDOWN: True,
        FileType.PYTHON: False,
    }

    def __init__(self, chunks: list[Chunk], tui: TUI) -> None:
        super().__init__(chunks, tui)

    def index_corpus(self, save_directory: str) -> None:
        for file_type in FileType:
            self._ui.print(f"Indexing {file_type} files...")
            self._index_by_type(save_directory, file_type)

    def _index_by_type(self, save_directory: str, file_type: FileType) -> None:
        chunks = [c for c in self._chunks if c.type == file_type]
        texts = [c.text for c in chunks]
        if self.stemmer_map[file_type]:
            stemmer = Stemmer.Stemmer("english")
            corpus = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer)
        else:
            corpus = bm25s.tokenize(texts)
        retriever = bm25s.BM25()
        retriever.index(corpus, show_progress=True, leave_progress=True)
        retriever.save(
            Path(save_directory) / file_type,
            corpus=[c.model_dump() for c in chunks],
            show_progress=False,
        )


class BM25IndexingProcessor(IndexingProcessor):
    def __init__(
        self,
        chunks: list[Chunk],
        tui: TUI,
        config: dict[str, Any],
        text_formatter: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(chunks, tui, config)
        if not config:
            self._config = BM25Configuration.default_config
        self._text_formatter = text_formatter

    def index_corpus(
        self,
        save_directory: str,
    ) -> None:
        if self._text_formatter is not None:
            texts = [
                self._text_formatter(chunk.text) for chunk in self._chunks
            ]
        else:
            texts = [chunk.text for chunk in self._chunks]
        corpus = bm25s.tokenize(texts)
        retriever = bm25s.BM25(**self._config)
        retriever.index(corpus, show_progress=True, leave_progress=True)
        retriever.save(
            save_directory,
            corpus=[c.model_dump() for c in self._chunks],
            show_progress=False,
        )
