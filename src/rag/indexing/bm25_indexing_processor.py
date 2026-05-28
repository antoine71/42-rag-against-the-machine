import logging

import bm25s
from pydantic_settings import BaseSettings

from rag.config.bm25 import BM25Configuration
from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk
from rag.tui import TUI

logger = logging.getLogger(__name__)


class BM25IndexingProcessor(IndexingProcessor):
    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: BaseSettings
    ) -> None:
        super().__init__(chunks, tui, config)
        self._config: BM25Configuration

    def index_corpus(
        self,
        save_directory: str,
    ) -> None:

        texts = [chunk.text for chunk in self._chunks]
        corpus = bm25s.tokenize(texts)
        retriever = bm25s.BM25(**self._config.bm25_settings.model_dump())
        retriever.index(corpus, show_progress=True, leave_progress=True)
        retriever.save(
            save_directory,
            corpus=[c.model_dump() for c in self._chunks],
            show_progress=False,
        )
