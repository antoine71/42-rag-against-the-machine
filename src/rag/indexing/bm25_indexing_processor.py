import logging

import bm25s
from pydantic_settings import BaseSettings

from rag.config.bm25 import BM25Configuration
from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk
from rag.tui import TUI

logger = logging.getLogger(__name__)


class BM25IndexingProcessor(IndexingProcessor):
    """An indexing processor that creates and saves a BM25 sparse index."""

    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: BaseSettings
    ) -> None:
        """Initializes the BM25IndexingProcessor.

        Args:
            chunks: A list of Chunk models to be indexed.
            tui: A TUI instance to handle user interface / progress output.
            config: A configuration object containing BM25 parameters.
        """
        super().__init__(chunks, tui, config)
        self._config: BM25Configuration

    def index_corpus(
        self,
        save_directory: str,
    ) -> None:
        """Indexes the text chunks using BM25 and saves the index to disk.

        Args:
            save_directory: The directory path where the BM25 index files
                should be saved.
        """

        texts = [chunk.text for chunk in self._chunks]
        corpus = bm25s.tokenize(texts)
        retriever = bm25s.BM25(**self._config.bm25_settings.model_dump())
        retriever.index(corpus, show_progress=True, leave_progress=True)
        retriever.save(
            save_directory,
            corpus=[c.model_dump() for c in self._chunks],
            show_progress=False,
        )
