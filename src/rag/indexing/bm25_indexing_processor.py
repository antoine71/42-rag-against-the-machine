import logging

import bm25s

from rag.config.bm25_config import BM25Configuration
from rag.config.indexing_config import IndexingConfig
from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk
from rag.models.file_category import FileCategory
from rag.models.indexing_method import IndexingMethod
from rag.tui import TUI
from rag.utils.files_manager import FilesManager
from rag.utils.measure import measure

logger = logging.getLogger(__name__)


class BM25IndexingProcessor(IndexingProcessor):
    """An indexing processor that creates and saves a BM25 sparse index."""

    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: IndexingConfig
    ) -> None:
        """Initializes the BM25IndexingProcessor.

        Args:
            chunks: A list of Chunk models to be indexed.
            tui: A TUI instance to handle user interface / progress output.
            config: A configuration object containing BM25 parameters.
        """
        super().__init__(chunks, tui, config)
        self._config: BM25Configuration

    def _index_and_save(
        self,
        processed_chunks: list[Chunk],
        save_directory: str,
        file_type: FileCategory,
    ) -> str:
        """Builds and saves a BM25 index for processed chunks.

        Args:
            processed_chunks: Text-processed chunks to index.
            save_directory: Root directory where the index is saved.
            file_type: File category represented by the index.

        Returns:
            Directory containing the saved BM25 index.
        """
        texts = [chunk.text for chunk in processed_chunks]
        corpus = bm25s.tokenize(texts)
        retriever = bm25s.BM25(**self._config.bm25_settings.model_dump())
        _, delta = measure(
            retriever.index, corpus, show_progress=True, leave_progress=False
        )
        self._tui.print_task_report(
            f"{self._config.TYPE} indexing", delta, "chunks", len(texts)
        )

        save_path = FilesManager.get_indexing_directory(
            save_directory, IndexingMethod.BM25, file_type
        )
        _, delta = measure(
            retriever.save,
            save_path,
            corpus=[c.model_dump() for c in processed_chunks],
            show_progress=False,
        )
        self._tui.print_task_report(
            "data persistence",
            delta,
            "chunks",
            len(texts),
        )
        return save_path
