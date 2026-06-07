from typing import Any, Mapping, cast

import bm25s

from rag.config.bm25_config import BM25Configuration
from rag.models.file_category import FileCategory
from rag.models.indexing_method import IndexingMethod
from rag.retrieving.retrieving_processor import RetrievingProcessor
from rag.tui import TUI
from rag.utils.files_manager import FilesManager
from rag.utils.measure import measure


class BM25RetrievingProcessor(RetrievingProcessor):
    """Retrieving processor that uses a saved BM25 sparse index for
    keyword search retrieval.
    """

    WEIGHT = 1.9

    def __init__(
        self, index_directory: str, tui: TUI, config: BM25Configuration
    ) -> None:
        """Initializes the BM25RetrievingProcessor.

        Args:
            index_directory: The directory path where the BM25 index is saved.
        """
        super().__init__(index_directory, tui, config)
        self._config: BM25Configuration

    def _load_and_retrieve(
        self, file_type: FileCategory, processed_queries: list[str], k: int
    ) -> list[list[Mapping[str, Any]]]:
        chunks_index = FilesManager.get_indexing_directory(
            self._index_directory, IndexingMethod.BM25, file_type
        )
        query_tokens = bm25s.tokenize(processed_queries)
        retriever: bm25s.BM25 = bm25s.BM25().load(
            chunks_index,
            load_corpus=True,
            **self._config.bm25_settings.model_dump(),
        )
        (results, _), delta = measure(
            retriever.retrieve, query_tokens, show_progress=True, k=k
        )
        self._tui.print_task_report(
            f"{self._config.TYPE} retrieving",
            delta,
            "questions",
            len(processed_queries),
            new_line_after=True,
        )
        return cast(list[list[Mapping[str, Any]]], results)
