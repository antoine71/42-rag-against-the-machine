from abc import ABC, abstractmethod
from typing import Any, Mapping

from rag.config.retrieving_config import RetrievingConfig
from rag.models.file_category import FileCategory
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.text_processing.pipeline_factory import TextProcessingPipelineFactory
from rag.tui import TUI
from rag.utils.measure import measure


class RetrievingProcessor(ABC):
    """Protocol defining the interface for all retrieving processors."""

    WEIGHT: float

    def __init__(
        self, index_directory: str, tui: TUI, config: RetrievingConfig
    ) -> None:
        """Initializes the VectorRetrievingProcessor.

        Args:
            index_directory: Path to ChromaDB database files.
            tui: A TUI instance to handle progress output.
        """
        self._index_directory = index_directory
        self._tui = tui
        self._config = config

    def _queries_text_processing(
        self, file_type: FileCategory, queries: list[UnansweredQuestion]
    ) -> list[str]:
        query_processing_pipeline_factory = TextProcessingPipelineFactory(
            self._config.query_processing, self._tui
        )
        query_processing_pipeline = query_processing_pipeline_factory.create(
            file_type
        )
        processed_queries, delta = measure(
            query_processing_pipeline.process_list,
            [query.question for query in queries],
        )

        self._tui.print_task_report(
            "Processing questions",
            delta,
            "questions",
            len(queries),
        )
        return processed_queries

    def retrieve(
        self,
        queries: list[UnansweredQuestion],
        k: int,
        file_type: FileCategory,
    ) -> StudentSearchResults:
        """Retrieves top-k most relevant sources for the given queries.

        Args:
            queries: A list of UnansweredQuestion objects containing the
                search queries.
            k: The number of top results to retrieve.

        Returns:
            A StudentSearchResults object containing retrieved sources for each
                query.
        """
        self._tui.print_phase_title(f"{self._config.TYPE}")
        processed_queries = self._queries_text_processing(file_type, queries)
        results = self._load_and_retrieve(file_type, processed_queries, k)
        search_result = [
            MinimalSearchResults.from_query_and_sources(query, sources)
            for query, sources in zip(queries, results)
        ]
        return StudentSearchResults(search_results=search_result, k=k)

    @abstractmethod
    def _load_and_retrieve(
        self, file_type: FileCategory, processed_queries: list[str], k: int
    ) -> list[list[Mapping[str, Any]]]: ...
