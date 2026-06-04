import bm25s

from rag.config.bm25_config import BM25Configuration
from rag.models.chunk import FileType
from rag.models.indexing_method import IndexingMethod
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor
from rag.text_processing.pipeline_factory import ProcessingPipelineFactory
from rag.tui import TUI
from rag.utils.files_manager import FilesManager


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

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int, file_type: FileType
    ) -> StudentSearchResults:
        """Perform BM25 keyword search and retrieve top-k sources.

        Args:
            queries: A list of UnansweredQuestion objects containing search
                queries.
            k: The number of top results to retrieve.

        Returns:
            A StudentSearchResults object.
        """
        chunks_index = FilesManager.get_indexing_directory(
            self._index_directory, IndexingMethod.BM25, file_type
        )
        query_processing_pipeline_factory = ProcessingPipelineFactory(
            self._config.query_processing, self._tui
        )
        query_processing_pipeline = query_processing_pipeline_factory.create(
            file_type
        )
        processed_queries = query_processing_pipeline.process_list(
            [query.question for query in queries]
        )

        query_tokens = bm25s.tokenize(processed_queries)
        retriever: bm25s.BM25 = bm25s.BM25().load(
            chunks_index,
            load_corpus=True,
            **self._config.bm25_settings.model_dump(),
        )
        results, _ = retriever.retrieve(
            query_tokens,
            show_progress=True,
            leave_progress=True,
        )
        search_result = [
            MinimalSearchResults.from_query_and_sources(query, sources)
            for query, sources in zip(queries, results)
        ]
        return StudentSearchResults(search_results=search_result, k=k)
