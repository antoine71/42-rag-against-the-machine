from pathlib import Path

import bm25s

from rag.config.bm25 import BM25Configuration
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor


class BM25RetrievingProcessor(RetrievingProcessor):
    """Retrieving processor that uses a saved BM25 sparse index for keyword search retrieval."""

    WEIGHT = 1.3

    def __init__(self, index_directory: str) -> None:
        """Initializes the BM25RetrievingProcessor.

        Args:
            index_directory: The directory path where the BM25 index is saved.
        """
        self._index_directory = Path(index_directory)
        self._config = BM25Configuration()

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        """Performs BM25 keyword search on the queries to retrieve top-k sources.

        Args:
            queries: A list of UnansweredQuestion objects containing search queries.
            k: The number of top results to retrieve.

        Returns:
            A StudentSearchResults object.
        """
        query_tokens = bm25s.tokenize([query.question for query in queries])
        retriever: bm25s.BM25 = bm25s.BM25().load(
            self._index_directory,
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
