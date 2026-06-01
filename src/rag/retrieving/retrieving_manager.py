from rag.config.rrf import RRFConfig
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor
from rag.utils.reciprocal_rank_fusion import reciprocal_rank_fusion


class RetrievingManager:
    """Manager class that coordinates retrieval from single or multiple
    retrieving processors.

    Uses Reciprocal Rank Fusion (RRF) when combining multiple retrieving
    processors.
    """

    def __init__(
        self, retrieving_processors: list[RetrievingProcessor]
    ) -> None:
        """Initializes the RetrievingManager.

        Args:
            retrieving_processors: A list of RetrievingProcessor instances to
                use.
        """
        self._retrieving_processors = retrieving_processors
        self._rrf_config = RRFConfig()

    def process(
        self,
        queries: list[UnansweredQuestion],
        k: int,
    ) -> StudentSearchResults:
        """Processes queries using the configured retrieving processors.

        Args:
            queries: List of UnansweredQuestion objects.
            k: The number of top results to retrieve.

        Returns:
            A StudentSearchResults containing combined, ranked sources.
        """
        if len(self._retrieving_processors) == 1:
            return self._simple_retrieving(queries, k)
        return self._multiple_retrieving(queries, k)

    def _multiple_retrieving(
        self,
        queries: list[UnansweredQuestion],
        k: int,
    ) -> StudentSearchResults:
        """Retrieves and reranks results from multiple retrieving processors
        using RRF.

        Args:
            queries: List of UnansweredQuestion objects.
            k: The number of top results to retrieve.

        Returns:
            A reranked StudentSearchResults object.
        """
        search_results = [
            processor.retrieve(queries, k * self._rrf_config.k_factor)
            for processor in self._retrieving_processors
        ]
        reranked_result = self._rerank_results(
            queries,
            search_results,
            [p.WEIGHT for p in self._retrieving_processors],
            k,
        )
        return reranked_result

    def _simple_retrieving(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        """Retrieves results using the single configured retrieving processor.

        Args:
            queries: List of UnansweredQuestion objects.
            k: The number of top results to retrieve.

        Returns:
            A StudentSearchResults object.
        """
        search_results = self._retrieving_processors[0].retrieve(queries, k)
        return search_results

    def _rerank_results(
        self,
        queries: list[UnansweredQuestion],
        search_results: list[StudentSearchResults],
        weights: list[float],
        k: int,
    ) -> StudentSearchResults:
        """Fuses and reranks multiple search result sets using RRF.

        Args:
            queries: List of UnansweredQuestion objects.
            search_results: A list of StudentSearchResults, one set per
                retrieving processor.
            weights: A list of weight floats associated with each retrieving
                processor.
            k: The number of top results to keep after reranking.

        Returns:
            A combined and reranked StudentSearchResults object.
        """
        reranked_search_results: list[MinimalSearchResults] = []
        for query in queries:
            sources_ranks = [
                next(
                    r.retrieved_sources
                    for r in search_result.search_results
                    if r.question_id == query.question_id
                )
                for search_result in search_results
            ]
            reranked_sources = reciprocal_rank_fusion(
                sources_ranks, weights=weights
            )
            reranked_search_results.append(
                MinimalSearchResults(
                    question_id=query.question_id,
                    question=query.question,
                    retrieved_sources=reranked_sources[:k],
                )
            )
        return StudentSearchResults(
            search_results=reranked_search_results, k=k
        )
