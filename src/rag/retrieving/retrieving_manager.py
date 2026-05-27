from collections.abc import Callable

from rag.models.minimal_source import MinimalSource
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor


class RetrievingManager:
    def __init__(self) -> None:
        self._retrieving_processors: list[RetrievingProcessor] = []

    def add_retrieving_processor(
        self, retrieving_processor: RetrievingProcessor
    ) -> None:
        self._retrieving_processors.append(retrieving_processor)

    def process(
        self,
        queries: list[UnansweredQuestion],
        k: int,
        k_factor: int,
        merge_fct: Callable[[list[list[str]]], list[str]],
    ):
        search_results: list[StudentSearchResults] = []
        for processor in self._retrieving_processors:
            search_results.append(processor.retrieve(queries, k * k_factor))
        reranked_result = self._rerank_results(
            queries,
            search_results,
            [p.WEIGHT for p in self._retrieving_processors],
            merge_fct,
            k,
        )
        return reranked_result

    def _rerank_results(
        self,
        queries: list[UnansweredQuestion],
        search_results: list[StudentSearchResults],
        weights: list[float],
        merge_fct: Callable[[list[list[MinimalSource]]], list[MinimalSource]],
        k: int,
    ) -> StudentSearchResults:
        result = StudentSearchResults(search_results=[], k=k)
        for query in queries:
            sources_ranks = [
                next(
                    r.retrieved_sources
                    for r in search_result.search_results
                    if r.question_id == query.question_id
                )
                for search_result in search_results
            ]
            reranked_sources = merge_fct(sources_ranks, weights=weights)
            result.search_results.append(
                MinimalSearchResults(
                    question_id=query.question_id,
                    question=query.question,
                    retrieved_sources=reranked_sources[:k],
                )
            )
        return result
