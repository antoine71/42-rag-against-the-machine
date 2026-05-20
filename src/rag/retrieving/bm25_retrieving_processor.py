import bm25s

from rag.models.minimal_source import MinimalSource
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults


class BM25RetrievingProcessor:
    def __init__(
        self,
    ) -> None:
        self._retriever = bm25s.BM25().load("data/processed", load_corpus=True)

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        # query_tokens = self._tokenize_batch(
        #     [query.question for query in queries]
        # )
        query_tokens = bm25s.tokenize([query.question for query in queries])
        # results, _ = self._retriever.retrieve(
        #     Tokenized(query_tokens, self._vocab), k=k
        # )
        results, _ = self._retriever.retrieve(query_tokens)
        search_result = []
        for i, result in enumerate(results):
            search_result.append(
                MinimalSearchResults(
                    question_id=queries[i].question_id,
                    question=queries[i].question,
                    retrieved_sources=[
                        MinimalSource(
                            file_path=source["source"],
                            first_character_index=source["start_index"],
                            last_character_index=source["end_index"],
                        )
                        for source in result
                    ],
                )
            )
        return StudentSearchResults(search_results=search_result, k=k)
