from pathlib import Path
from typing import Any

import bm25s

from rag.indexing.bm25_configuration import BM25Configuration
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor


class BM25RetrievingProcessor(RetrievingProcessor):
    WEIGHT = 1.3

    def __init__(self, index_directory: str, config: dict[str, Any]) -> None:
        self._index_directory = Path(index_directory)
        self._config = config
        if not config:
            self._config = BM25Configuration.default_config

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        query_tokens = bm25s.tokenize([query.question for query in queries])
        retriever: bm25s.BM25 = bm25s.BM25().load(
            self._index_directory, load_corpus=True, **self._config
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


#
# class BM25MultiRetrievingProcessor(RetrievingProcessor):
#
#     def retrieve_by_type(self, queries: list[UnansweredQuestion], k: int, file_type: FileType):
#         retriever = bm25s.BM25().load(
#             self._index_directory / file_type, load_corpus=True
#         )
#
#         if BM25MultiIndexingProcessor.stemmer_map[file_type]:
#             stemmer = Stemmer.Stemmer("english")
#             query_tokens = bm25s.tokenize([query.question for query in queries], stemmer=stemmer)
#         else:
#             query_tokens = bm25s.tokenize([query.question for query in queries])
#         results, scores = retriever.retrieve(
#             query_tokens, show_progress=True, leave_progress=True
#         )
#         return results, scores
#
#
#
#
#
#
#     def retrieve(self, qaueries: list[UnansweredQuestion], k: int) -> StudentSearchResults:
#
#
