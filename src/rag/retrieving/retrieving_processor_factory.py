from typing import Any

from rag.retrieving.bm25_retrieving_processor import BM25RetrievingProcessor
from rag.retrieving.retrieving_processor import RetrievingProcessor
from rag.retrieving.vector_retrieving_processor import (
    VectorRetrievingProcessor,
)


class RetrievingProcessorFactory:
    _processors: dict[str, type[RetrievingProcessor]] = {
        "vector": VectorRetrievingProcessor,
        "bm25": BM25RetrievingProcessor,
    }

    @classmethod
    def create(
        cls,
        retrieving_methods: list[str],
        index_directory: str,
        config: dict[str, Any],
    ) -> list[RetrievingProcessor]:
        processors: list[RetrievingProcessor] = []
        for retrieving_method in retrieving_methods:
            if retrieving_method == "bm25":
                processors.append(
                    BM25RetrievingProcessor(index_directory, config)
                )
            elif retrieving_method == "vector":
                processors.append(VectorRetrievingProcessor())
        return processors
