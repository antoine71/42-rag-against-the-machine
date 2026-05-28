from collections.abc import Callable

from rag.exceptions import RAGException
from rag.retrieving.bm25_retrieving_processor import BM25RetrievingProcessor
from rag.retrieving.retrieving_processor import RetrievingProcessor
from rag.retrieving.vector_retrieving_processor import (
    VectorRetrievingProcessor,
)
from rag.tui import TUI


class RetrievingProcessorFactory:
    @classmethod
    def create(
        cls, retrieving_method: str, index_directory: str, tui: TUI
    ) -> list[RetrievingProcessor]:
        bm25_factory: Callable[[], BM25RetrievingProcessor] = lambda: (
            BM25RetrievingProcessor(index_directory)
        )
        vector_factory: Callable[[], VectorRetrievingProcessor] = lambda: (
            VectorRetrievingProcessor(index_directory, tui)
        )
        match retrieving_method:
            case "bm25":
                return [bm25_factory()]
            case "vector":
                return [vector_factory()]
            case "hybrid":
                return [bm25_factory(), vector_factory()]
            case _:
                raise RAGException(
                    f"Invalid retrieving method '{retrieving_method}'"
                )
