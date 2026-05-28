from collections.abc import Callable

from rag.config.bm25 import BM25Configuration
from rag.config.embedding import EmbeddingConfig
from rag.exceptions import RAGException
from rag.indexing.bm25_indexing_processor import (
    BM25IndexingProcessor,
)
from rag.indexing.indexing_processor import IndexingProcessor
from rag.indexing.vector_embedding_processor import VectorEmbeddingProcessor
from rag.models.chunk import Chunk
from rag.tui import TUI


class IndexingProcessorFactory:
    @classmethod
    def create(
        cls,
        indexing_method: str,
        chunks: list[Chunk],
        tui: TUI,
    ) -> list[IndexingProcessor]:
        bm25_factory: Callable[[], BM25IndexingProcessor] = lambda: (
            BM25IndexingProcessor(chunks, tui, BM25Configuration())
        )
        vector_factory: Callable[[], VectorEmbeddingProcessor] = lambda: (
            VectorEmbeddingProcessor(chunks, tui, EmbeddingConfig())
        )
        match indexing_method:
            case "bm25":
                return [bm25_factory()]
            case "vector":
                return [vector_factory()]
            case "hybrid":
                return [bm25_factory(), vector_factory()]
            case _:
                raise RAGException(
                    f"Invalid indexing method: {indexing_method}"
                )
