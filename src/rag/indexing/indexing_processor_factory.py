from typing import Any

from rag.exceptions import RAGException
from rag.indexing.bm25_repository_indexing_processor import (
    BM25IndexingProcessor,
    BM25MultiIndexingProcessor,
)
from rag.indexing.indexing_processor import IndexingProcessor
from rag.indexing.vector_embedding_processor import VectorEmbeddingProcessor
from rag.models.chunk import Chunk
from rag.tui import TUI


class IndexingProcessorFactory:
    _processors: dict[str, type[IndexingProcessor]] = {
        "vector": VectorEmbeddingProcessor,
        "bm25": BM25IndexingProcessor,
        "bm25_multi": BM25MultiIndexingProcessor,
    }

    @classmethod
    def create(
        cls,
        indexing_method: str,
        chunks: list[Chunk],
        tui: TUI,
        config: dict[str, Any],
    ) -> IndexingProcessor:
        try:
            processor_cls = cls._processors[indexing_method]
        except KeyError:
            raise RAGException(f"Invalid indexing method: {indexing_method}")
        return processor_cls(chunks, tui, config)
