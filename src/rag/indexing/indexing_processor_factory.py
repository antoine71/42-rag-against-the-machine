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
    """Factory class to create indexing processor instances."""

    @classmethod
    def create(
        cls,
        indexing_method: str,
        chunks: list[Chunk],
        tui: TUI,
    ) -> list[IndexingProcessor]:
        """Creates a list of IndexingProcessor instances.

        Args:
            indexing_method: The indexing strategy ('bm25', 'vector', 'hybrid').
            chunks: A list of Chunk models to index.
            tui: A TUI instance to handle progress output.

        Returns:
            A list of IndexingProcessor instances.

        Raises:
            RAGException: If an invalid indexing method is specified.
        """
        def bm25_factory() -> BM25IndexingProcessor:
            return BM25IndexingProcessor(chunks, tui, BM25Configuration())

        def vector_factory() -> VectorEmbeddingProcessor:
            return VectorEmbeddingProcessor(chunks, tui, EmbeddingConfig())

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
