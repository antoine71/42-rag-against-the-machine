from rag.config.bm25_config import BM25Configuration
from rag.config.embedding import EmbeddingConfig
from rag.indexing.bm25_indexing_processor import (
    BM25IndexingProcessor,
)
from rag.indexing.indexing_processor import IndexingProcessor
from rag.indexing.vector_embedding_processor import VectorEmbeddingProcessor
from rag.models.chunk import Chunk
from rag.models.indexing_method import IndexingMethod
from rag.tui import TUI


class IndexingProcessorFactory:
    """Factory class to create indexing processor instances."""

    @classmethod
    def create(
        cls,
        indexing_method: IndexingMethod,
        chunks: list[Chunk],
        tui: TUI,
    ) -> list[IndexingProcessor]:
        """Creates a list of IndexingProcessor instances.

        Args:
            indexing_method: The indexing strategy ('bm25', 'vector',
                'hybrid').
            chunks: A list of Chunk models to index.
            tui: A TUI instance to handle progress output.

        Returns:
            A list of IndexingProcessor instances.

        Raises:
            RAGException: If an invalid indexing method is specified.
        """

        def bm25_factory() -> BM25IndexingProcessor:
            """Creates a BM25 indexing processor instance."""
            return BM25IndexingProcessor(chunks, tui, BM25Configuration())

        def vector_factory() -> VectorEmbeddingProcessor:
            """Creates a vector embedding indexing processor instance."""
            return VectorEmbeddingProcessor(chunks, tui, EmbeddingConfig())

        match indexing_method:
            case IndexingMethod.BM25:
                return [bm25_factory()]
            case IndexingMethod.VECTOR:
                return [vector_factory()]
            case IndexingMethod.HYBRID:
                return [bm25_factory(), vector_factory()]
