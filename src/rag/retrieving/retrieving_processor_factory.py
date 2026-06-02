from rag.config.bm25_config import BM25Configuration
from rag.config.embedding import EmbeddingConfig
from rag.models.indexing_method import IndexingMethod
from rag.retrieving.bm25_retrieving_processor import BM25RetrievingProcessor
from rag.retrieving.retrieving_processor import RetrievingProcessor
from rag.retrieving.vector_retrieving_processor import (
    VectorRetrievingProcessor,
)
from rag.tui import TUI


class RetrievingProcessorFactory:
    """Factory class to create retrieving processor instances."""

    @classmethod
    def create(
        cls,
        indexing_method: IndexingMethod,
        index_directory: str,
        tui: TUI,
    ) -> list[RetrievingProcessor]:
        """Creates RetrievingProcessor instances based on method name.

        Args:
            retrieving_method: The retrieval strategy to use
                ('bm25', 'vector', or 'hybrid').
            index_directory: Path where index and DB files are saved.
            tui: A TUI instance to handle progress output.

        Returns:
            A list of RetrievingProcessor instances.

        Raises:
            RAGException: If an invalid retrieving method is specified.
        """

        def bm25_factory() -> BM25RetrievingProcessor:
            """Creates a BM25 retrieving processor instance."""
            return BM25RetrievingProcessor(
                index_directory, tui, BM25Configuration()
            )

        def vector_factory() -> VectorRetrievingProcessor:
            """Creates a vector retrieving processor instance."""
            return VectorRetrievingProcessor(
                index_directory, tui, EmbeddingConfig()
            )

        match indexing_method:
            case IndexingMethod.BM25:
                return [bm25_factory()]
            case IndexingMethod.VECTOR:
                return [vector_factory()]
            case IndexingMethod.HYBRID:
                return [bm25_factory(), vector_factory()]
