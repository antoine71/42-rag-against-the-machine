from abc import ABC, abstractmethod

from rag.config.retrieving_config import RetrievingConfig
from rag.models.chunk import FileType
from rag.models.question import UnansweredQuestion
from rag.models.search_result import StudentSearchResults
from rag.text_processing.text_processing_factory import TextProcessingFactory
from rag.text_processing.text_processing_pipeline import TextProcessingPipeline
from rag.tui import TUI


class RetrievingProcessor(ABC):
    """Protocol defining the interface for all retrieving processors."""

    WEIGHT: float

    def __init__(
        self, index_directory: str, tui: TUI, config: RetrievingConfig
    ) -> None:
        """Initializes the VectorRetrievingProcessor.

        Args:
            index_directory: Path to ChromaDB database files.
            tui: A TUI instance to handle progress output.
        """
        self._index_directory = index_directory
        self._tui = tui
        self._config = config
        self._query_processing_factory = TextProcessingFactory(
            self._config.query_processing, self._tui
        )

    def _get_query_processing_manager(
        self, file_type: FileType
    ) -> TextProcessingPipeline:
        text_processors = self._query_processing_factory.create(file_type)
        return TextProcessingPipeline(text_processors)

    @abstractmethod
    def retrieve(
        self, queries: list[UnansweredQuestion], k: int, file_type: FileType
    ) -> StudentSearchResults:
        """Retrieves top-k most relevant sources for the given queries.

        Args:
            queries: A list of UnansweredQuestion objects containing the
                search queries.
            k: The number of top results to retrieve.

        Returns:
            A StudentSearchResults object containing retrieved sources for each
                query.
        """
        ...
