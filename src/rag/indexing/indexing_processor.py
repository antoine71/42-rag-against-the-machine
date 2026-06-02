from abc import ABC, abstractmethod

from rag.config.indexing_config import IndexingConfig
from rag.models.chunk import Chunk, FileType
from rag.text_processing.text_processing_factory import TextProcessingFactory
from rag.text_processing.text_processing_pipeline import TextProcessingPipeline
from rag.tui import TUI


class IndexingProcessor(ABC):
    """Abstract base class for all indexing processors."""

    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: IndexingConfig
    ) -> None:
        """Initializes the IndexingProcessor.

        Args:
            chunks: A list of Chunk models to be indexed.
            tui: A TUI instance to handle user interface / progress output.
            config: A configuration object containing indexing parameters.
        """
        self._chunks = chunks
        self._ui = tui
        self._config = config
        self._text_processing_factory = TextProcessingFactory(
            self._config.text_processing, self._ui
        )

    def _get_text_processing_manager(
        self, file_type: FileType
    ) -> TextProcessingPipeline:
        text_processors = self._text_processing_factory.create(file_type)
        return TextProcessingPipeline(text_processors)

    @abstractmethod
    def index_corpus(self, save_directory: str, file_type: FileType) -> None:
        """Abstract method to index the corpus and save it to the specified
        directory.

        Args:
            save_directory: The directory path where index data should be
                saved.
        """
        ...
