from abc import ABC, abstractmethod

from rag.config.indexing_config import IndexingConfig
from rag.models.chunk import Chunk, FileType
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

    @abstractmethod
    def index_corpus(self, save_directory: str, file_type: FileType) -> None:
        """Abstract method to index the corpus and save it to the specified
        directory.

        Args:
            save_directory: The directory path where index data should be
                saved.
        """
        ...
