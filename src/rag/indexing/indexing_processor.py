from abc import ABC, abstractmethod

from pydantic_settings import BaseSettings

from rag.models.chunk import Chunk
from rag.tui import TUI


class IndexingProcessor(ABC):
    """Abstract base class for all indexing processors."""

    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: BaseSettings
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
    def index_corpus(self, save_directory: str) -> None:
        """Abstract method to index the corpus and save it to the specified directory.

        Args:
            save_directory: The directory path where index data should be saved.
        """
        ...
