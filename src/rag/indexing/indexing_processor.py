from abc import ABC, abstractmethod

from pydantic_settings import BaseSettings

from rag.models.chunk import Chunk
from rag.tui import TUI


class IndexingProcessor(ABC):
    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: BaseSettings
    ) -> None:
        self._chunks = chunks
        self._ui = tui
        self._config = config

    @abstractmethod
    def index_corpus(self, save_directory: str) -> None: ...
