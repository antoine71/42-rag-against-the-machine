from abc import ABC, abstractmethod
from typing import Any

from rag.models.chunk import Chunk
from rag.tui import TUI


class IndexingProcessor(ABC):
    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: dict[str, Any]
    ) -> None:
        self._chunks = chunks
        self._corpus = [chunk.model_dump() for chunk in chunks]
        self._texts = [chunk.text for chunk in chunks]
        self._metadatas = [chunk.metadata for chunk in chunks]
        self._ui = tui
        self._config = config

    @abstractmethod
    def index_corpus(self, save_directory: str) -> None: ...
