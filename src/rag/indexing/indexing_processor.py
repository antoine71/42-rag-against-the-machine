from abc import ABC, abstractmethod

from rag.models.chunk import Chunk


class IndexingProcessor(ABC):
    def __init__(
        self,
        chunks: list[Chunk],
    ) -> None:
        self._chunks = chunks
        self._corpus = [chunk.model_dump() for chunk in chunks]
        self._texts = [chunk.text for chunk in chunks]
        self._metadatas = [chunk.metadata for chunk in chunks]

    @abstractmethod
    def index_corpus(self, save_directory: str) -> None: ...
