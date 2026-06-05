from abc import ABC, abstractmethod

from rag.config.indexing_config import IndexingConfig
from rag.models.chunk import Chunk, FileType
from rag.text_processing.pipeline_factory import TextProcessingPipelineFactory
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
        self._tui = tui
        self._config = config

    def index_corpus(self, save_directory: str, file_type: FileType) -> None:
        """Abstract method to index the corpus and save it to the specified
        directory.

        Args:
            save_directory: The directory path where index data should be
                saved.
        """
        self._tui.print(f"Indexing {file_type} with {self._config.TYPE}...")
        processed_chunks = self._process_chunks(file_type)
        save_path = self._index_and_save(
            processed_chunks, save_directory, file_type
        )
        self._tui.print(f"Ingestion complete! Indices saved under {save_path}")

    def _process_chunks(self, file_type: FileType) -> list[Chunk]:
        chunks = [chunk for chunk in self._chunks if chunk.type == file_type]
        chunks_processing_pipeline = TextProcessingPipelineFactory(
            self._config.text_processing, self._tui
        )
        text_processing_pipeline = chunks_processing_pipeline.create(file_type)
        processed_chunks = text_processing_pipeline.process_list(chunks)
        return processed_chunks

    @abstractmethod
    def _index_and_save(
        self,
        processed_chunks: list[Chunk],
        save_directory: str,
        file_type: FileType,
    ) -> str: ...
