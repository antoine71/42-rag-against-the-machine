from abc import ABC, abstractmethod

from rag.config.indexing_config import IndexingConfig
from rag.models.chunk import Chunk
from rag.models.file_category import FileCategory
from rag.models.file_type import FileType
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

    def index_corpus(
        self, save_directory: str, files_category: FileCategory
    ) -> None:
        """Abstract method to index the corpus and save it to the specified
        directory.

        Args:
            save_directory: The directory path where index data should be
                saved.
        """
        self._tui.print(
            f"Indexing {files_category} files with {self._config.TYPE}..."
        )
        processed_chunks: list[Chunk] = []
        for file_type in files_category.file_types:
            processed_chunks.extend(self._process_chunks(file_type))
        save_path = self._index_and_save(
            processed_chunks, save_directory, files_category
        )
        self._tui.print(f"Ingestion complete! Indices saved under {save_path}")

    def _process_chunks(self, file_type: FileType) -> list[Chunk]:
        chunks = [chunk for chunk in self._chunks if file_type == chunk.type]
        chunks_processing_pipeline = TextProcessingPipelineFactory(
            self._config.text_processing, self._tui
        )
        text_processing_pipeline = chunks_processing_pipeline.create(file_type)
        stages = " > ".join(
            stage.__class__.__name__
            for stage in text_processing_pipeline.stages
        )
        self._tui.print(
            f"Running Text Processing Pipeline for {file_type} chunks: {stages}"
        )
        processed_chunks = text_processing_pipeline.process_list(chunks)
        return processed_chunks

    @abstractmethod
    def _index_and_save(
        self,
        processed_chunks: list[Chunk],
        save_directory: str,
        file_type: FileCategory,
    ) -> str: ...
