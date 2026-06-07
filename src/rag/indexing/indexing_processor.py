from abc import ABC, abstractmethod

from rag.config.indexing_config import IndexingConfig
from rag.models.chunk import Chunk
from rag.models.file_category import FileCategory
from rag.models.file_type import FileType
from rag.text_processing.pipeline_factory import TextProcessingPipelineFactory
from rag.tui import TUI
from rag.utils.measure import measure


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
        self._tui.print_phase_title(f"{self._config.TYPE}")
        processed_chunks = self._run_text_processing(files_category)
        save_path = self._index_and_save(
            processed_chunks, save_directory, files_category
        )
        self._tui.print(f"\nOuput:\t{save_path}\n")

    def _run_text_processing(
        self, files_category: FileCategory
    ) -> list[Chunk]:
        """Runs text processing for all file types in a category.

        Args:
            files_category: File category being indexed.

        Returns:
            Processed chunks for the category.
        """
        processed_chunks: list[Chunk] = []
        for file_type in files_category.file_types:
            processed_chunks.extend(self._process_chunks(file_type))
        return processed_chunks

    def _process_chunks(self, file_type: FileType) -> list[Chunk]:
        """Runs the configured text processors for a file type.

        Args:
            file_type: Concrete file type to process.

        Returns:
            Processed chunks for the file type.
        """
        chunks = [chunk for chunk in self._chunks if file_type == chunk.type]
        chunks_processing_pipeline = TextProcessingPipelineFactory(
            self._config.text_processing, self._tui
        )
        text_processing_pipeline = chunks_processing_pipeline.create(file_type)
        processed_chunks, delta = measure(
            text_processing_pipeline.process_list, chunks
        )
        self._tui.print_task_report(
            f"Text Processing ({file_type} files)",
            delta,
            "chunks",
            len(chunks),
        )
        return processed_chunks

    @abstractmethod
    def _index_and_save(
        self,
        processed_chunks: list[Chunk],
        save_directory: str,
        file_type: FileCategory,
    ) -> str:
        """Indexes processed chunks and persists backend data.

        Args:
            processed_chunks: Chunks after text processing.
            save_directory: Root directory where index data is saved.
            file_type: File category represented by the index.

        Returns:
            Path where backend data was saved.
        """
        ...
