from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import FileType


class IndexingPipeline:
    """Manager class that coordinates multiple indexing processors to
    index the corpus."""

    def __init__(self, indexing_processors: list[IndexingProcessor]) -> None:
        """Initializes the IndexingManager.

        Args:
            indexing_processors: A list of IndexingProcessor instances to
                execute.
        """
        self._indexing_processors = indexing_processors

    def process(self, save_directory: str, file_type: str | None) -> None:
        """Executes indexing on all registered indexing processors.

        Args:
            save_directory: The directory path where index data should be
                saved.
        """
        file_types = (file_type,) if file_type is not None else FileType
        for processor in self._indexing_processors:
            for t in file_types:
                processor.index_corpus(save_directory, FileType(t))
