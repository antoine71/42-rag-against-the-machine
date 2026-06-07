from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.file_category import FileCategory


class IndexingPipeline:
    """Manager class that coordinates multiple indexing processors to
    index the corpus."""

    def __init__(self, indexing_processors: list[IndexingProcessor]) -> None:
        """Initializes the indexing pipeline.

        Args:
            indexing_processors: A list of IndexingProcessor instances to
                execute.
        """
        self._indexing_processors = indexing_processors

    def process(
        self, save_directory: str, files_category: FileCategory
    ) -> None:
        """Executes indexing on all registered indexing processors.

        Args:
            save_directory: The directory path where index data should be
                saved.
            files_category: File category represented by the generated index.
        """
        for processor in self._indexing_processors:
            processor.index_corpus(save_directory, files_category)
