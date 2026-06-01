from rag.indexing.indexing_processor import IndexingProcessor


class IndexingManager:
    """Manager class that coordinates multiple indexing processors to index the corpus."""

    def __init__(self, indexing_processors: list[IndexingProcessor]) -> None:
        """Initializes the IndexingManager.

        Args:
            indexing_processors: A list of IndexingProcessor instances to execute.
        """
        self._indexing_processors = indexing_processors

    def process(self, save_directory: str) -> None:
        """Executes indexing on all registered indexing processors.

        Args:
            save_directory: The directory path where index data should be saved.
        """
        for processor in self._indexing_processors:
            processor.index_corpus(save_directory)
