from rag.indexing.indexing_processor import IndexingProcessor


class IndexingManager:
    def __init__(self, indexing_processors: list[IndexingProcessor]) -> None:
        self._indexing_processors = indexing_processors

    def process(self, save_directory: str) -> None:
        for processor in self._indexing_processors:
            processor.index_corpus(save_directory)
