from rag.indexing.indexing_processor import IndexingProcessor


class IndexingManager:
    def __init__(self) -> None:
        self._indexing_processors: list[tuple[IndexingProcessor, str]] = []

    def add_indexing_processor(
        self, indexing_processor: IndexingProcessor, save_directory: str
    ) -> None:
        self._indexing_processors.append((indexing_processor, save_directory))

    def process(self):
        for processor, save_directory in self._indexing_processors:
            processor.index_corpus(save_directory)
