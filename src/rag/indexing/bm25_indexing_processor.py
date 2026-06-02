import logging

import bm25s

from rag.config.bm25_config import BM25Configuration
from rag.config.indexing_config import IndexingConfig
from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk, FileType
from rag.models.indexing_method import IndexingMethod
from rag.tui import TUI
from rag.utils.files_manager import FilesManager

logger = logging.getLogger(__name__)


class BM25IndexingProcessor(IndexingProcessor):
    """An indexing processor that creates and saves a BM25 sparse index."""

    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: IndexingConfig
    ) -> None:
        """Initializes the BM25IndexingProcessor.

        Args:
            chunks: A list of Chunk models to be indexed.
            tui: A TUI instance to handle user interface / progress output.
            config: A configuration object containing BM25 parameters.
        """
        super().__init__(chunks, tui, config)
        self._config: BM25Configuration

    def index_corpus(self, save_directory: str, file_type: FileType) -> None:
        """Indexes the text chunks using BM25 and saves the index to disk.

        Args:
            save_directory: The directory path where the BM25 index files
                should be saved.
        """
        self._ui.print(f"Indexing {file_type}...")
        chunks = [chunk for chunk in self._chunks if chunk.type == file_type]
        texts = [chunk.text for chunk in chunks]
        texts_processing_manager = self._get_text_processing_manager(file_type)
        processed_texts = texts_processing_manager.process_list(texts)
        for chunk, processed_text in zip(chunks, processed_texts):
            chunk.text = processed_text
        corpus = bm25s.tokenize(texts)
        retriever = bm25s.BM25(**self._config.bm25_settings.model_dump())
        retriever.index(corpus, show_progress=True, leave_progress=True)
        save_path = FilesManager.get_indexing_directory(
            save_directory, IndexingMethod.BM25, file_type
        )
        retriever.save(
            save_path,
            corpus=[c.model_dump() for c in chunks],
            show_progress=False,
        )
        self._ui.print(f"Ingestion complete! Indices saved under {save_path}")
