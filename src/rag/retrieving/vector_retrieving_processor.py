import time
from pathlib import Path
from typing import Any, Mapping

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score, semantic_search

from rag.config.embedding import EmbeddingConfig
from rag.exceptions import RAGException
from rag.models.file_category import FileCategory
from rag.models.indexing_method import IndexingMethod
from rag.retrieving.retrieving_processor import RetrievingProcessor
from rag.tui import TUI
from rag.utils.files_manager import FilesManager


class VectorRetrievingProcessor(RetrievingProcessor):
    """Retrieving processor using a vector database for semantic search."""

    WEIGHT = 1.0

    def __init__(
        self, index_directory: str, tui: TUI, config: EmbeddingConfig
    ) -> None:
        """Initializes the VectorRetrievingProcessor.

        Args:
            index_directory: Root directory containing persisted indexes.
            tui: A TUI instance to handle progress output.
            config: Embedding configuration used for query encoding.
        """
        super().__init__(index_directory, tui, config)
        self._config: EmbeddingConfig
        self._embedder = SentenceTransformer(self._config.model)

    def _load_and_retrieve(
        self, file_type: FileCategory, processed_queries: list[str], k: int
    ) -> list[list[Mapping[str, Any]]]:
        """Loads vector embeddings and retrieves source metadata.

        Args:
            file_type: File category to retrieve from.
            processed_queries: Query strings to encode.
            k: Number of sources to retrieve per query.

        Returns:
            Retrieved source metadata for each query.
        """
        chunks_index = FilesManager.get_indexing_directory(
            self._index_directory, IndexingMethod.VECTOR, file_type
        )
        if not Path(chunks_index).is_dir():
            raise RAGException(f"Dataset not found: '{chunks_index}'")
        store = chromadb.PersistentClient(chunks_index)
        collection = store.get_collection(self._config.collection)
        corpus = collection.get(
            include=["embeddings", "metadatas", "documents"]
        )
        corpus_embeddings = torch.tensor(
            corpus["embeddings"], dtype=torch.float32
        )
        results = []
        start = time.perf_counter()
        with self._tui.progress(
            "Semantic search", len(processed_queries), "query"
        ) as progress:
            for query in processed_queries:
                query_embedding = self._embedder.encode_query(
                    query,
                    show_progress_bar=False,
                )
                similarity_scores = semantic_search(
                    query_embedding,
                    corpus_embeddings,
                    top_k=k,
                    score_function=dot_score,
                )[0]
                indices = [int(s["corpus_id"]) for s in similarity_scores]
                metadatas = corpus["metadatas"]
                assert metadatas is not None
                results.append([metadatas[idx] for idx in indices])
                progress.update(1)
        end = time.perf_counter()
        delta = int((end - start) * 1000)
        self._tui.print_task_report(
            f"{self._config.TYPE} retrieving",
            delta,
            "questions",
            len(processed_queries),
            new_line_after=True,
        )
        return results
