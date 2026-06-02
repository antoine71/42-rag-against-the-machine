import chromadb
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score, semantic_search

from rag.config.embedding import EmbeddingConfig
from rag.models.chunk import FileType
from rag.models.indexing_method import IndexingMethod
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
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
            index_directory: Path to ChromaDB database files.
            tui: A TUI instance to handle progress output.
        """
        super().__init__(index_directory, tui, config)
        self._config: EmbeddingConfig
        self._embedder = SentenceTransformer(self._config.model)

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int, file_type: FileType
    ) -> StudentSearchResults:
        """Performs batch semantic search on queries using ChromaDB HNSW.

        Args:
            queries: A list of queries to search.
            k: The number of top results to retrieve.

        Returns:
            A StudentSearchResults object.
        """
        store = chromadb.PersistentClient(
            FilesManager.get_indexing_directory(
                self._index_directory, IndexingMethod.VECTOR, file_type
            )
        )
        collection = store.get_collection(self._config.collection)
        corpus = collection.get(
            include=["embeddings", "metadatas", "documents"]
        )
        corpus_embeddings = torch.tensor(
            corpus["embeddings"], dtype=torch.float32
        )
        results = []
        with self._tui.progress(
            "Semantic search", len(queries), "query"
        ) as progress:
            for query in queries:
                query_embedding = self._embedder.encode_query(
                    query.question,
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

        search_result = [
            MinimalSearchResults.from_query_and_sources(query, sources)
            for query, sources in zip(queries, results)
        ]
        return StudentSearchResults(search_results=search_result, k=k)
