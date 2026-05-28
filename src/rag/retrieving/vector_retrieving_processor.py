import chromadb
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score, semantic_search

from rag.config.embedding import EmbeddingConfig
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor
from rag.tui import TUI


class VectorRetrievingProcessor(RetrievingProcessor):
    WEIGHT = 1.0

    def __init__(self, index_directory: str, tui: TUI) -> None:
        self._config = EmbeddingConfig()
        self._embedder = SentenceTransformer(self._config.model)
        self._store = chromadb.PersistentClient(index_directory)
        self._tui = tui

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        collection = self._store.get_collection(self._config.collection)
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
            for i, query in enumerate(queries):
                query_embedding = self._embedder.encode_query(
                    query.question,
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
