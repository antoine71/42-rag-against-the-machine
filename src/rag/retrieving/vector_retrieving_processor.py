import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag.models.minimal_source import MinimalSource
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor


class VectorRetrievingProcessor(RetrievingProcessor):
    COLLECTION = "rag_vllm_repository"

    def __init__(self) -> None:
        self._embedder = SentenceTransformer(
            "sentence-transformers/msmarco-bert-base-dot-v5"
        )
        self._store = chromadb.PersistentClient("data/processed")

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        collection = self._store.get_collection(self.COLLECTION)
        corpus = collection.get(include=["embeddings", "metadatas"])
        corpus_embeddings = np.array(corpus["embeddings"], dtype=np.float32)
        results = []
        for query in queries:
            query_embedding = self._embedder.encode_query(
                query.question, convert_to_numpy=True, precision="float32"
            )
            similarity_scores = self._embedder.similarity(
                query_embedding, corpus_embeddings
            )[0]
            scores, indices = torch.topk(similarity_scores, k=k)
            print(scores, indices)
            results.append([corpus["metadatas"][idx] for idx in indices])

        search_result = []
        for i, result in enumerate(results):
            search_result.append(
                MinimalSearchResults(
                    question_id=queries[i].question_id,
                    question=queries[i].question,
                    retrieved_sources=[
                        MinimalSource(
                            file_path=source["source"],
                            first_character_index=source["start_index"],
                            last_character_index=source["end_index"],
                        )
                        for source in result
                    ],
                )
            )
        return StudentSearchResults(search_results=search_result, k=k)
