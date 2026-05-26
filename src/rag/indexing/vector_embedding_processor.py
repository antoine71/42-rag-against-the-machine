import itertools

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk
from rag.models.minimal_source import MinimalSource
from rag.models.question import UnansweredQuestion
from rag.models.search_result import MinimalSearchResults, StudentSearchResults


class VectorEmbeddingProcessor(IndexingProcessor):
    COLLECTION = "rag_vllm_repository"

    def __init__(self, chunks: list[Chunk]) -> None:
        super().__init__(chunks)
        self._embedder = SentenceTransformer(
            "sentence-transformers/msmarco-bert-base-dot-v5"
        )

    def index_corpus(self, save_directory: str) -> None:
        corpus_embeddings = self._embedder.encode_document(
            self._texts, show_progress_bar=True
        )

        chroma_client = chromadb.PersistentClient(save_directory)
        if self.COLLECTION in [
            c.name for c in chroma_client.list_collections()
        ]:
            chroma_client.delete_collection(self.COLLECTION)
        collection = chroma_client.create_collection(
            name="rag_vllm_repository"
        )
        ids = [f"id{i}" for i in range(len(corpus_embeddings))]

        BATCH_SIZE = 300
        batched_ids = itertools.batched(ids, BATCH_SIZE)
        batched_embeddings = itertools.batched(corpus_embeddings, BATCH_SIZE)
        batched_documents = itertools.batched(self._texts, BATCH_SIZE)
        batched_metadatas = itertools.batched(self._metadatas, BATCH_SIZE)
        while True:
            try:
                collection.add(
                    ids=list(next(batched_ids)),
                    embeddings=list(next(batched_embeddings)),
                    documents=list(next(batched_documents)),
                    metadatas=list(next(batched_metadatas)),
                )
            except StopIteration:
                break

    def special_index(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        corpus_embeddings = self._embedder.encode_document(
            self._texts, show_progress_bar=True
        )
        results = []
        for query in queries:
            query_embedding = self._embedder.encode_query(query.question)
            similarity_scores = self._embedder.similarity(
                query_embedding, corpus_embeddings
            )[0]
            scores, indices = torch.topk(similarity_scores, k=k)
            print(scores, indices)
            results.append([self._metadatas[idx] for idx in indices])

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
