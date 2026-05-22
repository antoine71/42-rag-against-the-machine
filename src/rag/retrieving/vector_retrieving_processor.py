import chromadb
import torch
from sentence_transformers import SentenceTransformer

from rag.models.question import UnansweredQuestion
from rag.models.search_result import StudentSearchResults
from rag.retrieving.retrieving_processor import RetrievingProcessor


class VectorRetrievingProcessor(RetrievingProcessor):
    COLLECTION = "rag_vllm_repository"

    def __init__(self) -> None:
        self._embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._store = chromadb.PersistentClient("data/processed")

    def retrieve(
        self, queries: list[UnansweredQuestion], k: int
    ) -> StudentSearchResults:
        print(self._store.list_collections())
        collection = self._store.get_collection(self.COLLECTION)
        corpus = collection.get(self.COLLECTION)
        print(corpus["embeddings"])
        print(len(corpus["metadatas"]))
        print(corpus.keys())
        for query in queries:
            query_embedding = self._embedder.encode_query(query.question)
            similarity_scores = self._embedder.similarity(
                query_embedding, corpus["embeddings"]
            )[0]
            scores, indices = torch.topk(similarity_scores, k=k)

            print("\nQuery:", query)
            print("Top k most similar sentences in corpus:")

            for score, idx in zip(scores, indices):
                print(f"(Score: {score:.4f})", corpus["metadatas"][idx])

        # query_tokens = bm25s.tokenize([query.question for query in queries])
        # results, _ = self._retriever.retrieve(
        #     query_tokens, show_progress=True, leave_progress=True
        # )
        # search_result = []
        # for i, result in enumerate(results):
        #     search_result.append(
        #         MinimalSearchResults(
        #             question_id=queries[i].question_id,
        #             question=queries[i].question,
        #             retrieved_sources=[
        #                 MinimalSource(
        #                     file_path=source["source"],
        #                     first_character_index=source["start_index"],
        #                     last_character_index=source["end_index"],
        #                 )
        #                 for source in result
        #             ],
        #         )
        #     )
        # return StudentSearchResults(search_results=search_result, k=k)
