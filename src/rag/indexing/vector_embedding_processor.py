import chromadb
from sentence_transformers import SentenceTransformer

from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk


class VectorEmbeddingProcessor(IndexingProcessor):
    COLLECTION = "rag_vllm_repository"

    def __init__(self, chunks: list[Chunk]) -> None:
        super().__init__(chunks)
        self._embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
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
        print(corpus_embeddings.tolist())
        collection.add(
            ids=[f"id{i}" for i in range(len(corpus_embeddings))],
            embeddings=corpus_embeddings.tolist(),
            documents=self._texts,
            metadatas=self._metadatas,
        )
        c = collection.get(self.COLLECTION)
        print(c["embeddings"])
