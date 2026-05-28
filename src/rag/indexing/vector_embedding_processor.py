import chromadb
from more_itertools import batched
from sentence_transformers import SentenceTransformer

from rag.config.embedding import EmbeddingConfig
from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk
from rag.tui import TUI


class VectorEmbeddingProcessor(IndexingProcessor):
    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: EmbeddingConfig
    ) -> None:
        super().__init__(chunks, tui, config)
        self._config: EmbeddingConfig
        self._embedder = SentenceTransformer(config.model)
        self._texts = [chunk.text for chunk in self._chunks]
        self._metadatas = [chunk.metadata for chunk in self._chunks]

    def index_corpus(self, save_directory: str) -> None:
        corpus_embeddings = self._embedder.encode_document(
            self._texts, show_progress_bar=True
        )
        chroma_client = chromadb.PersistentClient(save_directory)
        if self._config.collection in [
            c.name for c in chroma_client.list_collections()
        ]:
            chroma_client.delete_collection(self._config.collection)
        collection = chroma_client.create_collection(
            name=self._config.collection
        )
        ids = [f"id{i}" for i in range(len(corpus_embeddings))]

        batched_ids = batched(ids, self._config.batch_size)
        batched_embeddings = batched(
            corpus_embeddings, self._config.batch_size
        )
        batched_documents = batched(self._texts, self._config.batch_size)
        batched_metadatas = batched(self._metadatas, self._config.batch_size)
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
