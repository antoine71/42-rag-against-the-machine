import chromadb
from more_itertools import batched
from sentence_transformers import SentenceTransformer

from rag.config.embedding import EmbeddingConfig
from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk, FileType
from rag.models.indexing_method import IndexingMethod
from rag.tui import TUI
from rag.utils.files_manager import FilesManager


class VectorEmbeddingProcessor(IndexingProcessor):
    """An indexing processor that creates and saves vector embeddings to
    ChromaDB.
    """

    def __init__(
        self, chunks: list[Chunk], tui: TUI, config: EmbeddingConfig
    ) -> None:
        """Initializes the VectorEmbeddingProcessor.

        Args:
            chunks: A list of Chunk models to be embedded.
            tui: A TUI instance to handle user interface / progress output.
            config: An EmbeddingConfig object containing embedding
                configuration.
        """
        super().__init__(chunks, tui, config)
        self._config: EmbeddingConfig
        self._embedder = SentenceTransformer(config.model)

    def index_corpus(self, save_directory: str, file_type: FileType) -> None:
        """Encodes text chunks in batch and saves the resulting embeddings.

        Args:
            save_directory: The directory path where ChromaDB database files
                should be stored.
        """
        self._ui.print(f"Indexing {file_type} files...")
        chunks = [chunk for chunk in self._chunks if chunk.type == file_type]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        corpus_embeddings = self._embedder.encode_document(
            texts,
            show_progress_bar=True,
            batch_size=self._config.embedding_batch_size,
        )
        save_path = FilesManager.get_indexing_directory(
            save_directory, IndexingMethod.VECTOR, file_type
        )
        chroma_client = chromadb.PersistentClient(save_path)
        if self._config.collection in [
            c.name for c in chroma_client.list_collections()
        ]:
            chroma_client.delete_collection(self._config.collection)
        collection = chroma_client.create_collection(
            name=self._config.collection
        )
        ids = [f"id{i}" for i in range(len(corpus_embeddings))]

        batched_ids = batched(ids, self._config.chromadb_batch_size)
        batched_embeddings = batched(
            corpus_embeddings, self._config.chromadb_batch_size
        )
        batched_documents = batched(texts, self._config.chromadb_batch_size)
        batched_metadatas = batched(
            metadatas, self._config.chromadb_batch_size
        )
        with self._ui.progress(
            "Saving to database", len(chunks), "chunk"
        ) as progress:
            while True:
                try:
                    ids = list(next(batched_ids))
                    batch_size = len(ids)
                    collection.add(
                        ids=ids,
                        embeddings=list(next(batched_embeddings)),
                        documents=list(next(batched_documents)),
                        metadatas=list(next(batched_metadatas)),
                    )
                    progress.update(batch_size)
                except StopIteration:
                    break
        self._ui.print(f"Ingestion complete! Indices saved under {save_path}")
