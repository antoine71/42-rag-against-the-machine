import time

import chromadb
from more_itertools import batched
from sentence_transformers import SentenceTransformer

from rag.config.embedding import EmbeddingConfig
from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk
from rag.models.file_category import FileCategory
from rag.models.indexing_method import IndexingMethod
from rag.tui import TUI
from rag.utils.files_manager import FilesManager
from rag.utils.measure import measure
from rag.utils.patch_sentence_transformers_tqdm import patch_tqdm


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

    @patch_tqdm
    def _index_and_save(
        self,
        processed_chunks: list[Chunk],
        save_directory: str,
        file_type: FileCategory,
    ) -> str:
        """Embeds processed chunks and saves them in ChromaDB.

        Args:
            processed_chunks: Text-processed chunks to embed.
            save_directory: Root directory where the vector store is saved.
            file_type: File category represented by the vector store.

        Returns:
            Directory containing the saved ChromaDB collection.
        """
        texts = [chunk.text for chunk in processed_chunks]
        metadatas = [chunk.metadata for chunk in processed_chunks]
        corpus_embeddings, delta = measure(
            self._embedder.encode_document,
            texts,
            show_progress_bar=True,
            batch_size=self._config.embedding_batch_size,
        )
        self._tui.print_task_report(
            f"{self._config.TYPE}", delta, "chunks", len(texts)
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
        start_save = time.perf_counter()
        with self._tui.progress(
            "Saving to database", len(processed_chunks), "chunk"
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
        end_save = time.perf_counter()
        delta = int((end_save - start_save) * 1000)
        self._tui.print_task_report(
            "data persistence",
            delta,
            "chunks",
            len(texts),
        )
        return save_path
