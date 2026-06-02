from rag.config.indexing_config import IndexingConfig
from rag.config.retrieving_config import RetrievingConfig
from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import FileType


class EmbeddingConfig(IndexingConfig, RetrievingConfig):
    """Configuration for embedding model selection and ChromaDB batching."""

    model: str = "sentence-transformers/msmarco-bert-base-dot-v5"
    # model: str = "sentence-transformers/all-mpnet-base-v2"
    # model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_batch_size: int = 1
    chromadb_batch_size: int = 300
    collection: str = "rag_vllm_repository"
    text_processing: TextProcessingConfig = TextProcessingConfig(
        processors={FileType.DOCUMENTATION: [], FileType.CODE: []}
    )
    query_processing: TextProcessingConfig = TextProcessingConfig(
        processors={FileType.DOCUMENTATION: [], FileType.CODE: []}
    )
