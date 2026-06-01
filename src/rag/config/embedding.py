from pydantic_settings import BaseSettings


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding model selection and ChromaDB batching."""

    # model: str = "Qwen/Qwen3-Embedding-0.6B"
    model: str = "sentence-transformers/msmarco-bert-base-dot-v5"
    batch_size: int = 300
    collection: str = "rag_vllm_repository"
