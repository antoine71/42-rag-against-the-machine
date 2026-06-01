from pydantic_settings import BaseSettings


class BM25Settings(BaseSettings):
    """Configuration settings used by the BM25 retrieval model."""

    k1: float = 0.88
    b: float = 0.7


class BM25Configuration(BaseSettings):
    """Encapsulates BM25 settings for indexing and retrieval."""

    bm25_settings: BM25Settings = BM25Settings()
