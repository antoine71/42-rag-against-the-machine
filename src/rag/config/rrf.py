from pydantic_settings import BaseSettings


class RRFConfig(BaseSettings):
    """Configuration for Reciprocal Rank Fusion (RRF) reranking factor."""

    k_factor: int = 4
