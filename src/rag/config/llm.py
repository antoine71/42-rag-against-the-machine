from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """Configuration for the LLM model, batching, and generation settings."""

    model: str = "Qwen/Qwen3-0.6B"
    batch_size: int = 1
    max_new_tokens: int = 256
