from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    model: str = "Qwen/Qwen3-0.6B"
    batch_size: int = 1
