from functools import cached_property

from pydantic_settings import BaseSettings


class ChunkingConfig(BaseSettings):
    chunk_size: int = 2000
    overlap_ratio: float = 0.2

    @cached_property
    def overlap(self) -> int:
        return int(self.chunk_size * self.overlap_ratio)
