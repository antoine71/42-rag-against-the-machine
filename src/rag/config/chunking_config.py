from functools import cached_property

from pydantic_settings import BaseSettings


class ChunkingConfig(BaseSettings):
    """Configuration for text chunk sizes and overlap."""

    chunk_size: int = 2000
    overlap_ratio: float = 0.2

    @cached_property
    def overlap(self) -> int:
        """Returns the chunk overlap size in characters.

        Returns:
            The overlap computed from `chunk_size` and `overlap_ratio`.
        """
        return int(self.chunk_size * self.overlap_ratio)
