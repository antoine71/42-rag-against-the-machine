from abc import ABC
from typing import ClassVar

from pydantic_settings import BaseSettings

from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import Chunk
from rag.models.file_type import FileType


class IndexingConfig(ABC, BaseSettings):
    """Base configuration shared by indexing processors."""

    TYPE: ClassVar[str]

    text_processing: TextProcessingConfig[FileType, Chunk]
