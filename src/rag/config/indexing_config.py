from abc import ABC
from typing import ClassVar

from pydantic_settings import BaseSettings

from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import Chunk


class IndexingConfig(ABC, BaseSettings):
    TYPE: ClassVar[str]

    text_processing: TextProcessingConfig[Chunk]
