from typing import ClassVar

from pydantic_settings import BaseSettings

from rag.config.indexing_config import IndexingConfig
from rag.config.retrieving_config import RetrievingConfig
from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import Chunk
from rag.models.file_category import FileCategory
from rag.models.file_type import FileType
from rag.text_processing.text_processors import (
    CodeCleaningProcessor,
    FilePathExpansionProcessor,
    LemmatizationProcessor,
)


class BM25Settings(BaseSettings):
    """Configuration settings used by the BM25 retrieval model."""

    k1: float = 1.2
    b: float = 0.75


class BM25Configuration(IndexingConfig, RetrievingConfig):
    """Encapsulates BM25 settings for indexing and retrieval."""

    TYPE: ClassVar[str] = "BM25"

    bm25_settings: BM25Settings = BM25Settings()
    text_processing: TextProcessingConfig[FileType, Chunk] = (
        TextProcessingConfig(
            processors={
                FileType.MARKDOWN: [
                    LemmatizationProcessor,
                    FilePathExpansionProcessor,
                ],
                FileType.TEXT: [
                    LemmatizationProcessor,
                    FilePathExpansionProcessor,
                ],
                FileType.PYTHON: [
                    CodeCleaningProcessor,
                    FilePathExpansionProcessor,
                ],
            }
        )
    )
    query_processing: TextProcessingConfig[FileCategory, str] = (
        TextProcessingConfig(
            processors={
                FileCategory.DOCUMENTATION: [LemmatizationProcessor],
                FileCategory.CODE: [CodeCleaningProcessor],
                FileCategory.ALL: [LemmatizationProcessor],
            }
        )
    )
