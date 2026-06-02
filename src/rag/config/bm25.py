from pydantic_settings import BaseSettings

from rag.config.indexing_config import IndexingConfig
from rag.config.retrieving_config import RetrievingConfig
from rag.config.text_processing import TextProcessingConfig
from rag.indexing.text_processors import (
    LemmatizationProcessor,
    MarkdownCleaningProcessor,
)
from rag.models.chunk import FileType


class BM25Settings(BaseSettings):
    """Configuration settings used by the BM25 retrieval model."""

    k1: float = 0.88
    b: float = 0.7


class BM25Configuration(IndexingConfig, RetrievingConfig):
    """Encapsulates BM25 settings for indexing and retrieval."""

    bm25_settings: BM25Settings = BM25Settings()
    text_processing: TextProcessingConfig = TextProcessingConfig(
        processors={
            FileType.DOCUMENTATION: [
                MarkdownCleaningProcessor,
                LemmatizationProcessor,
            ],
            FileType.CODE: [],
        }
    )
    query_processing: TextProcessingConfig = TextProcessingConfig(
        processors={
            FileType.DOCUMENTATION: [
                LemmatizationProcessor,
            ],
            FileType.CODE: [],
        }
    )
