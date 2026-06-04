from pydantic_settings import BaseSettings

from rag.config.indexing_config import IndexingConfig
from rag.config.retrieving_config import RetrievingConfig
from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import FileType
from rag.text_processing.text_processors import (
    CodeCleaningProcessor,
    FilePathExpansionProcessor,
)


class BM25Settings(BaseSettings):
    """Configuration settings used by the BM25 retrieval model."""

    k1: float = 1.2
    b: float = 0.75


class BM25Configuration(IndexingConfig, RetrievingConfig):
    """Encapsulates BM25 settings for indexing and retrieval."""

    bm25_settings: BM25Settings = BM25Settings()
    text_processing: TextProcessingConfig = TextProcessingConfig(
        processors={
            FileType.DOCUMENTATION: [FilePathExpansionProcessor],
            FileType.CODE: [CodeCleaningProcessor, FilePathExpansionProcessor],
        }
    )
    query_processing: TextProcessingConfig = TextProcessingConfig(
        processors={
            FileType.DOCUMENTATION: [],
            FileType.CODE: [CodeCleaningProcessor],
        }
    )
