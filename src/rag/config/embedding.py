from typing import ClassVar

from rag.chunking.chunk_enchancement_processor import (
    FilePathExpansionProcessor,
)
from rag.config.indexing_config import IndexingConfig
from rag.config.retrieving_config import RetrievingConfig
from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import Chunk, FileType
from rag.text_processing.text_processors import (
    CodeCleaningProcessor,
)


class EmbeddingConfig(IndexingConfig, RetrievingConfig):
    """Configuration for embedding model selection and ChromaDB batching."""

    TYPE: ClassVar[str] = "Vector embedding"

    model: str = "sentence-transformers/msmarco-bert-base-dot-v5"
    embedding_batch_size: int = 1
    chromadb_batch_size: int = 300
    collection: str = "rag_vllm_repository"
    text_processing: TextProcessingConfig[Chunk] = TextProcessingConfig(
        processors={
            FileType.DOCUMENTATION: [FilePathExpansionProcessor],
            FileType.CODE: [CodeCleaningProcessor, FilePathExpansionProcessor],
        }
    )
    query_processing: TextProcessingConfig[str] = TextProcessingConfig(
        processors={
            FileType.DOCUMENTATION: [],
            FileType.CODE: [],
        }
    )
