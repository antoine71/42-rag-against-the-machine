from typing import ClassVar

from rag.chunking.chunk_enchancement_processor import (
    FilePathExpansionProcessor,
)
from rag.config.indexing_config import IndexingConfig
from rag.config.retrieving_config import RetrievingConfig
from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import Chunk
from rag.models.file_category import FileCategory
from rag.models.file_type import FileType
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
    text_processing: TextProcessingConfig[FileType, Chunk] = (
        TextProcessingConfig(
            processors={
                FileType.MARKDOWN: [FilePathExpansionProcessor],
                FileType.TEXT: [FilePathExpansionProcessor],
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
                FileCategory.DOCUMENTATION: [],
                FileCategory.CODE: [],
                FileCategory.ALL: [],
            }
        )
    )
