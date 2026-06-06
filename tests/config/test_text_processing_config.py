import pytest
from pydantic import ValidationError

from rag.config.bm25_config import BM25Configuration
from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import Chunk
from rag.models.file_category import FileCategory
from rag.text_processing.text_processors import (
    CodeCleaningProcessor,
    FilePathExpansionProcessor,
)


class TestTextProcessingConfig:
    def test_bm25_config_accepts_polymorphic_processor_classes(self) -> None:
        config = BM25Configuration()

        assert config.text_processing.processors_for(FileCategory.CODE) == [
            CodeCleaningProcessor,
            FilePathExpansionProcessor,
        ]
        assert config.query_processing.processors_for(FileCategory.CODE) == [
            CodeCleaningProcessor
        ]

    def test_rejects_non_processor_classes(self) -> None:
        with pytest.raises(ValidationError):
            TextProcessingConfig[Chunk](
                processors={FileCategory.CODE: [object]}
            )
