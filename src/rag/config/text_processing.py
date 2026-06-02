from pydantic_settings import BaseSettings

from rag.indexing.text_processors import (
    LemmatizationProcessor,
    MarkdownCleaningProcessor,
    TextProcessor,
)
from rag.models.chunk import FileType


class TextProcessingConfig(BaseSettings):
    processors: dict[FileType, list[type[TextProcessor]]] = {
        FileType.DOCUMENTATION: [
            MarkdownCleaningProcessor,
            LemmatizationProcessor,
        ],
        FileType.CODE: [],
    }
