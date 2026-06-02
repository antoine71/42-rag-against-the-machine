from pydantic_settings import BaseSettings

from rag.models.chunk import FileType
from rag.text_processing.text_processors import (
    LemmatizationProcessor,
    MarkdownCleaningProcessor,
    TextProcessor,
)


class TextProcessingConfig(BaseSettings):
    processors: dict[FileType, list[type[TextProcessor]]] = {
        FileType.DOCUMENTATION: [
            MarkdownCleaningProcessor,
            LemmatizationProcessor,
        ],
        FileType.CODE: [],
    }
