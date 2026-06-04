from typing import Generic

from pydantic_settings import BaseSettings

from rag.models.chunk import FileType
from rag.text_processing.text_processors import (
    TextProcessor,
)
from rag.text_processing.text_type import TextType


class TextProcessingConfig(BaseSettings, Generic[TextType]):
    processors: dict[FileType, list[type[TextProcessor]]]
