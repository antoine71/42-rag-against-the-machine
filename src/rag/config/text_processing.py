from typing import Any, Generic, TypeVar, cast

from pydantic_settings import BaseSettings

from rag.models.file_category import FileCategory
from rag.models.file_type import FileType
from rag.text_processing.text_processors import (
    TextProcessor,
)
from rag.text_processing.text_type import TextType

KeyType = TypeVar("KeyType", FileCategory, FileType)


class TextProcessingConfig(BaseSettings, Generic[KeyType, TextType]):
    processors: dict[KeyType, list[Any]]

    def processors_for(
        self, file_type: KeyType
    ) -> list[type[TextProcessor[TextType]]]:
        return cast(
            list[type[TextProcessor[TextType]]],
            self.processors[file_type],
        )
