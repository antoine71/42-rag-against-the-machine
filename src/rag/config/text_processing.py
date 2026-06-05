from typing import Any, Generic, cast

from pydantic_settings import BaseSettings

from rag.models.chunk import FileType
from rag.text_processing.text_processors import (
    TextProcessor,
)
from rag.text_processing.text_type import TextType


class TextProcessingConfig(BaseSettings, Generic[TextType]):
    processors: dict[FileType, list[Any]]

    def processors_for(
        self, file_type: FileType
    ) -> list[type[TextProcessor[TextType]]]:
        return cast(
            list[type[TextProcessor[TextType]]],
            self.processors[file_type],
        )
