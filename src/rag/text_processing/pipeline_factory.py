from typing import Generic

from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import FileType
from rag.text_processing.text_processing_pipeline import TextProcessingPipeline
from rag.text_processing.text_type import TextType
from rag.tui import TUI


class TextProcessingPipelineFactory(Generic[TextType]):
    def __init__(
        self, config: TextProcessingConfig[TextType], tui: TUI
    ) -> None:
        self._config: TextProcessingConfig[TextType] = config
        self._tui = tui

    def create(self, file_type: FileType) -> TextProcessingPipeline[TextType]:
        text_processor_classes = self._config.processors_for(file_type)
        text_processors = [cls(self._tui) for cls in text_processor_classes]
        return TextProcessingPipeline(text_processors, self._tui)
