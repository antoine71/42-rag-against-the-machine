from typing import Generic

from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import FileType
from rag.text_processing.text_processing_pipeline import TextProcessingPipeline
from rag.text_processing.text_type import TextType
from rag.tui import TUI


class ProcessingPipelineFactory(Generic[TextType]):
    def __init__(
        self, config: TextProcessingConfig[TextType], tui: TUI
    ) -> None:
        self._config = config
        self._tui = tui

    def create(self, file_type: FileType) -> TextProcessingPipeline[TextType]:
        text_processor_factories = self._config.processors[file_type]
        text_processors = [
            factory(self._tui) for factory in text_processor_factories
        ]
        return TextProcessingPipeline(text_processors)
