from rag.config.text_processing import TextProcessingConfig
from rag.models.chunk import FileType
from rag.text_processing.text_processors import TextProcessor
from rag.tui import TUI


class TextProcessingFactory:
    def __init__(
        self, text_processing_config: TextProcessingConfig, tui: TUI
    ) -> None:
        self._config = text_processing_config
        self._tui = tui

    def create(self, file_type: FileType) -> list[TextProcessor]:
        text_processors = self._config.processors[file_type]
        return [processor(self._tui) for processor in text_processors]
