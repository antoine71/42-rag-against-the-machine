from typing import Generic

from rag.config.text_processing import KeyType, TextProcessingConfig
from rag.text_processing.text_processing_pipeline import TextProcessingPipeline
from rag.text_processing.text_type import TextType
from rag.tui import TUI


class TextProcessingPipelineFactory(Generic[KeyType, TextType]):
    """Factory that builds text processing pipelines from configuration."""

    def __init__(
        self, config: TextProcessingConfig[KeyType, TextType], tui: TUI
    ) -> None:
        """Initializes the factory.

        Args:
            config: Text processing configuration.
            tui: Terminal UI passed to processor instances.
        """
        self._config: TextProcessingConfig[KeyType, TextType] = config
        self._tui = tui

    def create(self, file_type: KeyType) -> TextProcessingPipeline[TextType]:
        """Creates a text processing pipeline for a key.

        Args:
            file_type: File type or category used to select processors.

        Returns:
            Configured text processing pipeline.
        """
        text_processor_classes = self._config.processors_for(file_type)
        text_processors = [cls(self._tui) for cls in text_processor_classes]
        return TextProcessingPipeline(text_processors, self._tui)
