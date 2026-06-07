from typing import Generic

from rag.text_processing.text_processors import TextProcessor
from rag.text_processing.text_type import TextType
from rag.tui import TUI


class TextProcessingPipeline(Generic[TextType]):
    """Runs a sequence of text processors over text-like items."""

    def __init__(
        self, processing_stages: list[TextProcessor[TextType]], tui: TUI
    ) -> None:
        """Initializes the processing pipeline.

        Args:
            processing_stages: Ordered processors to run.
            tui: Terminal UI retained by the pipeline.
        """
        self._processing_stages: list[TextProcessor[TextType]] = (
            processing_stages
        )
        self._tui = tui

    def process_list(self, texts: list[TextType]) -> list[TextType]:
        """Processes a list through each configured stage.

        Args:
            texts: Items to process.

        Returns:
            Processed items.
        """
        for processing_stage in self._processing_stages:
            texts = processing_stage.process_list(texts)
        return texts

    @property
    def stages(self) -> list[TextProcessor[TextType]]:
        """Returns the configured processor stages.

        Returns:
            Ordered processor instances.
        """
        return self._processing_stages
