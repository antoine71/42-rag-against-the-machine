from typing import Generic

from rag.text_processing.text_processors import TextProcessor
from rag.text_processing.text_type import TextType
from rag.tui import TUI


class TextProcessingPipeline(Generic[TextType]):
    def __init__(
        self, processing_stages: list[TextProcessor[TextType]], tui: TUI
    ) -> None:
        self._processing_stages: list[TextProcessor[TextType]] = (
            processing_stages
        )
        self._tui = tui

    def process_list(self, texts: list[TextType]) -> list[TextType]:
        for processing_stage in self._processing_stages:
            texts = processing_stage.process_list(texts)
        return texts

    @property
    def stages(self) -> list[TextProcessor[TextType]]:
        return self._processing_stages
