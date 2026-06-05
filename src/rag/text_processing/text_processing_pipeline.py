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
        if self._processing_stages:
            stages = " > ".join(
                stage.__class__.__name__ for stage in self._processing_stages
            )
            self._tui.print(f"Running Text Processing Pipeline: {stages}")
        for processing_stage in self._processing_stages:
            texts = processing_stage.process_list(texts)
        return texts
