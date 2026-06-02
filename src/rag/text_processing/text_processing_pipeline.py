from rag.text_processing.text_processors import TextProcessor


class TextProcessingPipeline:
    def __init__(self, processing_stages: list[TextProcessor]) -> None:
        self._processing_stages = processing_stages

    def process_list(self, texts: list[str]) -> list[str]:
        for processing_stage in self._processing_stages:
            texts = processing_stage.process_list(texts)
        return texts
