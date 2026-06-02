from abc import ABC, abstractmethod

import spacy
from bs4 import BeautifulSoup
from markdown import markdown

from rag.tui import TUI


class TextProcessor(ABC):
    def __init__(self, tui: TUI) -> None:
        self._tui = tui

    def process_list(self, texts: list[str]) -> list[str]:
        processed_texts: list[str] = []
        with self._tui.progress(
            self.__class__.__name__, len(texts), "text"
        ) as progess:
            for text in texts:
                processed_texts.append(self.process(text))
                progess.update(1)
        return processed_texts

    @abstractmethod
    def process(self, text: str) -> str: ...


class LemmatizationProcessor(TextProcessor):
    def process(self, text: str) -> str:
        return self._lemmatize(text)

    def _lemmatize(self, text: str) -> str:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return text + " " + " ".join([token.lemma_ for token in doc])


class MarkdownCleaningProcessor(TextProcessor):
    def process(self, text: str) -> str:
        return BeautifulSoup(markdown(text), "html.parser").get_text("\n")


class CodeCleaningProcessor(TextProcessor):
    def process(self, text: str) -> str:
        return text.lower().replace("_", " ").replace("-", " ")
