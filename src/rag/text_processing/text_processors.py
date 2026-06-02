import functools
import logging
import unicodedata
from abc import ABC
from textwrap import dedent
from typing import Callable

import spacy
from bs4 import BeautifulSoup
from markdown import markdown

from rag.config.llm import LLMConfig
from rag.llm.llm_manager import LLMManager
from rag.tui import TUI

logger = logging.getLogger(__name__)


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

    @staticmethod
    def process_logger(
        func: Callable[["TextProcessor", str], str],
    ) -> Callable[["TextProcessor", str], str]:
        @functools.wraps(func)
        def wrapper(self, text: str) -> str:
            logger.debug(f"{self.__class__.__name__}: processing:\n{text}")
            text = func(self, text)
            logger.debug(
                f"{self.__class__.__name__}: processing result:\n{text}"
            )
            return text

        return wrapper

    @process_logger
    def process(self, text: str) -> str:
        return self._process_core(text)

    def _process_core(self, text: str) -> str:
        return text


class LemmatizationProcessor(TextProcessor):
    def __init__(self, tui: TUI) -> None:
        super().__init__(tui)
        self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def _process_core(self, text: str) -> str:
        return self._lemmatize(text)

    def _lemmatize(self, text: str) -> str:
        doc = self._nlp(text)
        return " ".join([token.lemma_ for token in doc])


class LowerCasingProcessor(TextProcessor):
    def _process_core(self, text: str) -> str:
        return text.lower()


class MarkdownCleaningProcessor(TextProcessor):
    def _process_core(self, text: str) -> str:
        return BeautifulSoup(markdown(text), "html.parser").get_text("\n")


class CodeCleaningProcessor(TextProcessor):
    def _process_core(self, text: str) -> str:
        return text.lower().replace("_", " ").replace("-", " ")


class UnicodeNormalizerProcessor(TextProcessor):
    def _process_core(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        return text


class SynonymExpansionProcessor(TextProcessor):
    def __init__(self, tui: TUI) -> None:
        super().__init__(tui)
        self._llm_manager = LLMManager(tui, LLMConfig())

    def _get_prompt(self, query: str) -> list[dict[str, str]]:
        return [
            {
                "system": dedent("""\
                Generate 5 high-quality query expansions for RAG retrieval.

                Rules:
                - preserve meaning
                - include synonyms
                - expand acronyms
                - avoid overly broad terms
                - prefer technical / retrieval useful variants
                - return only short phrases

                Output format:
                comma-separated list only""")
            },
            {"user": query},
        ]

    def process_list(self, texts: list[str]) -> list[str]:
        queries = [self._get_prompt(query) for query in texts]
        responses = self._llm_manager.answer_queries(
            queries, self.__class__.__name__
        )
        return [
            text + " " + response for text, response in zip(texts, responses)
        ]
