import functools
import logging
import re
from abc import ABC
from pathlib import Path
from typing import Callable, Generic

from rag.models.chunk import Chunk
from rag.text_processing.text_type import TextType
from rag.tui import TUI

logger = logging.getLogger(__name__)


class TextProcessor(ABC, Generic[TextType]):
    def __init__(self, tui: TUI) -> None:
        self._tui = tui

    def process_list(self, texts: list[TextType]) -> list[TextType]:
        processed_texts: list[TextType] = []
        with self._tui.progress(
            self.__class__.__name__, len(texts), "text"
        ) as progess:
            for text in texts:
                processed_texts.append(self.process(text))
                progess.update(1)
        return processed_texts

    @staticmethod
    def process_logger(
        func: Callable[["TextProcessor[TextType]", TextType], TextType],
    ) -> Callable[["TextProcessor[TextType]", TextType], TextType]:
        @functools.wraps(func)
        def wrapper(self, item: TextType) -> TextType:
            logger.debug(f"{self.__class__.__name__}: processing:\n{item}")
            item = func(self, item)
            logger.debug(
                f"{self.__class__.__name__}: processing result:\n{item}"
            )
            return item

        return wrapper

    @process_logger
    def process(self, item: TextType) -> TextType:
        return self._process_core(item)

    def _process_core(self, item: TextType) -> TextType:
        return item


class CodeCleaningProcessor(TextProcessor[TextType]):
    def _process_core(self, item: TextType) -> TextType:
        return self._process_unit(item)

    def _humanize(self, name: str) -> str:
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", s)
        return s.lower()

    def _remove_pascal_case(self, content: str) -> str:
        content = re.sub(
            r"\b[A-Z][a-zA-Z0-9]*\b",
            lambda m: self._humanize(m.group(0)),
            content,
        )
        return content

    @functools.singledispatchmethod
    def _process_unit(self, item: TextType) -> TextType:
        raise NotImplementedError

    @_process_unit.register
    def _(self, item: Chunk) -> Chunk:
        cleaned_code = item.text.replace("_", " ").replace("-", " ")
        cleaned_code = self._remove_pascal_case(cleaned_code)
        cleaned_code = re.sub(
            r"[()\[\]{}.,:;=+\-*/<>!&|%^~`\"']", " ", cleaned_code
        )
        item.text = item.text + "\n" + cleaned_code
        return item

    @_process_unit.register
    def _(self, item: str) -> str:
        cleaned_code = item.replace("_", " ").replace("-", " ")
        cleaned_code = self._remove_pascal_case(cleaned_code)
        cleaned_code = re.sub(
            r"[()\[\]{}.,:;=+\-*/<>!&|%^~`\"']", " ", cleaned_code
        )
        item = self._remove_pascal_case(cleaned_code)
        item = item + "\n" + cleaned_code
        return item


class FilePathExpansionProcessor(TextProcessor[Chunk]):
    def _generate_breadcrumbs(self, file_path: Path, repository: str) -> str:
        relative_path = file_path.relative_to(repository)
        path_components = list(relative_path.parts[:-1])
        path_components.append(relative_path.stem)
        return " ".join(c.replace("_", " ") for c in path_components)

    def _process_core(self, item: Chunk) -> Chunk:
        breadcrumbs = self._generate_breadcrumbs(
            Path(item.file_path), item.repository
        )
        item.text = breadcrumbs + "\n" + item.text
        return item


# class LemmatizationProcessor(TextProcessor):
#     def __init__(self, tui: TUI) -> None:
#         super().__init__(tui)
#         self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
#
#     def _process_core(self, item: str) -> str:
#         return self._lemmatize(item)
#
#     def _lemmatize(self, item: str) -> str:
#         doc = self._nlp(item)
#         return " ".join([token.lemma_ for token in doc])
#
#
# class LowerCasingProcessor(TextProcessor):
#     def _process_core(self, text: str) -> str:
#         return text.lower()
#
#
# class MarkdownCleaningProcessor(TextProcessor):
#     def _process_core(self, text: str) -> str:
#         text = BeautifulSoup(markdown(text), "html.parser").get_text(" ")
#         return text
#
#
# class CodeCleaningProcessor(TextProcessor):
#     def _process_core(self, text: str) -> str:
#         return text.lower().replace("_", " ").replace("-", " ")
#
#
# class MarkdownBreadcrumbsExpansionProcessor(TextProcessor):
#     def _process_core(self, chunk: Chunk) -> Chunk:
#         if chunk.first_character_index == 0:
#             return chunk
#         headers_hierarchy = header_hierarchy_at(
#             Path(chunk.file_path).read_text(), chunk.first_character_index
#         )
#         breadcrumbs = " > ".join(
#             header["title"] for header in headers_hierarchy
#         )
#         chunk.text = breadcrumbs + "\n" + chunk.text
#         return chunk
#
#
# class SynonymExpansionProcessor(TextProcessor):
#     def __init__(self, tui: TUI) -> None:
#         super().__init__(tui)
#         self._llm_manager = LLMManager(tui, LLMConfig())
#
#     def _get_prompt(self, query: str) -> list[dict[str, str]]:
#         return [
#             {
#                 "system": dedent("""\
#                 Generate 5 high-quality query expansions for RAG retrieval.
#
#                 Rules:
#                 - preserve meaning
#                 - include synonyms
#                 - expand acronyms
#                 - avoid overly broad terms
#                 - prefer technical / retrieval useful variants
#                 - return only short phrases
#
#                 Output format:
#                 comma-separated list only""")
#             },
#             {"user": query},
#         ]
#
#     def process_list(self, texts: list[str]) -> list[str]:
#         queries = [self._get_prompt(query) for query in texts]
#         responses = self._llm_manager.answer_queries(
#             queries, self.__class__.__name__
#         )
#         return [
#             text + " " + response for text, response in zip(texts, responses)
#         ]
