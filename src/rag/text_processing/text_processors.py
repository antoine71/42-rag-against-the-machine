import functools
import logging
import re
from abc import ABC
from pathlib import Path
from typing import Callable, Generic

import spacy

from rag.models.chunk import Chunk
from rag.text_processing.text_type import TextType
from rag.tui import TUI

logger = logging.getLogger(__name__)


class TextProcessor(ABC, Generic[TextType]):
    """Base class for text processors used in indexing and retrieval."""

    def __init__(self, tui: TUI) -> None:
        """Initializes the text processor.

        Args:
            tui: Terminal UI used for progress output.
        """
        self._tui = tui

    def process_list(self, texts: list[TextType]) -> list[TextType]:
        """Processes a list of text-like items.

        Args:
            texts: Items to process.

        Returns:
            Processed items.
        """
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
        """Wraps a processing function with debug logging.

        Args:
            func: Processing function to wrap.

        Returns:
            Wrapped processing function.
        """

        @functools.wraps(func)
        def wrapper(
            self: "TextProcessor[TextType]", item: TextType
        ) -> TextType:
            """Logs input and output around a processing call.

            Args:
                self: Text processor instance.
                item: Item to process.

            Returns:
                Processed item.
            """
            logger.debug(f"{self.__class__.__name__}: processing:\n{item}")
            item = func(self, item)
            logger.debug(
                f"{self.__class__.__name__}: processing result:\n{item}"
            )
            return item

        return wrapper

    @process_logger
    def process(self, item: TextType) -> TextType:
        """Processes a single item.

        Args:
            item: Item to process.

        Returns:
            Processed item.
        """
        return self._process_core(item)

    def _process_chunk(self, item: Chunk) -> Chunk:
        """Processes the text content of a chunk in place.

        Args:
            item: Chunk to process.

        Returns:
            Chunk with processed text.
        """
        cleaned_code = self._process_str(item.text)
        item.text = cleaned_code
        return item

    @functools.lru_cache
    def _process_str(self, item: str) -> str:
        """Processes a plain string.

        Args:
            item: Text to process.

        Returns:
            Processed text.

        Raises:
            NotImplementedError: Always raised by the base implementation.
        """
        raise NotImplementedError

    def _process_core(self, item: TextType) -> TextType:
        """Dispatches processing based on item type.

        Args:
            item: String or chunk to process.

        Returns:
            Processed item.
        """
        if isinstance(item, Chunk):
            return self._process_chunk(item)
        return self._process_str(item)


class CodeCleaningProcessor(TextProcessor[TextType]):
    """Expands and normalizes code-like text for lexical retrieval."""

    def _humanize(self, name: str) -> str:
        """Splits camel-case or PascalCase identifiers into words.

        Args:
            name: Identifier to split.

        Returns:
            Lowercase words separated by spaces.
        """
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", name)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", s)
        return s.lower()

    def _remove_pascal_case(self, content: str) -> str:
        """Rewrites PascalCase tokens inside text.

        Args:
            content: Text that may contain PascalCase tokens.

        Returns:
            Text with PascalCase tokens expanded.
        """
        content = re.sub(
            r"\b[A-Z][a-zA-Z0-9]*\b",
            lambda m: self._humanize(m.group(0)),
            content,
        )
        return content

    @functools.lru_cache
    def _process_str(self, item: str) -> str:
        """Normalizes code text and appends an expanded variant.

        Args:
            item: Code text to normalize.

        Returns:
            Original-like text plus normalized code text.
        """
        cleaned_code = item.replace("_", " ").replace("-", " ")
        cleaned_code = self._remove_pascal_case(cleaned_code)
        cleaned_code = re.sub(
            r"[()\[\]{}.,:;=+\-*/<>!&|%^~`\"']", " ", cleaned_code
        )
        item = self._remove_pascal_case(cleaned_code)
        item = item + "\n" + cleaned_code
        return item


class FilePathExpansionProcessor(TextProcessor[Chunk]):
    """Prepends path-derived breadcrumbs to chunk text."""

    @functools.lru_cache
    def _generate_breadcrumbs(self, file_path: Path, repository: str) -> str:
        """Builds search-friendly breadcrumbs relative to a repository.

        Args:
            file_path: Source file path.
            repository: Repository root path.

        Returns:
            Space-separated path components without the file suffix.
        """
        relative_path = file_path.relative_to(repository)
        path_components = list(relative_path.parts[:-1])
        path_components.append(relative_path.stem)
        return " ".join(c.replace("_", " ") for c in path_components)

    @functools.lru_cache
    def _concatenate_breadcrumbs(self, text: str, breadcrumbs: str) -> str:
        """Prepends breadcrumbs to text.

        Args:
            text: Original chunk text.
            breadcrumbs: Path-derived breadcrumbs.

        Returns:
            Text prefixed by breadcrumbs.
        """
        return breadcrumbs + "\n" + text

    def _process_core(self, item: Chunk) -> Chunk:
        """Adds path breadcrumbs to a chunk.

        Args:
            item: Chunk to enrich.

        Returns:
            Chunk with breadcrumbs prepended to its text.
        """
        breadcrumbs = self._generate_breadcrumbs(
            Path(item.file_path), item.repository
        )
        item.text = self._concatenate_breadcrumbs(item.text, breadcrumbs)
        return item


class LemmatizationProcessor(TextProcessor[str]):
    """Appends spaCy lemmas to text for lexical matching."""

    def __init__(self, tui: TUI) -> None:
        """Initializes the lemmatizer.

        Args:
            tui: Terminal UI used for progress output.
        """
        super().__init__(tui)
        self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    @functools.lru_cache
    def _process_str(self, item: str) -> str:
        """Appends the lemmatized form of a string.

        Args:
            item: Text to lemmatize.

        Returns:
            Original text followed by its lemmas.
        """
        return item + "\n" + self._lemmatize(item)

    def _lemmatize(self, item: str) -> str:
        """Lemmatizes text with spaCy.

        Args:
            item: Text to lemmatize.

        Returns:
            Lemmas joined by spaces.
        """
        doc = self._nlp(item)
        return " ".join([token.lemma_ for token in doc])
