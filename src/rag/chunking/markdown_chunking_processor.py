import logging

from langchain_core.documents import Document
from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter,
    RecursiveCharacterTextSplitter,
)
from markdown_it import MarkdownIt

from rag.chunking.chunking_processor import ChunkingProcessor
from rag.tui import TUI

logger = logging.getLogger(__name__)


class MarkdownChunkingProcessor(ChunkingProcessor):
    def __init__(self, max_chunk_size: int, overlap: int, tui: TUI) -> None:
        self._headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
        ]
        self._markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
            length_function=len,
        )
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            add_start_index=True,
            length_function=len,
        )
        self._cleanup_splitter = ExperimentalMarkdownSyntaxTextSplitter(
            headers_to_split_on=self._headers_to_split_on, strip_headers=True
        )
        self._tui = tui

    def _enrich_header_splits(self, md_header_splits: list[Document]) -> None:
        start_index = 0
        for doc in md_header_splits:
            doc.metadata["first_character_index"] = start_index
            start_index += len(doc.page_content)

    def _generate_breadcrumbs(self, metadata: dict[str, str]) -> str:
        valid_headers = {header for _, header in self._headers_to_split_on}
        filtered_items = [
            (header, title)
            for header, title in metadata.items()
            if header in valid_headers
        ]
        sorted_items = sorted(filtered_items, key=lambda item: item[0])
        titles = [title for _, title in sorted_items]

        return " > ".join(titles)

    def _header_split(self, document: Document) -> list[Document]:
        md_header_splits = self._markdown_splitter.split_text(
            document.page_content
        )

        start_index = 0
        for doc in md_header_splits:
            doc.metadata |= document.metadata
            doc.metadata["breadcrumbs"] = self._generate_breadcrumbs(
                doc.metadata
            )
            doc.metadata["first_character_index"] = start_index
            start_index += len(doc.page_content)

        return md_header_splits

    def _char_level_split(
        self,
        md_header_splits: list[Document],
    ) -> list[Document]:
        splits = self._text_splitter.split_documents(md_header_splits)
        for doc in splits:
            doc.metadata["start_index"] += doc.metadata.pop(
                "first_character_index"
            )
        return splits

    def _is_lonely_header(self, text: str) -> bool:
        md = MarkdownIt()
        tokens = md.parse(text)
        meaningful_tokens = [
            token for token in tokens if token.type != "inline"
        ]
        return all(
            token.type in ("heading_open", "heading_close")
            for token in meaningful_tokens
        )

    def _cleanup_lonely_headers(
        self, char_level_splits: list[Document]
    ) -> list[Document]:
        return [
            doc
            for doc in char_level_splits
            if not self._is_lonely_header(doc.page_content)
        ]

    def _split(self, document: Document) -> list[Document]:

        md_header_splits = self._header_split(document)
        char_level_splits = self._char_level_split(md_header_splits)

        return char_level_splits

    def split_documents(self, documents: list[Document]) -> list[Document]:

        return self._markdown_splitter.split_documents(documents)
