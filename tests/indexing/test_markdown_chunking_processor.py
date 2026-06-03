import pprint
from textwrap import dedent
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from rag.chunking.markdown_chunking_processor import MarkdownChunkingProcessor


@pytest.fixture
def md_chunking_processor() -> MarkdownChunkingProcessor:
    return MarkdownChunkingProcessor(100, 10, Mock())


@pytest.fixture
def md_document() -> Document:
    text = dedent("""\
        # Sample Test Document

        ## Overview

        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

        ### Key Points

        - First sample item
        - Second sample item
        - Third sample item

        > This is a placeholder quote for testing purposes.

        **Bold text**, *italic text*, and `inline code` can be used to
        validate formatting.

        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Ut enim ad minim veniam.

        ### other header 3

        anything.

        ## header 2

        end
        """)
    return Document(page_content=text)


class TestMarkdownChunkingProcessor:
    def test_generate_breadcrumbs(
        self, md_chunking_processor: MarkdownChunkingProcessor
    ) -> None:
        metadata = {"h1": "how", "h3": "you", "h2": "are"}
        assert (
            md_chunking_processor._generate_breadcrumbs(metadata)
            == "how > are > you"
        )

    def test_header_split(
        self,
        md_chunking_processor: MarkdownChunkingProcessor,
        md_document: Document,
    ) -> None:
        md_header_split = md_chunking_processor._header_split(md_document)
        print(md_header_split)
        assert [doc.metadata["breadcrumbs"] for doc in md_header_split] == [
            "Sample Test Document",
            "Sample Test Document > Overview",
            "Sample Test Document > Overview > Key Points",
            "Sample Test Document > Overview > other header 3",
            "Sample Test Document > header 2",
        ]
        for document in md_header_split:
            assert document.metadata[
                "first_character_index"
            ] == md_document.page_content.find(document.page_content)

    def test_char_level_split(self) -> None:
        md_chunking_processor = MarkdownChunkingProcessor(5, 2, Mock())
        documents = [
            Document(
                page_content="abcdefghijklmnop",
                metadata={"first_character_index": 0},
            ),
            Document(
                page_content="qrstuvwxyzABCDEF",
                metadata={"first_character_index": 16},
            ),
        ]
        char_level_split = md_chunking_processor._char_level_split(documents)

    def test_split(
        self,
        md_chunking_processor: MarkdownChunkingProcessor,
        md_document: Document,
    ) -> None:
        documents = md_chunking_processor._split(md_document)
        pprint.pprint(documents)
        for document in documents:
            start_index = document.metadata["start_index"]
            assert start_index == md_document.page_content.find(
                document.page_content
            )
