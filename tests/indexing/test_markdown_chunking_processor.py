import pytest

from rag.indexing.langchain_chunking_processor import MarkdownChunkingProcessor


@pytest.fixture
def md_chunking_processor() -> MarkdownChunkingProcessor:
    return MarkdownChunkingProcessor()


class TestMarkdownChunkingProcessor:
    def test_generate_breadcrumbs(
        self, md_chunking_processor: MarkdownChunkingProcessor
    ) -> None:
        metadata = {"h1": "how", "h3": "you", "h2": "are"}
        assert (
            md_chunking_processor._generate_breadcrumbs(metadata)
            == "how > are > you"
        )
