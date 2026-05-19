from rag.chunking.base_parser import BaseParser


class MarkdownFileParser(BaseParser):
    def chunk(self, source: str, max_tokens=300) -> list[dict[str, Any]]:
        pass
