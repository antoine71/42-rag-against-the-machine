import ast
from collections.abc import Callable
from typing import Any


class PythonFileParser:
    def __init__(self, encode: Callable[[str], list[int]]) -> None:
        self._encode = encode

    def _count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def _get_position(
        self, lines: list[str], start_line: int, end_line: int
    ) -> tuple[int, int]:
        pos_start = sum(len(l) for l in lines[:start_line])
        pos_end = (
            pos_start
            + sum(len(l) for l in lines[start_line : end_line + 1])
            - 1
        )
        return pos_start, pos_end

    def _split_text(self, node: ast.ClassDef) -> None:
        sub_chunks = []

    def _split_class(self, node: ast.ClassDef) -> None:
        node_start_line = node.lineno - 1
        sub_nodes = (
            sub_node
            for sub_node in node.body
            if isinstance(sub_node, ast.FunctionDef)
        )
        for sub_node in sub_nodes:
            start_line = sub_node.lineno - 1
            end_line = sub_node.end_lineno
            assert end_line is not None

    def chunk(self, source: str, max_tokens=300) -> list[dict[str, Any]]:
        chunks = []
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        nodes = (
            node
            for node in tree.body
            if isinstance(
                node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
        )
        for node in nodes:
            line_start = node.lineno - 1
            line_end = (node.end_lineno or len(lines)) - 1
            text = "".join(lines[line_start : line_end + 1])
            if self._count_tokens(text) > max_tokens:
                sub_chunks = self._split_chunk(text)

            else:
                pos_start, pos_end = self._get_position(
                    lines, line_start, line_end
                )
                chunks.append(
                    {
                        "text": text,
                        "type": type(node).__name__,
                        "name": node.name,
                        "start": pos_start,
                        "end": pos_end,
                    }
                )
        return chunks
