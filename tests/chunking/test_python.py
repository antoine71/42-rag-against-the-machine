from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.chunking.python_file import PythonFileParser


class TestPythonFileParser:
    @pytest.mark.parametrize(
        "line_nbrs,expected",
        [
            ((0, 0), (0, 3)),
            ((0, 1), (0, 7)),
            ((0, 2), (0, 9)),
            ((1, 1), (4, 7)),
            ((1, 2), (4, 9)),
            ((2, 2), (8, 9)),
        ],
    )
    def test_get_position(
        self, line_nbrs: tuple[int, int], expected: tuple[int, int]
    ) -> None:
        parser = PythonFileParser(Mock())
        lines = ["123\n", "567\n", "89"]
        assert parser._get_position(lines, *line_nbrs) == expected

    def test_chunck(self) -> None:
        source_file = Path() / "tests" / "data" / "mock_python_file.py"
        source = source_file.read_text()
        parser = PythonFileParser(Mock())
        chunks = parser.chunk(source)

        assert len(chunks) == 2

        assert chunks[0]["type"] == "ClassDef"
        assert chunks[0]["name"] == "TestClass"
        assert "TestClass" in chunks[0]["text"]

        assert chunks[1]["type"] == "FunctionDef"
        assert chunks[1]["name"] == "test_fct"
        assert "test_fct" in chunks[1]["text"]
        assert 'return "test return value"' in chunks[1]["text"]
