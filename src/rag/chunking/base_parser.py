from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class BaseParser(ABC):
    def __init__(self, encode: Callable[[str], list[int]]) -> None:
        self._encode = encode

    def _count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    @abstractmethod
    def chunk(self, source: str, max_tokens=300) -> list[dict[str, Any]]: ...
