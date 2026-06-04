from typing import TypeVar

from rag.models.chunk import Chunk

TextType = TypeVar("TextType", str, Chunk)
