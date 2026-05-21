from typing import cast

import bm25s
from bm25s.tokenization import Tokenized

from rag.models.chunk import Chunk


class BM25RepositoryIndexingProcessor:
    def __init__(
        self,
        chunks: list[Chunk],
    ) -> None:
        self._chunks = [chunk.model_dump() for chunk in chunks]
        self._retriever = bm25s.BM25()

    def index_corpus(self, save_directory: str) -> None:
        corpus = cast(
            Tokenized,
            bm25s.tokenize([chunk["text"] for chunk in self._chunks]),
        )
        self._retriever.index(corpus, show_progress=True, leave_progress=True)
        self._retriever.save(
            save_directory, corpus=self._chunks, show_progress=False
        )
