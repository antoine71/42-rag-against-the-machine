from collections.abc import Callable

import bm25s
from bm25s.tokenization import Tokenized

from rag.models.chunk import Chunk


class BM25RepositoryIndexingProcessor:
    def __init__(
        self,
        chunks: list[Chunk],
        tokenize_batch: Callable[[list[str]], list[list[int]]],
        vocab: dict[str, int],
    ) -> None:
        self._chunks = [chunk.model_dump() for chunk in chunks]
        self._tokenize_batch = tokenize_batch
        self._vocab = vocab
        self._retriever = bm25s.BM25(corpus=chunks)

    def index_corpus(self, save_directory: str) -> None:
        tokens = self._tokenize_batch(
            [chunk["text"] for chunk in self._chunks]
        )
        self._retriever.index(Tokenized(tokens, self._vocab))
        self._retriever.save(save_directory, corpus=self._chunks)
