from typing import cast

import bm25s
from bm25s.tokenization import Tokenized

from rag.indexing.indexing_processor import IndexingProcessor
from rag.models.chunk import Chunk


class BM25RepositoryIndexingProcessor(IndexingProcessor):
    def __init__(
        self,
        chunks: list[Chunk],
    ) -> None:
        super().__init__(chunks)
        self._retriever = bm25s.BM25()

    def index_corpus(self, save_directory: str) -> None:
        corpus = cast(Tokenized, bm25s.tokenize(self._texts))
        self._retriever.index(corpus, show_progress=True, leave_progress=True)
        self._retriever.save(
            save_directory, corpus=self._corpus, show_progress=False
        )
