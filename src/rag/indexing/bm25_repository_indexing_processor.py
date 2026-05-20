import bm25s

from rag.models.chunk import Chunk


class BM25RepositoryIndexingProcessor:
    def __init__(
        self,
        chunks: list[Chunk],
    ) -> None:
        self._chunks = [chunk.model_dump() for chunk in chunks]
        self._retriever = bm25s.BM25(corpus=chunks)

    def index_corpus(self, save_directory: str) -> None:
        # tokens = self._tokenize_batch(
        #     [chunk["text"] for chunk in self._chunks]
        # )
        tokens = bm25s.tokenize([chunk["text"] for chunk in self._chunks])

        # self._retriever.index(
        #     Tokenized(tokens, self._vocab), show_progress=False
        # )
        self._retriever.index(tokens, show_progress=True, leave_progress=True)
        self._retriever.save(
            save_directory, corpus=self._chunks, show_progress=True
        )
