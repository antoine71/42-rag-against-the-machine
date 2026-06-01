from enum import Enum


class IndexingMethod(str, Enum):
    """Enumerates supported indexing and retrieval strategies."""

    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
