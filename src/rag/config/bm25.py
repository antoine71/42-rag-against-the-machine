from pydantic_settings import BaseSettings


class BM25Settings(BaseSettings):
    k1: float = 0.88
    b: float = 0.7


class BM25Configuration(BaseSettings):
    bm25_settings: BM25Settings = BM25Settings()
