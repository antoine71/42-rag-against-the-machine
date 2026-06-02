from abc import ABC

from pydantic_settings import BaseSettings

from rag.config.text_processing import TextProcessingConfig


class IndexingConfig(ABC, BaseSettings):
    text_processing: TextProcessingConfig
