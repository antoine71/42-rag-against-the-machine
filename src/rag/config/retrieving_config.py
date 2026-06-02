from abc import ABC

from pydantic_settings import BaseSettings

from rag.config.text_processing import TextProcessingConfig


class RetrievingConfig(ABC, BaseSettings):
    query_processing: TextProcessingConfig
