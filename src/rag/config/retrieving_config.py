from abc import ABC
from typing import ClassVar

from pydantic_settings import BaseSettings

from rag.config.text_processing import TextProcessingConfig


class RetrievingConfig(ABC, BaseSettings):
    TYPE: ClassVar[str]

    query_processing: TextProcessingConfig[str]
