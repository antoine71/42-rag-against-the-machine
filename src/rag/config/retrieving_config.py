from abc import ABC
from typing import ClassVar

from pydantic_settings import BaseSettings

from rag.config.text_processing import TextProcessingConfig
from rag.models.file_category import FileCategory


class RetrievingConfig(ABC, BaseSettings):
    TYPE: ClassVar[str]

    query_processing: TextProcessingConfig[FileCategory, str]
