from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, DirectoryPath, Field, FilePath

from rag.models.file_category import FileCategory
from rag.models.indexing_method import IndexingMethod


def validate_save_dir(value: str) -> str:
    path = Path(value)
    if not path.exists():
        path.mkdir(parents=True)
    return value


class Index(BaseModel):
    """Validation model for the index CLI command."""

    max_chunk_size: Annotated[int, Field(gt=0)]
    repository: DirectoryPath
    save_directory: Annotated[
        DirectoryPath, BeforeValidator(validate_save_dir)
    ]
    indexing_method: IndexingMethod
    files_category: FileCategory


class Search(BaseModel):
    """Validation model for the search CLI command."""

    query: str
    index_directory: DirectoryPath
    indexing_method: IndexingMethod
    file_type: FileCategory
    k: Annotated[int, Field(gt=0)]


class SearchDataset(BaseModel):
    """Validation model for the search_dataset CLI command."""

    dataset_path: FilePath
    index_directory: DirectoryPath
    save_directory: Annotated[
        DirectoryPath, BeforeValidator(validate_save_dir)
    ]
    indexing_method: IndexingMethod
    files_category: FileCategory
    k: Annotated[int, Field(gt=0)]


class Answer(BaseModel):
    """Validation model for the answer CLI command."""

    query: str
    index_directory: DirectoryPath
    indexing_method: IndexingMethod
    files_category: FileCategory
    k: int


class AnswerDataset(BaseModel):
    """Validation model for the answer_dataset CLI command."""

    student_search_result_path: FilePath
    save_directory: Annotated[
        DirectoryPath, BeforeValidator(validate_save_dir)
    ]
    k: int


class Evaluate(BaseModel):
    """Validation model for the evaluate CLI command."""

    student_answer_path: FilePath
    dataset_path: FilePath
