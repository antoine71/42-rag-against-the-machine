from typing import Annotated

from pydantic import BaseModel, DirectoryPath, Field, FilePath

from rag.models.indexing_method import IndexingMethod


class Index(BaseModel):
    """Validation model for the index CLI command."""

    max_chunk_size: Annotated[int, Field(gt=0)]
    repository: DirectoryPath
    save_directory: DirectoryPath
    indexing_method: IndexingMethod


class Search(BaseModel):
    """Validation model for the search CLI command."""

    query: str
    index_directory: DirectoryPath
    retrieving_method: IndexingMethod
    k: Annotated[int, Field(gt=0)]


class SearchDataset(BaseModel):
    """Validation model for the search_dataset CLI command."""

    dataset_path: FilePath
    index_directory: DirectoryPath
    save_directory: DirectoryPath
    retrieving_method: IndexingMethod
    k: Annotated[int, Field(gt=0)]


class Answer(BaseModel):
    """Validation model for the answer CLI command."""

    query: str
    index_directory: DirectoryPath
    retrieving_method: IndexingMethod
    k: int


class AnswerDataset(BaseModel):
    """Validation model for the answer_dataset CLI command."""

    student_search_result_path: FilePath
    save_directory: DirectoryPath
    k: int


class Evaluate(BaseModel):
    """Validation model for the evaluate CLI command."""

    student_answer_path: FilePath
    dataset_path: FilePath
