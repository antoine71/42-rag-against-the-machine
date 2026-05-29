from typing import Annotated

from pydantic import BaseModel, DirectoryPath, Field, FilePath

from rag.models.indexing_method import IndexingMethod


class Index(BaseModel):
    max_chunk_size: Annotated[int, Field(gt=0)]
    repository: DirectoryPath
    save_directory: DirectoryPath
    indexing_method: IndexingMethod


class Search(BaseModel):
    query: str
    index_directory: DirectoryPath
    retrieving_method: IndexingMethod
    k: Annotated[int, Field(gt=0)]


class SearchDataset(BaseModel):
    dataset_path: FilePath
    index_directory: DirectoryPath
    save_directory: DirectoryPath
    retrieving_method: IndexingMethod
    k: Annotated[int, Field(gt=0)]


class Answer(BaseModel):
    query: str
    index_directory: DirectoryPath
    retrieving_method: IndexingMethod
    k: int


class AnswerDataset(BaseModel):
    student_search_result_path: FilePath
    save_directory: DirectoryPath
    k: int


class Evaluate(BaseModel):
    student_answer_path: FilePath
    dataset_path: FilePath
