import logging
import uuid
from pathlib import Path

from pydantic import ValidationError

from rag.config.llm import LLMConfig
from rag.evaluation.evaluation_processor import EvaluationProcessor
from rag.exceptions import RAGException
from rag.indexing.files_repository_scanner import FilesRepositoryScanner
from rag.indexing.indexing_manager import IndexingManager
from rag.indexing.indexing_processor_factory import IndexingProcessorFactory
from rag.indexing.langchain_chunking_processor import (
    LangChainChunkingProcessor,
)
from rag.llm.llm_chat_processor import LLMChatProcessor
from rag.llm.llm_manager import LLMManager
from rag.models.cli_models import (
    Answer,
    AnswerDataset,
    Evaluate,
    Index,
    Search,
    SearchDataset,
)
from rag.models.indexing_method import IndexingMethod
from rag.models.question import UnansweredQuestion
from rag.models.search_result import (
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from rag.retrieving.retrieving_manager import RetrievingManager
from rag.retrieving.retrieving_processor_factory import (
    RetrievingProcessorFactory,
)
from rag.tui import TUI
from rag.utils.files_manager import FilesManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self) -> None:
        self._files_manager = FilesManager()
        self._tui = TUI()

    def index(
        self,
        max_chunk_size: int = 2000,
        repository: str = "data/raw/vllm-0.10.1",
        save_directory: str = "data/processed/",
        indexing_method: IndexingMethod = IndexingMethod.BM25,
    ) -> None:
        try:
            Index(
                max_chunk_size=max_chunk_size,
                repository=Path(repository),
                save_directory=Path(save_directory),
                indexing_method=indexing_method,
            )
        except ValidationError as e:
            raise RAGException(f"Command line argument error:\n{e}") from e
        files = FilesRepositoryScanner(repository).list_files()
        self._tui.print(
            f"Found {len(files)} py and md files from '{repository}'."
        )
        logger.debug(
            f"Chunking '{repository}' into chunks of size '{max_chunk_size}'."
        )

        chunks = LangChainChunkingProcessor(max_chunk_size, files).split()
        self._tui.print(f"Split {len(files)} files into {len(chunks)} chunks.")

        indexing_processors = IndexingProcessorFactory.create(
            indexing_method, chunks, self._tui
        )
        indexing_manager = IndexingManager(indexing_processors)
        indexing_manager.process(save_directory)
        self._tui.print(
            f"Ingestion complete! Indices saved under {save_directory}"
        )

    def search(
        self,
        query: str,
        index_directory: str = "data/processed",
        retrieving_method: IndexingMethod = IndexingMethod.BM25,
        k: int = 10,
    ) -> None:
        try:
            Search(
                query=query,
                index_directory=Path(index_directory),
                retrieving_method=retrieving_method,
                k=k,
            )
        except ValidationError as e:
            raise RAGException(f"Command line argument error:\n{e}") from e
        retrievers = RetrievingProcessorFactory.create(
            retrieving_method, index_directory, self._tui
        )
        retrieving_manager = RetrievingManager(retrievers)
        question = UnansweredQuestion(
            question_id=str(uuid.uuid4()), question=query
        )
        results = retrieving_manager.process([question], k)
        self._tui.print(results.model_dump_json(indent=4))

    def search_dataset(
        self,
        dataset_path: str = (
            "datasets_public/public/UnansweredQuestions"
            "/dataset_code_public.json"
        ),
        index_directory: str = "data/processed",
        save_directory: str = "data/output/search_results",
        retrieving_method: IndexingMethod = IndexingMethod.BM25,
        k: int = 10,
    ) -> None:
        try:
            SearchDataset(
                dataset_path=Path(dataset_path),
                index_directory=Path(index_directory),
                save_directory=Path(save_directory),
                retrieving_method=retrieving_method,
                k=k,
            )
        except ValidationError as e:
            raise RAGException(f"Command line argument error:\n{e}") from e
        dataset = self._files_manager.load_dataset(
            dataset_path, "unanswered_questions"
        )
        retrievers = RetrievingProcessorFactory.create(
            retrieving_method, index_directory, self._tui
        )
        retrieving_manager = RetrievingManager(retrievers)
        results = retrieving_manager.process(dataset.rag_questions, k)
        save_file_name = Path(dataset_path).name
        save_file_path_obj = Path(save_directory) / save_file_name
        self._files_manager.save_results(results, str(save_file_path_obj))
        self._tui.print(
            f"Saved student_search_results to '{save_file_path_obj}'"
        )

    def answer(
        self,
        query: str,
        index_directory: str = "data/processed",
        retrieving_method: IndexingMethod = IndexingMethod.BM25,
        k: int = 10,
    ) -> None:
        try:
            Answer(
                query=query,
                index_directory=Path(index_directory),
                retrieving_method=retrieving_method,
                k=k,
            )
        except ValidationError as e:
            raise RAGException(f"Command line argument error:\n{e}") from e
        retrievers = RetrievingProcessorFactory.create(
            retrieving_method, index_directory, self._tui
        )
        retrieving_manager = RetrievingManager(retrievers)
        question = UnansweredQuestion(
            question_id=str(uuid.uuid4()), question=query
        )
        results = retrieving_manager.process([question], k)
        llm_manager = LLMManager(self._tui, LLMConfig())
        chat_processor = LLMChatProcessor(self._files_manager, k, llm_manager)
        outputs = chat_processor.answer_queries(list(results.search_results))
        answers = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(
                    answer=outputs[0],
                    **results.search_results[0].model_dump(),
                )
            ],
            k=k,
        )
        self._tui.print(answers.model_dump_json(indent=4))

    def answer_dataset(
        self,
        student_search_result_path: str = (
            "data/output/search_results/dataset_code_public.json"
        ),
        save_directory: str = "data/output/search_result_and_answer",
        k: int = 10,
    ) -> None:
        try:
            AnswerDataset(
                student_search_result_path=Path(student_search_result_path),
                save_directory=Path(save_directory),
                k=k,
            )
        except ValidationError as e:
            raise RAGException(f"Command line argument error:\n{e}") from e
        search_results = self._files_manager.load_search_results(
            student_search_result_path
        )
        llm_manager = LLMManager(self._tui, LLMConfig())
        chat_processor = LLMChatProcessor(self._files_manager, k, llm_manager)
        outputs = chat_processor.answer_queries(
            list(search_results.search_results)
        )
        answers = StudentSearchResultsAndAnswer(
            search_results=[
                MinimalAnswer(answer=output, **result.model_dump())
                for result, output in zip(
                    search_results.search_results, outputs
                )
            ],
            k=search_results.k,
        )
        save_file_name = Path(student_search_result_path).name
        save_file_path_obj = Path(save_directory) / save_file_name
        self._files_manager.save_results(answers, str(save_file_path_obj))
        self._tui.print(
            "Saved student_search_results_and_answers to "
            f"'{save_file_path_obj}'"
        )

    def evaluate(
        self,
        student_answer_path: str = (
            "data/output/search_results/dataset_code_public.json"
        ),
        dataset_path: str = (
            "datasets_public/public/AnsweredQuestions/dataset_code_public.json"
        ),
    ) -> None:
        try:
            Evaluate(
                student_answer_path=Path(student_answer_path),
                dataset_path=Path(dataset_path),
            )
        except ValidationError as e:
            raise RAGException(f"Command line argument error\n{e}") from e
        evaluator = EvaluationProcessor(self._files_manager)
        metrics = evaluator.evaluate(student_answer_path, dataset_path)
        self._tui.print_evaluation_results(
            metrics, dataset_path, student_answer_path
        )
