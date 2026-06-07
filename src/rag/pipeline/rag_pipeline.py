import logging
import uuid
from pathlib import Path

from rag.chunking.chunking_manager import ChunkingManager
from rag.config.chunking_config import ChunkingConfig
from rag.config.llm import LLMConfig
from rag.evaluation.evaluation_processor import EvaluationProcessor
from rag.indexing.files_repository_scanner import FilesRepositoryScanner
from rag.indexing.indexing_pipeline import IndexingPipeline
from rag.indexing.indexing_processor_factory import IndexingProcessorFactory
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
from rag.models.file_category import FileCategory
from rag.models.indexing_method import IndexingMethod
from rag.models.question import UnansweredQuestion
from rag.models.search_result import (
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from rag.retrieving.retrieving_pipeline import RetrievingPipeline
from rag.retrieving.retrieving_processor_factory import (
    RetrievingProcessorFactory,
)
from rag.tui import TUI
from rag.utils.cli_validator import validate_with
from rag.utils.files_manager import FilesManager
from rag.utils.measure import measure

logger = logging.getLogger(__name__)


class RAGPipeline:
    """The central RAG pipeline CLI entry point.

    Exposes all command entry points using Python Fire.
    """

    def __init__(self) -> None:
        """Initializes the RAGPipeline with default helper components."""
        self._files_manager = FilesManager()
        self._tui = TUI()

    @validate_with(Index)
    def index(
        self,
        max_chunk_size: int = 2000,
        repository: str = "data/raw/vllm-0.10.1",
        save_directory: str = "data/processed/",
        indexing_method: str = "bm25",
        files_category: str = "all",
    ) -> None:
        """Scans the repository, chunks files, and builds a searchable index.

        Args:
            max_chunk_size: Maximum character length of each chunk.
            repository: Path to the directory containing files to index.
            save_directory: Directory where the generated indices should be
                saved.
            indexing_method: The indexing strategy to use (
                'bm25', 'vector', or 'hybrid').
            files_category: File category to index ('code', 'documentation',
                or 'all').
        """
        self._tui.print_title("indexing", ("dataset", repository))
        files_category = FileCategory(files_category)
        indexing_method = IndexingMethod(indexing_method)
        scanner = FilesRepositoryScanner(Path(repository), self._tui)
        files, delta = measure(scanner.list_files, files_category)
        self._tui.print_task_report("discovery", delta, "files", len(files))

        chunking_config = ChunkingConfig(chunk_size=max_chunk_size)
        chunking_manager = ChunkingManager(
            chunking_config, files, self._tui, repository
        )
        chunks, delta = measure(chunking_manager.split)
        self._tui.print_task_report(
            "chunking", delta, "chunks", len(chunks), new_line_after=True
        )

        indexing_processors = IndexingProcessorFactory.create(
            indexing_method, chunks, self._tui
        )
        indexing_pipeline = IndexingPipeline(indexing_processors)
        indexing_pipeline.process(save_directory, files_category)

    @validate_with(Search)
    def search(
        self,
        query: str,
        index_directory: str = "data/processed",
        indexing_method: str = "bm25",
        files_category: str = "all",
        k: int = 10,
    ) -> None:
        """Searches the knowledge base for a single query and prints top
        results.

        Args:
            query: The search query string.
            index_directory: The directory containing index files.
            indexing_method: The retrieval strategy to use (
                'bm25', 'vector', or 'hybrid').
            files_category: File category to retrieve from ('code',
                'documentation', or 'all').
            k: Number of top results to retrieve.
        """
        files_category = FileCategory(files_category)
        self._tui.print_title("search")
        retrievers = RetrievingProcessorFactory.create(
            IndexingMethod(indexing_method),
            index_directory,
            self._tui,
        )
        retrieving_pipeline = RetrievingPipeline(retrievers, self._tui)
        question = UnansweredQuestion(
            question_id=str(uuid.uuid4()), question=query
        )
        results = retrieving_pipeline.process(
            [question], k, FileCategory(files_category)
        )
        self._tui.print("Ouput:\n")
        self._tui.print(results.model_dump_json(indent=4))

    @validate_with(SearchDataset)
    def search_dataset(
        self,
        dataset_path: str = (
            "datasets_public/public/UnansweredQuestions"
            "/dataset_docs_public.json"
        ),
        index_directory: str = "data/processed",
        save_directory: str = "data/output/search_results",
        indexing_method: str = "bm25",
        files_category: str = "all",
        k: int = 10,
    ) -> None:
        """Processes queries from a JSON dataset and saves search results.

        Args:
            dataset_path: Path to the JSON dataset containing unanswered
                questions.
            index_directory: The directory containing index files.
            save_directory: The directory where search results should be
                saved.
            indexing_method: The retrieval strategy to use (
                'bm25', 'vector', or 'hybrid').
            files_category: File category to retrieve from ('code',
                'documentation', or 'all').
            k: Number of top results to retrieve per query.
        """
        indexing_method = IndexingMethod(indexing_method)
        files_category = FileCategory(files_category)

        self._tui.print_title("search dataset", ("dataset", dataset_path))
        dataset = self._files_manager.load_dataset(
            dataset_path, "unanswered_questions"
        )
        retrievers = RetrievingProcessorFactory.create(
            indexing_method,
            index_directory,
            self._tui,
        )
        retrieving_pipeline = RetrievingPipeline(retrievers, self._tui)
        results = retrieving_pipeline.process(
            dataset.rag_questions, k, files_category
        )
        save_file_name = Path(dataset_path).name
        save_file_path_obj = Path(save_directory) / save_file_name
        self._files_manager.save_results(results, str(save_file_path_obj))
        self._tui.print(f"Ouput:\t{save_file_path_obj}\n")

    @validate_with(Answer)
    def answer(
        self,
        query: str,
        index_directory: str = "data/processed",
        indexing_method: str = "bm25",
        files_category: str = "all",
        k: int = 10,
    ) -> None:
        """Retrieves context and generates a natural answer for a single query.

        Args:
            query: The user query string.
            index_directory: The directory containing index files.
            indexing_method: The retrieval strategy to use (
                'bm25', 'vector', or 'hybrid').
            files_category: File category to retrieve from ('code',
                'documentation', or 'all').
            k: Number of top results to retrieve as context.
        """
        self._tui.print_title("answer")
        retrievers = RetrievingProcessorFactory.create(
            IndexingMethod(indexing_method),
            index_directory,
            self._tui,
        )
        retrieving_pipeline = RetrievingPipeline(retrievers, self._tui)
        question = UnansweredQuestion(
            question_id=str(uuid.uuid4()), question=query
        )
        results = retrieving_pipeline.process(
            [question], k, FileCategory(files_category)
        )
        llm_manager = LLMManager(self._tui, LLMConfig())
        chat_processor = LLMChatProcessor(
            self._files_manager, k, llm_manager, self._tui
        )
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

    @validate_with(AnswerDataset)
    def answer_dataset(
        self,
        student_search_result_path: str = (
            "data/output/search_results/dataset_docs_public.json"
        ),
        save_directory: str = "data/output/search_result_and_answer",
        k: int = 10,
    ) -> None:
        """Generates answers for a search dataset file and saves them.

        Args:
            student_search_result_path: Path to the search results JSON file.
            save_directory: The directory where final answers should be saved.
            k: Number of context sources to include.
        """
        self._tui.print_title(
            "answer dataset", ("dataset", student_search_result_path)
        )
        search_results = self._files_manager.load_search_results(
            student_search_result_path
        )
        llm_manager = LLMManager(self._tui, LLMConfig())
        chat_processor = LLMChatProcessor(
            self._files_manager, k, llm_manager, self._tui
        )
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
        self._tui.print(f"Ouput:\t{save_file_path_obj}\n")

    @validate_with(Evaluate)
    def evaluate(
        self,
        student_answer_path: str = (
            "data/output/search_results/dataset_docs_public.json"
        ),
        dataset_path: str = (
            "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
        ),
    ) -> None:
        """Evaluate student search results against the answered questions
        ground truth.

        Args:
            student_answer_path: Path to the student search results JSON file.
            dataset_path: Path to the ground truth answered questions dataset
                JSON file.
        """
        self._tui.print_title(
            "evaluate",
            ("dataset", student_answer_path),
            ("reference", dataset_path),
        )
        evaluator = EvaluationProcessor(self._files_manager)
        metrics = evaluator.evaluate(student_answer_path, dataset_path)
        self._tui.print_evaluation_results(metrics)
