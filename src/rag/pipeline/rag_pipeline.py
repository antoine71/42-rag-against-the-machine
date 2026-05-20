import logging
from pathlib import Path

from rag.evaluating.evaluation_processor import EvaluationProcessor
from rag.indexing.bm25_repository_indexing_processor import (
    BM25RepositoryIndexingProcessor,
)
from rag.indexing.files_repository_scanner import FilesRepositoryScanner
from rag.indexing.langchain_chunking_processor import (
    LangChainChunkingProcessor,
)
from rag.llm.llm_chat_processor import LLMChatProcessor
from rag.llm.llm_manager import LLMManager
from rag.models.question import UnansweredQuestion
from rag.models.search_result import (
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from rag.retrieving.bm25_retrieving_processor import BM25RetrievingProcessor
from rag.utils.files_manager import FilesManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main class for managing various RAG-related operations.

    This class provides an interface to index data, search through datasets,
    generate answers using a RAG model, and evaluate overall performance.
    """

    def __init__(self) -> None:
        self._model = LLMManager("Qwen/Qwen3-0.6B")
        self._files_manager = FilesManager()

    def index(
        self,
        max_chunk_size: int = 2000,
        repository: str = "data/raw/vllm-0.10.1",
        save_directory: str = "data/processed/",
    ) -> None:
        """Index the dataset.

        Prints a message indicating that indexing is in progress.
        """
        repository_scanner = FilesRepositoryScanner(repository)
        files = repository_scanner.list_files()
        chunking_processor = LangChainChunkingProcessor(
            max_chunk_size, files, self._model.tokenize, self._model.tokenizer
        )
        chunks = chunking_processor.split()
        indexing_processor = BM25RepositoryIndexingProcessor(
            chunks, self._model.tokenize_batch, self._model.get_vocab()
        )
        indexing_processor.index_corpus(save_directory)
        logger.info(
            f"Ingestion complete! Indices saved under {save_directory}"
        )

    def answer(self, query: str, k: int) -> None:
        """Answer questions using the RAG model.

        Prints a message indicating that answering is in progress.
        """
        question = UnansweredQuestion(question=query)
        retriever = BM25RetrievingProcessor(
            self._model.tokenize_batch, self._model.get_vocab()
        )
        results = retriever.retrieve([question], k)

    def answer_dataset(
        self,
        student_search_result_path="data/output/search_results/dataset_docs_public.json",
        save_directory="data/output/search_result_and_answer",
    ) -> None:
        """Generate answers for a dataset.

        Prints a message indicating that answering a dataset is in progress.
        """
        search_results = self._files_manager.load_search_results(
            student_search_result_path
        )
        chat_processor = LLMChatProcessor(self._files_manager)
        answers = StudentSearchResultsAndAnswer(
            search_results=[], k=search_results.k
        )
        for result in search_results.search_results:
            output = chat_processor.answer_single_query(
                result.question, result.retrieved_sources
            )
            answers.search_results.append(
                MinimalAnswer(answer=output, **result.model_dump())
            )
        save_file_name = Path(student_search_result_path).name
        save_file_path_obj = Path(save_directory) / save_file_name
        self._files_manager.save_results(answers, str(save_file_path_obj))
        logger.info(
            f"Saved student_search_results_and_answers to '{save_file_path_obj}'"
        )

    def search(self) -> None:
        """Search the dataset.

        Prints a message indicating that searching is in progress.
        """

        print("search")

    def search_dataset(
        self,
        dataset_path: str = "datasets_public/public/UnansweredQuestions/dataset_code_public.json",
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Search a dataset for specific items.

        Prints a message indicating that searching a dataset is in progress.
        """
        dataset = self._files_manager.load_dataset(
            dataset_path, "unanswered_questions"
        )
        retriever = BM25RetrievingProcessor(
            self._model.tokenize_batch, self._model.get_vocab()
        )
        results = retriever.retrieve(dataset.rag_questions, k)
        save_file_name = Path(dataset_path).name
        save_file_path_obj = Path(save_directory) / save_file_name
        self._files_manager.save_results(results, str(save_file_path_obj))
        logger.info(f"Saved student_search_results to '{save_file_path_obj}'")

    def evaluate(
        self,
        student_answer_path: str = "data/output/search_results/dataset_code_public.json",
        dataset_path: str = "datasets_public/public/AnsweredQuestions/dataset_code_public.json",
    ) -> None:
        """Evaluate the performance of the RAG pipeline.

        Prints a message indicating that evaluation is in progress.
        """
        evaluator = EvaluationProcessor(self._files_manager)
        metrics = evaluator.evaluate(student_answer_path, dataset_path)
        print(metrics)
