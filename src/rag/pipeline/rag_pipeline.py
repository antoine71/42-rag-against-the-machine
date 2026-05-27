import logging
import uuid
from pathlib import Path
from typing import Any

from more_itertools import batched

from rag.evaluation.evaluation_processor import EvaluationProcessor
from rag.indexing.files_repository_scanner import FilesRepositoryScanner
from rag.indexing.indexing_processor_factory import IndexingProcessorFactory
from rag.indexing.langchain_chunking_processor import (
    LangChainChunkingProcessor,
)
from rag.llm.llm_chat_processor import LLMChatProcessor
from rag.models.question import UnansweredQuestion
from rag.models.search_result import (
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from rag.retrieving.bm25_retrieving_processor import BM25RetrievingProcessor
from rag.retrieving.retrieving_manager import RetrievingManager
from rag.retrieving.retrieving_processor_factory import (
    RetrievingProcessorFactory,
)
from rag.retrieving.vector_retrieving_processor import (
    VectorRetrievingProcessor,
)
from rag.tui import TUI
from rag.utils.files_manager import FilesManager
from rag.utils.reciprocal_rank_fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self) -> None:
        self._files_manager = FilesManager()
        self._tui = TUI()

    def tune_bm25s(self):
        k1_s = [0.8, 1.2, 1.5, 2.0]
        b_s = [0.3, 0.5, 0.75, 0.9]
        for k1 in k1_s:
            for b in b_s:
                print(f"k1: {k1}, b: {b}")
                self.index(config=dict(k1=k1, b=b))
                self.search_dataset(config=dict(k1=k1, b=b))
                self.evaluate()

    def index(
        self,
        max_chunk_size: int = 2000,
        repository: str = "data/raw/vllm-0.10.1",
        save_directory: str = "data/processed/",
        indexing_method: str = "bm25",
        config: dict[str, Any] = {},
    ) -> None:
        files = FilesRepositoryScanner(repository).list_files()
        self._tui.print(
            f"Found {len(files)} py and md files from '{repository}'."
        )
        logger.debug(
            f"Chunking '{repository}' into chunks of size '{max_chunk_size}'."
        )

        chunks = LangChainChunkingProcessor(max_chunk_size, files).split()
        self._tui.print(f"Split {len(files)} files into {len(chunks)} chunks.")

        indexing_processor = IndexingProcessorFactory.create(
            indexing_method, chunks, self._tui, config
        )
        indexing_processor.index_corpus(save_directory)
        self._tui.print(
            f"Ingestion complete! Indices saved under {save_directory}"
        )

    def search(self, query: str, k: int = 10) -> None:
        retriever = VectorRetrievingProcessor()
        question = UnansweredQuestion(
            question_id=str(uuid.uuid4()), question=query
        )
        results = retriever.retrieve([question], k)
        self._tui.print(results.model_dump_json(indent=4))

    def search_dataset(
        self,
        dataset_path: str = (
            "datasets_public/public/UnansweredQuestions"
            "/dataset_docs_public.json"
        ),
        index_directory: str = "data/processed",
        save_directory: str = "data/output/search_results",
        retrieving_method: list[str] = ["bm25"],
        k: int = 10,
        config: dict[str, Any] = {},
    ) -> None:
        dataset = self._files_manager.load_dataset(
            dataset_path, "unanswered_questions"
        )
        retrievers = RetrievingProcessorFactory.create(
            retrieving_method, index_directory, config
        )
        retrieving_manager = RetrievingManager()
        for retriever in retrievers:
            retrieving_manager.add_retrieving_processor(retriever)
        results = retrieving_manager.process(
            dataset.rag_questions, k, 4, reciprocal_rank_fusion
        )
        save_file_name = Path(dataset_path).name
        save_file_path_obj = Path(save_directory) / save_file_name
        self._files_manager.save_results(results, str(save_file_path_obj))
        self._tui.print(
            f"Saved student_search_results to '{save_file_path_obj}'"
        )

    def answer(self, query: str, k: int = 5, model="Qwen/Qwen3-0.6B") -> None:
        """Answer questions using the RAG model.

        Prints a message indicating that answering is in progress.
        """
        retriever = BM25RetrievingProcessor()
        question = UnansweredQuestion(
            question_id=str(uuid.uuid4()), question=query
        )
        results = retriever.retrieve([question], k)
        chat_processor = LLMChatProcessor(self._files_manager, model, k)
        with self._tui.progress("Processing queries", 1, "query") as progress:
            outputs = chat_processor.answer_batch_query(
                (results.search_results[0],)
            )
            answers = StudentSearchResultsAndAnswer(
                search_results=[
                    MinimalAnswer(
                        answer=outputs[0],
                        **results.search_results[0].model_dump(),
                    )
                ],
                k=k,
            )
            progress.update(1)
        self._tui.print(answers.model_dump_json(indent=4))

    def answer_dataset(
        self,
        student_search_result_path=(
            "data/output/search_results/dataset_docs_public.json"
        ),
        save_directory="data/output/search_result_and_answer",
        k=5,
        model="Qwen/Qwen3-0.6B",
        batch_size=1,
    ) -> None:
        search_results = self._files_manager.load_search_results(
            student_search_result_path
        )
        chat_processor = LLMChatProcessor(self._files_manager, model, k)
        answers = StudentSearchResultsAndAnswer(
            search_results=[], k=search_results.k
        )
        total = len(search_results.search_results)
        with self._tui.progress(
            "Processing queries", total, "query"
        ) as progress:
            for results in batched(search_results.search_results, batch_size):
                outputs = chat_processor.answer_batch_query(results)
                for result, output in zip(results, outputs):
                    answers.search_results.append(
                        MinimalAnswer(answer=output, **result.model_dump())
                    )
                progress.update(len(outputs))
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
            "data/output/search_results/dataset_docs_public.json"
        ),
        dataset_path: str = (
            "datasets_public/public/AnsweredQuestions/dataset_docs_public.json"
        ),
    ) -> None:
        """Evaluate the performance of the RAG pipeline.

        Prints a message indicating that evaluation is in progress.
        """
        evaluator = EvaluationProcessor(self._files_manager)
        metrics = evaluator.evaluate(student_answer_path, dataset_path)
        self._tui.print_evaluation_results(
            metrics, dataset_path, student_answer_path
        )
