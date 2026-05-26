import itertools
import logging
import uuid
from pathlib import Path
from typing import Literal

from rag.evaluation.evaluation_processor import EvaluationProcessor
from rag.exceptions import RAGException
from rag.indexing.bm25_repository_indexing_processor import (
    BM25RepositoryIndexingProcessor,
)
from rag.indexing.files_repository_scanner import FilesRepositoryScanner
from rag.indexing.langchain_chunking_processor import (
    LangChainChunkingProcessor,
)
from rag.indexing.vector_embedding_processor import VectorEmbeddingProcessor
from rag.llm.llm_chat_processor import LLMChatProcessor
from rag.models.question import UnansweredQuestion
from rag.models.search_result import (
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from rag.retrieving.bm25_retrieving_processor import BM25RetrievingProcessor
from rag.retrieving.vector_retrieving_processor import (
    VectorRetrievingProcessor,
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
        indexing_method: str = "bm25",
    ) -> None:
        repository_scanner = FilesRepositoryScanner(repository)
        files = repository_scanner.list_files()
        self._tui.print(
            f"Found {len(files)} py and md files from '{repository}'."
        )

        logger.debug(
            f"Chunking '{repository}' into chunks of size '{max_chunk_size}'."
        )
        chunking_processor = LangChainChunkingProcessor(max_chunk_size, files)
        chunks = chunking_processor.split()
        self._tui.print(f"Split {len(files)} files into {len(chunks)} chunks.")

        if indexing_method == "bm25":
            indexing_processor = BM25RepositoryIndexingProcessor(chunks)
        elif indexing_method == "vector":
            indexing_processor = VectorEmbeddingProcessor(chunks)
        else:
            raise RAGException(f"Invalid indexing method: '{indexing_method}'")
        indexing_processor.index_corpus(save_directory)
        self._tui.print(
            f"Ingestion complete! Indices saved under {save_directory}"
        )

    def test_vector(
        self,
        max_chunk_size: int = 2000,
        repository: str = "data/raw/vllm-0.10.1",
        save_directory: str = "data/processed/",
        indexing_method: str = "bm25",
        query="What are the differences between mm_kwargs and tok_kwargs when using the _call_hf_processor method in vLLM multimodal processing?",
    ) -> None:
        repository_scanner = FilesRepositoryScanner(repository)
        files = repository_scanner.list_files()
        self._tui.print(
            f"Found {len(files)} py and md files from '{repository}'."
        )

        chunking_processor = LangChainChunkingProcessor(max_chunk_size, files)
        chunks = chunking_processor.split()
        self._tui.print(f"Split {len(files)} files into {len(chunks)} chunks.")

        indexing_processor = VectorEmbeddingProcessor(chunks)
        question = UnansweredQuestion(
            question_id=str(uuid.uuid4()), question=query
        )
        results = indexing_processor.special_index([question], 10)
        self._tui.print(results.model_dump_json(indent=4))

    def search(self, query: str, k=10) -> None:
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
        k: int = 10,
        save_directory: str = "data/output/search_results",
        retrieving_method: Literal["bm25", "vector"] = "bm25",
    ) -> None:
        dataset = self._files_manager.load_dataset(
            dataset_path, "unanswered_questions"
        )
        if retrieving_method == "bm25":
            retriever = BM25RetrievingProcessor()
        elif retrieving_method == "vector":
            retriever = VectorRetrievingProcessor()
        else:
            raise NotImplementedError
        retriever.retrieve(dataset.rag_questions, k)
        results = retriever.retrieve(dataset.rag_questions, k)
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
            for results in itertools.batched(
                search_results.search_results, batch_size
            ):
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
