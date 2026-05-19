from rag.indexing.bm25_repository_indexing_processor import (
    BM25RepositoryIndexingProcessor,
)
from rag.indexing.files_repository_scanner import FilesRepositoryScanner
from rag.indexing.langchain_chunking_processor import (
    LangChainChunkingProcessor,
)
from rag.llm.llm_manager import LLMManager
from rag.models.question import UnansweredQuestion
from rag.retrieving.bm25_retrieving_processor import BM25RetrievingProcessor
from rag.utils.files_manager import FilesManager


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
        max_chunk_size: int = 1000,
        repository: str = "fly_in",
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
        print(f"Ingestion complete! Indices saved under {save_directory}")

    def answer(self, query: str, k: int) -> None:
        """Answer questions using the RAG model.

        Prints a message indicating that answering is in progress.
        """
        question = UnansweredQuestion(question=query)
        retriever = BM25RetrievingProcessor(
            self._model.tokenize_batch, self._model.get_vocab()
        )
        results = retriever.retrieve([question], k)
        self._files_manager.save_results(
            results, "data/output/single_query.json"
        )

    def answer_dataset(self) -> None:
        """Generate answers for a dataset.

        Prints a message indicating that answering a dataset is in progress.
        """

        print("answer dataset")

    def search(self) -> None:
        """Search the dataset.

        Prints a message indicating that searching is in progress.
        """

        print("search")

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> None:
        """Search a dataset for specific items.

        Prints a message indicating that searching a dataset is in progress.
        """

    def evaluate(self) -> None:
        """Evaluate the performance of the RAG pipeline.

        Prints a message indicating that evaluation is in progress.
        """
