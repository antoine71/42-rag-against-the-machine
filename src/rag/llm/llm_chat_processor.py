from rag.exceptions import RAGException
from rag.llm.llm_manager import LLMManager
from rag.llm.prompt_generator import (
    ChatTemplatePromptGenerator,
)
from rag.models.search_result import MinimalSearchResults
from rag.utils.files_manager import FilesManager


class LLMChatProcessorError(RAGException):
    """Exception raised when an error occurs during LLM chat processing."""


class LLMChatProcessor:
    """Processor that coordinates prompt generation and LLM query answering."""

    def __init__(
        self, files_manager: FilesManager, k: int, llm_manager: LLMManager
    ) -> None:
        """Initializes the LLMChatProcessor.

        Args:
            files_manager: Utility to load code chunks from disk.
            k: The number of retrieved sources to use in context.
            llm_manager: The LLMManager instance to run inference.
        """
        self._files_manager = files_manager
        self._llm_manager = llm_manager
        self._k = k
        self._prompt_generator = ChatTemplatePromptGenerator(
            self._files_manager, k
        )

    def answer_queries(self, input: list[MinimalSearchResults]) -> list[str]:
        """Formats the search results into LLM prompts and generates query
        responses.

        Args:
            input: A list of MinimalSearchResults containing queries and
                retrieved sources.

        Returns:
            A list of natural language answer strings.
        """
        prompt_messages = self._prompt_generator.generate_prompt(input)
        output = self._llm_manager.answer_queries(prompt_messages)
        return output
