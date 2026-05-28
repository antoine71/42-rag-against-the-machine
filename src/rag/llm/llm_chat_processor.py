from rag.exceptions import RAGException
from rag.llm.llm_manager import LLMManager
from rag.llm.prompt_generator import (
    ChatTemplatePromptGenerator,
)
from rag.models.search_result import MinimalSearchResults
from rag.utils.files_manager import FilesManager


class LLMChatProcessorError(RAGException):
    pass


class LLMChatProcessor:
    def __init__(
        self, files_manager: FilesManager, k: int, llm_manager: LLMManager
    ) -> None:
        self._files_manager = files_manager
        self._llm_manager = llm_manager
        self._k = k
        self._prompt_generator = ChatTemplatePromptGenerator(
            self._files_manager, k
        )

    def answer_queries(self, input: list[MinimalSearchResults]) -> list[str]:
        prompt_messages = self._prompt_generator.generate_prompt(input)
        output = self._llm_manager.answer_queries(prompt_messages)
        return output
