import itertools

from rag.models.minimal_source import MinimalSource
from rag.models.search_result import MinimalSearchResults
from rag.utils.files_manager import FilesManager


class ChatTemplatePromptGenerator:
    """Generates structured chat templates with retrieved contexts for the
    LLM.
    """

    def __init__(self, files_manager: FilesManager, k: int) -> None:
        """Initializes the ChatTemplatePromptGenerator.

        Args:
            files_manager: Utility to load code chunks from disk.
            k: The number of retrieved sources to use in context.
        """
        self._files_manager = files_manager
        self._k = k

    def generate_prompt(
        self, input: list[MinimalSearchResults]
    ) -> list[list[dict[str, str]]]:
        """Creates the batch of structured chat prompt lists for the LLM.

        Args:
            input: A list of MinimalSearchResults containing queries and
                retrieved sources.

        Returns:
            A list of chat messages (list of role/content dicts) for each
                query.
        """
        messages = self._build_messages(input)
        return messages

    def _build_messages(
        self, input: list[MinimalSearchResults]
    ) -> list[list[dict[str, str]]]:
        """Constructs system and user message pairs for each query search
        result.

        Args:
            input: A list of MinimalSearchResults objects.

        Returns:
            A list of system/user role messages.
        """
        return [
            [
                self._system_prompt,
                self._generate_user_prompt(
                    result.question, result.retrieved_sources
                ),
            ]
            for result in input
        ]

    def _prepare_context(self, sources: list[MinimalSource]) -> str:
        """Loads and formats the top-k retrieved source contexts.

        Args:
            sources: A list of retrieved MinimalSource objects.

        Returns:
            A single formatted context string.
        """
        chunks = [
            (source.file_path, self._files_manager.load_chunk(source))
            for source in itertools.islice(sources, self._k)
        ]
        formatted_sources = []
        for i, (file_path, content) in enumerate(chunks, start=1):
            formatted_sources.append(
                f"[Source {i}] File: {file_path}\nContent: {content}\n"
            )
        return "\n".join(formatted_sources)

    def _generate_user_prompt(
        self, query: str, sources: list[MinimalSource]
    ) -> dict[str, str]:
        """Generates the user message dictionary containing query and
        formatted context.

        Args:
            query: The user query string.
            sources: A list of retrieved MinimalSource objects.

        Returns:
            A dictionary with 'role': 'user' and 'content' prompt string.
        """
        return {
            "role": "user",
            "content": (
                "Retrieved Context:\n"
                "---\n"
                f"{self._prepare_context(sources)}"
                "---\n\n"
                f"Question: {query}\n\n"
                "Answer based only on the retrieved context above."
            ),
        }

    @property
    def _system_prompt(self) -> dict[str, str]:
        """Gets the system instructions prompt dictionary for the
        assistant's behavior.

        Returns:
            A dictionary with 'role': 'system' and 'content' instruction
                string.
        """
        return {
            "role": "system",
            "content": (
                "You are a precise and helpful assistant. Answer the user's "
                "question using ONLY the retrieved context provided below. "
                "Follow these rules strictly:\n"
                "- If the answer is not in the context, say: \"I don't have "
                'enough information to answer that."\n'
                "- Do not use outside knowledge or make up information.\n"
                "- Keep answers concise and grounded in the provided text.\n"
                "- When possible, cite which document/source supports your "
                "answer."
            ),
        }
