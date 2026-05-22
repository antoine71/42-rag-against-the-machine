import itertools
from abc import abstractmethod
from typing import cast

from transformers import (
    BatchEncoding,
    TokenizersBackend,
)

from rag.models.minimal_source import MinimalSource
from rag.models.search_result import MinimalSearchResults
from rag.utils.files_manager import FilesManager


class PromptGenerator:
    def __init__(
        self, tokenizer: TokenizersBackend, files_manager: FilesManager, k: int
    ) -> None:
        self._tokenizer = tokenizer
        self._files_manager = files_manager
        self._k = k

    @abstractmethod
    def generate_tokens(
        self, input: tuple[MinimalSearchResults, ...]
    ) -> BatchEncoding: ...

    def _build_messages(self, input: tuple[MinimalSearchResults, ...]):
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
        return {
            "role": "system",
            "content": (
                "You are a precise and helpful assistant. Answer the user's "
                "question using ONLY the "
                "retrieved context provided below. Follow these rules strictly:\n"
                '- If the answer is not in the context, say: "I don\'t have enough information to answer that."\n'
                "- Do not use outside knowledge or make up information.\n"
                "- Keep answers concise and grounded in the provided text.\n"
                "- When possible, cite which document/source supports your answer."
            ),
        }


class SimplePromptGenerator(PromptGenerator):
    def generate_tokens(
        self, input: tuple[MinimalSearchResults, ...]
    ) -> BatchEncoding:
        messages = self._build_messages(input)
        prompts: list[str] = []
        for message in messages:
            prompts.append("\n\n".join(m["content"] for m in message))
        return cast(
            BatchEncoding,
            self._tokenizer(
                prompts,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            ),
        )


class ChatTemplatePromptGenerator(PromptGenerator):
    def generate_tokens(
        self, input: tuple[MinimalSearchResults, ...]
    ) -> BatchEncoding:
        messages = self._build_messages(input)
        return cast(
            BatchEncoding,
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            ),
        )
