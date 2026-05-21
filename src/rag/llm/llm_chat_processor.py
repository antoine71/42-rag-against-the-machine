import itertools
from typing import cast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    TokenizersBackend,
)
from transformers._typing import GenerativePreTrainedModel

from rag.exceptions import RAGException
from rag.models.minimal_source import MinimalSource
from rag.models.search_result import MinimalSearchResults
from rag.utils.files_manager import FilesManager


class LLMChatProcessorError(RAGException):
    pass


class LLMChatProcessor:
    def __init__(
        self, files_manager: FilesManager, model_name: str, k: int
    ) -> None:
        self._files_manager = files_manager
        self._k = k
        try:
            self._tokenizer = cast(
                TokenizersBackend,
                AutoTokenizer.from_pretrained(
                    model_name,
                    padding_side="left",
                ),
            )
            self._model = cast(
                GenerativePreTrainedModel,
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                ),
            )
        except OSError as e:
            raise LLMChatProcessorError(
                f"Invalid model name: '{model_name}'"
            ) from e

    def answer_batch_query(
        self, input: tuple[MinimalSearchResults, ...]
    ) -> list[str]:
        messages = [
            [
                self._system_prompt,
                self._generate_user_prompt(
                    result.question, result.retrieved_sources
                ),
            ]
            for result in input
        ]
        prompt_tokens = cast(
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
        prompt_tokens = prompt_tokens.to(self._model.device)
        generated_ids = self._model.generate(
            **prompt_tokens, max_new_tokens=256
        )
        input_length = prompt_tokens["input_ids"].shape[1]
        new_tokens = generated_ids[:, input_length:]
        output = self._tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True
        )
        return output

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
